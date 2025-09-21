from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from flax import nnx
from helpers.config import modelConfig, trainConfig
import optax
import argparse
from helpers.preprocess_data_torch import load_latents
from tqdm import tqdm
import os
from glob import glob
from copy import deepcopy
import optax
import orbax.checkpoint as ocp
import time
import jax_dataloader as jdl
import json


class Diffusion():
    def __init__(self, variance_min, variance_max, steps):
        self.steps = steps
        self.variance_min = variance_min
        self.variance_max = variance_max
        variances = jnp.linspace(variance_max, variance_min, steps)
        alpha = 1 - variances
        self.alpha_bar = jnp.cumprod(alpha)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]



@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, x, t, y):
  def loss_fn(model):
    y_pred = model(x, t)  
    return optax.losses.squared_error(y_pred, y).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)  # in-place updates

  return loss


import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        json_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(json_record)

def setup_logging(experiment_path):
    log_path = os.path.join(experiment_path, "training.log")
    json_log_path = os.path.join(experiment_path, "training.json.log")

    print(f"Logs will be saved to: {log_path} and {json_log_path}")

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler for traditional logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Create a file handler for JSON logging
    json_file_handler = logging.FileHandler(json_log_path)
    json_file_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)


from helpers.porting import save_checkpoint, restore_checkpoint

def main(args):
    start_time = time.time()
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        gpu_device = None
    assert gpu_device != None
    assert args.model_config == "DiT-S", "Currently, the only model config implemented is DiT-S."

    if args.resume:
        experiment_path = args.resume
        models_dir = os.path.join(experiment_path, "models")
    else:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_number = len(glob(f"{args.results_dir}/*"))
        experiment_path = os.path.join(args.results_dir, f"experiment-v{experiment_number}")
        models_dir = os.path.join(experiment_path, "models")
        os.makedirs(models_dir)

    # Set up logging and log initial experiment information.
    setup_logging(experiment_path)
    logging.info(f"Starting experiment at {experiment_path}")
    logging.info(f"Arguments: {args}")

    train_latents, train_labels = load_latents("./data/train_latent")
    logging.info(f"Training latents shape: {train_latents.shape}")

    trainconfig = trainConfig()
    modelconfig = modelConfig()
    DiTmodel = DiffusionTransformer(modelconfig)
    opt = optax.adamw(learning_rate=1e-3)
    optimizer = nnx.Optimizer(DiTmodel, opt, wrt=nnx.Param)

    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
    mngr = ocp.CheckpointManager(models_dir, options=options)

    if args.resume:
        extra_params = restore_checkpoint(mngr, DiTmodel)
        start_epoch = extra_params['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Log shape of the model
    logging.info(jax.tree.map(lambda x: str(type(x)), nnx.split(DiTmodel)[1]))  # Initial state

    diffusion = Diffusion(trainconfig.linear_variance_min, trainconfig.linear_variance_max, trainconfig.tmax)

    train_latents = jdl.ArrayDataset(train_latents, train_labels)
    train_latents = jdl.DataLoader(
        train_latents, 
        backend='jax', 
        batch_size=trainconfig.batch_size, 
        shuffle=True,
        drop_last=False,
    )
    train_steps = 0
    random_key = jax.random.PRNGKey(0)
    for epoch in range(start_epoch, trainconfig.epochs):
        logging.info(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        running_loss = 0.0
        for i, (batch, labels) in enumerate(tqdm(train_latents)):

            t = jax.random.randint(random_key, (batch.shape[0],), 0, 1000)

            alpha_bar_t = diffusion.get_alpha_bar(t)[:, None, None, None]
            noise = jnp.sqrt(1-alpha_bar_t) * jax.random.normal(key=jax.random.PRNGKey(0), shape=batch.shape)
            
            batch *= .18215 
            noisy_batch = batch * jnp.sqrt(alpha_bar_t) + noise

            loss = train_step(DiTmodel, optimizer, noisy_batch, t, noise)
            running_loss += loss.item()
            if (i + 1) % trainconfig.log_frequency == 0:
                avg_loss = running_loss / trainconfig.log_frequency
                logging.info(f"Epoch {epoch} | Step {i+1} | Loss: {avg_loss:.4f}")
                running_loss = 0.0
                break
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} finished in {epoch_time:.2f} seconds")
        if epoch % trainconfig.ckpt_frequency == 0:
            save_checkpoint(mngr, DiTmodel, optimizer, epoch, trainconfig, args)

        logging.info(f"Training completed in {time.time() - start_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    parser.add_argument("--results_dir", "-r", default="./results", help="Directory to save results to.")
    parser.add_argument("--resume", "-re", help="Path to the experiment directory to resume training from.")
    
    args = parser.parse_args()
    main(args)
