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
from helpers.checkpoint import save_checkpoint, restore_checkpoint
from helpers.logging_utils import setup_logging
from helpers.diffusion import Diffusion
import logging
import json


@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, x, t, y):
  def loss_fn(model):
    y_pred = model(x, t)  
    return optax.losses.squared_error(y_pred, y).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)  # in-place updates

  return loss

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
        models_dir = os.path.abspath(os.path.join(experiment_path, "models"))
    else:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_number = len(glob(f"{args.results_dir}/*"))
        experiment_path = os.path.join(args.results_dir, f"experiment-v{experiment_number}")
        models_dir = os.path.abspath(os.path.join(experiment_path, "models"))
        os.makedirs(models_dir)

    # Set up logging and log initial experiment information.
    setup_logging(experiment_path)
    logging.info(f"Starting experiment at {experiment_path}")
    logging.info(f"Arguments: {args}")

    train_latents, train_labels = load_latents("./data/train_latent_5_samples_per_image")
    logging.info(f"Training latents shape: {train_latents.shape}")

    trainconfig = trainConfig()
    modelconfig = modelConfig()

    options = ocp.CheckpointManagerOptions(max_to_keep=5)
    mngr = ocp.CheckpointManager(models_dir, options=options)

    if args.resume:
        DiTmodel, extra_params = restore_checkpoint(models_dir, modelconfig, trainconfig, gpu_device)
        start_epoch = extra_params['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        DiTmodel = DiffusionTransformer(modelconfig)
        start_epoch = 0
    opt = optax.adamw(learning_rate=trainconfig.learning_rate)
    optimizer = nnx.Optimizer(DiTmodel, opt, wrt=nnx.Param)

    # Log shape of the model
    # logging.info(jax.tree.map(lambda x: str(type(x)), nnx.split(DiTmodel)[1]))  # Initial state

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

            iter_key, random_key = jax.random.split(random_key)
            t_key, noise_key = jax.random.split(iter_key)

            t = jax.random.randint(t_key, (batch.shape[0],), 0, 1000)

            alpha_bar_t = diffusion.get_alpha_bar(t)[:, None, None, None]
            
            epsilon = jax.random.normal(key=noise_key, shape=batch.shape)
            
            # batch *= .18215 
            noisy_batch = batch * jnp.sqrt(alpha_bar_t) + jnp.sqrt(1 - alpha_bar_t) * epsilon

            loss = train_step(DiTmodel, optimizer, noisy_batch, t, epsilon)
            running_loss += loss.item()
            if (i + 1) % trainconfig.log_frequency == 0:
                avg_loss = running_loss / trainconfig.log_frequency
                logging.info(f"Epoch {epoch} | Step {i+1} | Loss: {avg_loss:.4f}")
                running_loss = 0.0
                
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} finished in {epoch_time:.2f} seconds")
        if (epoch + 1) % trainconfig.ckpt_frequency == 0:
            save_checkpoint(mngr, DiTmodel, optimizer, epoch, trainconfig, args)
        logging.info(f"Total time passed is: {time.time() - start_time} seconds")
    logging.info(f"Training completed in {time.time() - start_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    parser.add_argument("--results_dir", "-r", default="./results", help="Directory to save results to.")
    parser.add_argument("--resume", "-re", help="Path to the experiment directory to resume training from.")
    
    args = parser.parse_args()
    main(args)
