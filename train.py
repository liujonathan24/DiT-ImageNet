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

def main(args):
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        gpu_device = None
    assert gpu_device != None
    assert args.model_config == "DiT-S", "Currently, the only model config implemented is DiT-S."

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_number = len(glob(f"{args.results_dir}/*"))
    experiment_path = os.path.join(args.results_dir, f"experiment-v{experiment_number}")
    models_dir = os.path.join(experiment_path, "models")
    os.makedirs(experiment_path)


    train_latents, train_labels = load_latents("./data/train_latent")
    print(type(train_latents))
    print(train_latents.shape)



    test_latents, test_labels = load_latents("./data/test_latent")

    trainconfig = trainConfig()
    modelconfig = modelConfig()
    DiTmodel = DiffusionTransformer(modelconfig)
    opt = optax.adamw(learning_rate=1e-3)
    optimizer = nnx.Optimizer(DiTmodel, opt, wrt=nnx.Param)

    diffusion = Diffusion(trainconfig.linear_variance_min, trainconfig.linear_variance_max, trainconfig.tmax)

    train_latents = jdl.ArrayDataset(train_latents, train_labels)
    train_latents = jdl.DataLoader(
            train_latents, # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
        backend='jax', # Use 'jax' backend for loading data
        batch_size= trainconfig.batch_size, # Batch size
        shuffle=True, # Shuffle the dataloader every iteration or not
        drop_last=False, # Drop the last batch or not

    )


    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    random_key = jax.random.PRNGKey(0)

    for epoch in range(trainconfig.epochs):
        print(f"Starting epoch {epoch}.")
        for i, batch, labels in enumerate(tqdm(train_latents)): # maybe convert to dataloader again?
            

            t = jax.random.randint(random_key, (batch.shape[0],), 0, 1000)

            # noise = 
            alpha_bar_t = diffusion.get_alpha_bar(t)[:, None, None, None]
            noise = jnp.sqrt(1-alpha_bar_t) * jax.random.normal(key=jax.random.PRNGKey(0), shape=batch.shape)
            
            batch *= .18215 
            noisy_batch = batch * jnp.sqrt(alpha_bar_t) + noise

            loss = train_step(DiTmodel, optimizer, noisy_batch, t, noise)
            if i % 100 == 0:
                running_loss = loss
        print(f"loss on epoch {epoch} is {loss}")
        if epoch % trainconfig.ckpt_frequency == 0:
            # Bundle states into checkpoint and save for later EMA.
            model_state = nnx.state(deepcopy(DiTmodel))
            ckpt = {'model': model_state, 'config': trainconfig.to_dict(), 'epoch': epoch} #, "args": vars(args)}
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(os.path.abspath(os.path.join(models_dir, f'ckpt_{epoch}')), ckpt)
            checkpointer.wait_until_finished()
        
            # Save args separately as JSON
            args_path = os.path.join(models_dir, f'ckpt_{epoch}_args.json')
            with open(args_path, 'w') as f:
                json.dump(vars(args), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    parser.add_argument("--results_dir", "-r", default="./results", help="Directory to save results to.")
    
    args = parser.parse_args()
    main(args)
