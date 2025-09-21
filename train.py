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


class diffusion():
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

    diffusion = diffusion(trainConfig.linear_variance_min, trainConfig.linear_variance_max, trainConfig.tmax)


    train_latents, train_labels = load_latents("./data/train_latent")
    print(type(train_latents))
    print(train_latents.shape)

    test_latents, test_labels = load_latents("./data/test_latent")

    trainConfig = trainConfig()
    modelConfig = modelConfig()
    DiTmodel = DiffusionTransformer(modelConfig)
    opt = optax.adamw(learning_rate=1e-3)
    optimizer = nnx.Optimizer(DiTmodel, opt, wrt=nnx.Param)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        for batch in tqdm(train_latents): # maybe convert to dataloader again?
            

            t = jnp.randint(0, 1000, (batch.shape[0],))

            # noise = 
            alpha_bar_t = diffusion.get_alpha_bar(t)
            noise = jnp.sqrt(1-alpha_bar_t) * jnp.random.normal(key=jax.random.PRNGKey(0), shape=batch.shape)

            batch *= 18215 * jnp.sqrt(alpha_bar_t)

            batch += noise
            
            jax.device_put(batch, device=gpu_device)
            print(batch.shape)
            jax.device_put(t, device=gpu_device)
            jax.device_put(noise, device=gpu_device)

            train_step(DiTmodel, optimizer, batch, t, noise)

        if epoch % trainConfig.ckpt_frequency == 0:
            # Bundle states into checkpoint and save for later EMA.
            model_state = nnx.state(deepcopy(DiTmodel))
            ckpt = {'model': model_state, 'config': trainConfig, 'epoch': epoch, "args": args}
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(os.path.join(models_dir, f'ckpt_{epoch//args.ckpt_frequency}'), ckpt)
    

if __name__ == "__main__":
    parser = argparse.ArgumentPareser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    parser.add_argument("--results_dir", "-r", default="./results", help="Directory to save results to.")
    
    args = parser.parse_args()
    main(args)