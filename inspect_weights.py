from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig, trainConfig
import jax
import jax.numpy as jnp
from flax.nnx.statelib import State
from flax import nnx
import argparse
import os
from helpers.checkpoint import save_checkpoint, restore_checkpoint

from flax import nnx

def print_param_stats(params, path=''):
    for key, value in params.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, State):
            print_param_stats(value, new_path)
        else:
            print(f"{new_path}:")
            print(f"  mean: {jnp.mean(value)}")
            print(f"  std: {jnp.std(value)}")

def main(args):
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        gpu_device = None
    assert gpu_device != None

    # Load model configuration
    modelconfig = modelConfig()
    trainconfig = trainConfig()
    
    # Load model if desired:
    if args.resume:
        experiment_path = args.resume
        models_dir = os.path.abspath(os.path.join(experiment_path, "models"))

        model, extra_params = restore_checkpoint(models_dir, modelconfig, trainconfig, gpu_device)
    else:
        # Create the model
        model = DiffusionTransformer(modelconfig, jax.random.PRNGKey(0))

    # Get the model's parameters
    params = nnx.state(model)

    # Print the statistics for each parameter
    print_param_stats(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-re", help="Path to the experiment directory to resume training from.")
    args = parser.parse_args()
    main(args)
