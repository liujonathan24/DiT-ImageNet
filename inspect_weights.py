from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig
import jax
import jax.numpy as jnp
from flax.nnx.statelib import State

def print_param_stats(params, path=''):
    for key, value in params.items():
        new_path = f"{path}/{key}" if path else key
        if isinstance(value, State):
            print_param_stats(value, new_path)
        else:
            print(f"{new_path}:")
            print(f"  mean: {jnp.mean(value)}")
            print(f"  std: {jnp.std(value)}")

def main():
    # Load model configuration
    config = modelConfig()

    # Create the model
    model = DiffusionTransformer(config)

    # Get the model's parameters
    params = model.get_state()

    # Print the statistics for each parameter
    print_param_stats(params)

if __name__ == "__main__":
    main()