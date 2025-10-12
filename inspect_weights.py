from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig
import jax
import jax.numpy as jnp
from flax import nnx

def main():
    # Load model configuration
    config = modelConfig()

    # Create the model
    model = DiffusionTransformer(config)

    # Get the model's parameters
    params = nnx.state(model)

    # Iterate through the parameters and print their stats
    for path, value in params.items():
        print(f"{path}:")
        print(f"  mean: {jnp.mean(value)}")
        print(f"  std: {jnp.std(value)}")

if __name__ == "__main__":
    main()
