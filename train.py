from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from jax import grad, vmap
from config import modelConfig
import optax
import argparse



def main(args):
    assert args.model_config == "DiT-S", "Currently, the only model config implemented is DiT-S."
    config = modelConfig()
    batch = 8
    # test_input = jnp.ones((batch, config.token_length, config.DiT_hidden_size))
    # test_condit = jnp.ones((batch, config.DiT_hidden_size))

    test_DiTBlock = DiffusionTransformer(config)

    # x = test_DiTBlock(test_input, test_condit)
    # print(test_input.shape, x.shape)

    # for num_batches in range(batches):
        # test_DiTBlock

    def loss(pred, true_noise):
        loss = optax.losses.squared_error(pred, true_noise).mean()
    

if __name__ == "__main__":
    parser = argparse.ArgumentPareser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    args = parser.parse_args()
    main(args)