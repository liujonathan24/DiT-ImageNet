from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from jax import grad, vmap
from flax import nnx
from helpers.config import modelConfig
import optax
import argparse



# https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf
# loss is l_simple 
# # jk no vlb in the code I think? 
# + lambda * l_vlb




class Model(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    pass
  def __call__(self, x):
    pass
  

model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, x, y):
  def loss_fn(model):
    y_pred = model(x)  # call methods directly
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)  # in-place updates

  return loss






def main(args):
    assert args.model_config == "DiT-S", "Currently, the only model config implemented is DiT-S."
    config = modelConfig()
    batch = 256
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