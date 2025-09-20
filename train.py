from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from jax import grad, vmap
from flax import nnx
from helpers.config import modelConfig, trainConfig
import optax
import argparse
from helpers.preprocess_data_torch import load_latents
from tqdm import tqdm



# https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf
# loss is l_simple 
# # jk no vlb in the code I think? 
# + lambda * l_vlb

@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, x, y):
  def loss_fn(model):
    y_pred = model(x)  
    return optax.losses.squared_error(y_pred, y).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)  # in-place updates

  return loss


def main(args):
    assert args.model_config == "DiT-S", "Currently, the only model config implemented is DiT-S."

    trainConfig = trainConfig()
    train_latents, train_labels = load_latents("./data/train_latent")
    test_latents, test_labels = load_latents("./data/test_latent")


    modelConfig = modelConfig()
    test_DiT = DiffusionTransformer(modelConfig)
    optimizer = nnx.Optimizer(test_DiT, optax.adam(1e-3), wrt=nnx.Param)

    for batch in tqdm(train_latents): # maybe convert to dataloader again?
        #TODO: 
        # sampled_batch = 
        # noise_level = 
        # noise = 

        x = None # sampled batch + noise * noise_level
        y = None # noise * noise_level

        train_step(test_DiT, optimizer, x, y)
        # Evaluate?
        #TODO: 
    

if __name__ == "__main__":
    parser = argparse.ArgumentPareser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    args = parser.parse_args()
    main(args)