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
import optax
import orbax.checkpoint as ocp
import time
import jax_dataloader as jdl
from helpers.diffusion import Diffusion
from vae.import_sd_vae_torch import get_sd_vae
from PIL import Image
import numpy as np

from helpers.porting import restore_checkpoint

def main(args):
    #if jax.devices("gpu"):
    #    gpu_device = jax.devices("gpu")[0]
    #else:
    #    gpu_device = jax.devices("cpu")[0]
    gpu_device = jax.devices("cpu")[0]
    # assert gpu_device != None

    sd_vae, _ = get_sd_vae()


    modelconfig = modelConfig()
    trainconfig = trainConfig()
    model = DiffusionTransformer(modelconfig)

    model, extra_params = restore_checkpoint(args.checkpoint_path, modelconfig, trainconfig, gpu_device)

    print("Model restored from checkpoint")
    print(f"Restored epoch: {extra_params['epoch']}")

    config = modelConfig()
    trainconfig = trainConfig()
    diffusion = Diffusion(trainconfig.linear_variance_min, trainconfig.linear_variance_max, trainconfig.tmax)

    os.makedirs(args.output_dir, exist_ok=True)
    rngs = jax.random.PRNGKey(42)
    # Sample 1000 images for FID.
    for i in range(int(jnp.ceil(1000/trainconfig.batch_size))):
        # Diffusion process. Starts with [b, c, h, w] = [b, 4, 32, 32] ~ N(0, 1)
        x_t = jax.random.normal(rngs, shape=(trainconfig.batch_size, config.image_channels, config.input_size, config.input_size))
        for t in range(2, 0, -1): # range(1000, 0, -1):
            # t = jnp.ones((trainconfig.batch_size)) * t
            # print(x_t.shape, t.shape)
            t_vec = jnp.ones((trainconfig.batch_size)) * t
            prediction = model(x_t, t_vec)
            modified_x_t = x_t - prediction * (1-diffusion.alphas[t])/(jnp.sqrt(1-diffusion.alpha_bars[t]))

            modified_x_t *= 1/jnp.sqrt(diffusion.alphas[t])

            z_t = jax.random.normal(rngs, shape=(trainconfig.batch_size, config.image_channels, config.input_size, config.input_size)) if t >1 else jnp.zeros((trainconfig.batch_size, config.image_channels, config.input_size, config.input_size))
            noise_t = jnp.sqrt(diffusion.variances[t]) * z_t

            x_t = modified_x_t + noise_t
        # Decode:
        print(f"Final shape is {x_t.shape}")
        img = sd_vae.decode(x_t)
        print(f"Img shape is: {img.shape}")
        im = Image.fromarray(np.array(img))
        im.save(f"sample_{i}.jpeg")
        break






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", "-c", help="Path to the checkpoint directory to restore from.")
    parser.add_argument("--output_dir", "-o", help="Path to the directory to save images to.")
    args = parser.parse_args()
    main(args)
