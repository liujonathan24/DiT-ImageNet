import jax 
import jax.numpy as jnp
from flax import nnx
import numpy as np
import argparse
import os
from PIL import Image
from tqdm import tqdm

from helpers.config import modelConfig, trainConfig
from helpers.checkpoint import restore_checkpoint
from helpers.diffusion import Diffusion
from vae.import_sd_vae_torch import get_sd_vae


def main(args):
    # --- Device Setup ---
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        raise ValueError("GPU not found. This script requires a GPU.")

    # --- Load Models and Configs ---
    print("Loading models and configurations...")
    model_config = modelConfig(type='DiT-XL')
    train_config = trainConfig() # Needed for diffusion schedule

    # Restore the JAX model from the converted checkpoint
    model, _ = restore_checkpoint(args.checkpoint_path, model_config, train_config, gpu_device)
    print(f"Model restored from {args.checkpoint_path}")

    # Load the VAE
    vae = get_sd_vae()
    vae.to('cuda')
    vae.eval()

    # --- Diffusion Setup ---
    diffusion = Diffusion(train_config.linear_variance_min, train_config.linear_variance_max, train_config.tmax)
    rng = jax.random.PRNGKey(args.seed)

    # --- Inference Loop ---
    print(f"Starting diffusion sampling for {args.steps} steps...")
    
    # 1. Start with random noise
    rng, noise_rng = jax.random.split(rng)
    latents = jax.random.normal(noise_rng, (1, model_config.image_channels, model_config.input_size, model_config.input_size))

    # 2. Denoising loop
    for t in tqdm(reversed(range(args.steps)), total=args.steps):
        t_batch = jnp.array([t])
        
        # Predict noise and variance
        model_output = model(latents, t_batch)
        predicted_noise, predicted_variance = jnp.split(model_output, 2, axis=1)
        
        # Get diffusion schedule parameters for the current timestep
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alpha_bars[t]
        beta_t = diffusion.betas[t]

        # Denoise one step using only the predicted noise (epsilon)
        # Formula from DDPM paper (https://arxiv.org/abs/2006.11239), Eq. 11
        coeff = (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t)
        latents = (1 / jnp.sqrt(alpha_t)) * (latents - coeff * predicted_noise)

        # Add noise back in (except for the last step)
        if t > 0:
            rng, noise_rng = jax.random.split(rng)
            z = jax.random.normal(noise_rng, latents.shape)
            # Formula from DDPM paper, Eq. 7
            latents += jnp.sqrt(beta_t) * z

    print("Diffusion process complete.")

    # --- Decode and Save Image ---
    print("Decoding latents with VAE...")
    # The model was trained on latents scaled by 0.18215
    latents = latents / 0.18215
    import torch
    latents_torch = torch.from_numpy(np.array(latents)).to(vae.device) 
    image = vae.decode(latents_torch).sample

    # Convert to numpy and rescale to [0, 255]
    image = image.detach().cpu().numpy()
    image = (image / 2 + 0.5).clip(0, 1)
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image[0], (1, 2, 0))

    # Save the image
    pil_image = Image.fromarray(image)
    if args.output_path == "generated_image.png":
        pil_image.save(f"{args.steps}_generated_image.png")
    else:
        pil_image.save(args.output_path)
    print(f"Image saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion inference with a trained DiT model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the directory containing the JAX model checkpoint.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion steps.")
    parser.add_argument("--output_path", type=str, default="generated_image.png", help="Path to save the generated image.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise generation.")
    args = parser.parse_args()
    main(args)
