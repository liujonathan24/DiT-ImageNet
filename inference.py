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

    # --- Classifier-Free Guidance Setup ---
    # Create labels for CFG
    class_labels = [int(label) for label in args.class_labels.split(',')]
    n = len(class_labels)
    y = jnp.array(class_labels)
    y_null = jnp.array([model_config.num_classes] * n) # Use num_classes as the null label
    y_batch = jnp.concatenate([y, y_null], axis=0)

    # --- Inference Loop --- 
    print(f"Starting diffusion sampling with CFG (scale={args.cfg_scale})...")
    
    # 1. Start with random noise
    rng, noise_rng = jax.random.split(rng)
    latents = jax.random.normal(noise_rng, (n, model_config.image_channels, model_config.input_size, model_config.input_size))
    # Duplicate noise for CFG
    latents = jnp.concatenate([latents, latents], axis=0)

    # 2. Denoising loop
    for t in tqdm(reversed(range(args.steps)), total=args.steps):
        t_batch = jnp.array([t] * (n * 2)) # Create a batch of timesteps for CFG
        
        # Predict noise and variance with CFG
        model_output = model(latents, t_batch, y_batch)
        
        # Split the output into conditional and unconditional predictions
        cond_output, uncond_output = jnp.split(model_output, 2, axis=0)
        
        # Combine them using the CFG formula
        cfg_output = uncond_output + args.cfg_scale * (cond_output - uncond_output)

        # Split the combined output into predicted noise and variance
        predicted_noise, predicted_variance = jnp.split(cfg_output, 2, axis=1)
        
        # Get diffusion schedule parameters for the current timestep
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alpha_bars[t]
        beta_t = diffusion.betas[t]

        # Denoise one step using only the predicted noise (epsilon)
        # This update is applied to the full batch (cond and uncond latents)
        coeff = (1 - alpha_t) / jnp.sqrt(1 - alpha_bar_t)
        latents = (1 / jnp.sqrt(alpha_t)) * (latents - coeff * predicted_noise)

        # Add noise back in (except for the last step)
        if t > 0:
            rng, noise_rng = jax.random.split(rng)
            z = jax.random.normal(noise_rng, latents.shape)
            latents += jnp.sqrt(beta_t) * z

    print("Diffusion process complete.")

    # --- Decode and Save Image ---
    print("Decoding latents with VAE...")
    # Take only the conditional samples
    latents = latents[0:n]
    
    # The model was trained on latents scaled by 0.18215
    latents = latents / 0.18215
    import torch
    latents_torch = torch.from_numpy(np.array(latents)).to(vae.device) 
    image = vae.decode(latents_torch).sample

    # Convert to numpy and rescale to [0, 255]
    image = image.detach().cpu().numpy()
    image = (image / 2 + 0.5).clip(0, 1)
    image = (image * 255).astype(np.uint8)
    
    # Save the images
    for i in range(n):
        img_data = image[i]
        img_data = np.transpose(img_data, (1, 2, 0))
        pil_image = Image.fromarray(img_data)
        
        # Create a unique filename for each image
        base, ext = os.path.splitext(args.output_path)
        output_filename = f"{base}_class_{class_labels[i]}_steps_{args.steps}{ext}"
        pil_image.save(output_filename)
        print(f"Image saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion inference with a trained DiT model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the directory containing the JAX model checkpoint.")
    parser.add_argument("--steps", type=int, default=250, help="Number of diffusion steps.")
    parser.add_argument("--output_path", type=str, default="generated_image.png", help="Path to save the generated image(s).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for noise generation.")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="Scale for classifier-free guidance.")
    parser.add_argument("--class_labels", type=str, default="207", help="Comma-separated list of ImageNet class labels to generate.")
    args = parser.parse_args()
    main(args)