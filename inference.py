import jax 
import jax.numpy as jnp
import numpy as np
import argparse
import os
from PIL import Image
import torch

from helpers.config import modelConfig, trainConfig
from helpers.checkpoint import restore_checkpoint
from helpers.create_diffusion import create_diffusion
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
    
    # Restore the JAX model from the converted checkpoint
    model, _ = restore_checkpoint(args.checkpoint_path, model_config, trainConfig(), gpu_device)
    print(f"Model restored from {args.checkpoint_path}")

    # Load the VAE
    vae = get_sd_vae()
    vae.to('cuda')
    vae.eval()

    # --- Diffusion Setup ---
    diffusion = create_diffusion(args.steps)
    rng = jax.random.PRNGKey(args.seed)

    # --- Prepare Class Labels ---
    class_labels = [int(label) for label in args.class_labels.split(',')]
    n = len(class_labels)
    y = jnp.array(class_labels)

    # --- Inference ---
    print(f"\nStarting diffusion sampling with CFG (scale={args.cfg_scale})...")
    
    # Create initial noise
    shape = (n, model_config.image_channels, model_config.input_size, model_config.input_size)
    rng, noise_rng = jax.random.split(rng)
    
    # Run the sampling loop
    latents = diffusion.p_sample_loop(model, shape, y, args.cfg_scale, noise_rng, progress=True)

    print("Diffusion process complete.")

    # --- Decode and Save Image ---
    print("\n--- Decoding latents with VAE ---")
    latents = latents / 0.18215
    
    # Cast to float32 for PyTorch VAE
    latents = latents.astype(jnp.float32)
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