from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from helpers.config import modelConfig, trainConfig
import argparse
import os
from diffusion import create_diffusion # Use PyTorch diffusion
from vae.import_sd_vae_torch import get_sd_vae
from PIL import Image
import numpy as np
import torch
from helpers.checkpoint import restore_checkpoint
from torchvision.utils import save_image # For saving images

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
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

    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=args.learn_sigma)
    vae = get_sd_vae()
    vae.to('cuda')
    vae.eval()

    # Labels to condition the model with (feel free to change):
    class_labels = [int(label) for label in args.class_labels.split(',')]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    print(z.shape)
    samples = diffusion.p_sample_loop(
        model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    print(samples.shape)
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(n):
        output_filename = os.path.join(args.output_dir, f"sample_class_{class_labels[i]}.png")
        save_image(samples[i], output_filename, normalize=True, value_range=(-1, 1))
        print(f"Image saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/scratch/network/jl0796/DiT-ImageNet/results/experiment-v215/ema_model", help="Path to the directory containing the pretrained pytorch model checkpoint.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Path to the directory to save generated images.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learn-sigma", action="store_true", default=False, help="Set to true to use a model that learns sigma.")
    parser.add_argument("--class_labels", type=str, default="207,360,387,974,88,979,417,279", help="Comma-separated list of ImageNet class labels to generate.")
    args = parser.parse_args()
    main(args)
