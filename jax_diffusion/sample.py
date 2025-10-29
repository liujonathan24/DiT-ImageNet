# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from jax_diffusion import create_diffusion
import argparse

import jax 
import jax.numpy as jnp
import torch

from helpers.config import modelConfig, trainConfig
from helpers.checkpoint import restore_checkpoint
from vae.import_sd_vae_torch import get_sd_vae


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
    model_config = modelConfig()
    
    # Restore the JAX model from the converted checkpoint
    model, _ = restore_checkpoint(args.checkpoint_path, model_config, trainConfig(), gpu_device)
    
    print(f"Model restored from {args.checkpoint_path}")

    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=args.learn_sigma)
    vae = get_sd_vae()
    vae.to('cuda')
    vae.eval()

    # Labels to condition the model with (feel free to change):
    class_labels = [210, 363, 390, 977, 91, 982, 417, 279] # Changed 420 to 417 to match original sample.py

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Convert initial noise and labels to JAX arrays
    z_jax = jnp.array(z.detach().cpu().numpy())
    y_jax = jnp.array(y.detach().cpu().numpy())

    # Setup classifier-free guidance (using JAX arrays):
    z_in = jnp.concatenate([z_jax, z_jax], axis=0)
    y_null_jax = jnp.array([1000] * n, dtype=jnp.int32)
    y_in = jnp.concatenate([y_jax, y_null_jax], axis=0)
    model_kwargs = dict(y=y_in, cfg_scale=args.cfg_scale)

    # Sample images:
    print(z_in.shape)
    samples_jax = diffusion.p_sample_loop(
        model, z_in.shape, z_in, clip_denoised=False, model_kwargs=model_kwargs, progress=True 
    )
    print(samples_jax.shape)
    
    # Convert JAX samples back to PyTorch for VAE decoding and saving
    samples_torch = torch.from_numpy(np.array(samples_jax)).to(device)
    samples_torch, _ = samples_torch.chunk(2, dim=0)  # Remove null class samples
    samples_torch = vae.decode(samples_torch / 0.18215).sample

    # Save and display images:
    save_image(samples_torch, f"sample-v215_samples_{"_".join([str(i) for i in class_labels])}.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/scratch/network/jl0796/DiT-ImageNet/results/experiment-v215/ema_model", help="Path to the directory containing the pretrained pytorch model checkpoint.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--learn-sigma", action="store_true", default=False, help="Set to true to use a model that learns sigma.")
    args = parser.parse_args()
    main(args)
