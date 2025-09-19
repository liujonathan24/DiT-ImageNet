import numpy as np
from PIL import Image
import jax.numpy as jnp
from diffusers import FlaxAutoencoderKL
from import_sd_vae import get_sd_vae

# Load model & weights
vae, params = get_sd_vae()

path = "./vae/original.png"
img = Image.open(path).convert("RGB")

# Convert to [1, 3, H, W] format with range [-1, 1] 
x = np.asarray(img).astype(np.float32) / 255.0        # [0,1]
x = (x - 0.5) / 0.5                                   # [-1,1]
x = jnp.asarray(x)[None, ...]                         # [1, H, W, 3] 
x = jnp.transpose(x, (0, 3, 1, 2))               # [1, 3, H, W]
print(f"Original shape: {x.shape}")


# Encode image
distr = vae.apply(
                    {"params": params}, 
                    x, 
                    method=vae.encode, 
                    deterministic=True)
patches = distr.latent_dist.mean  

print(f"Patches shape: {patches.shape}")

# Decode
decoded_x = vae.apply(
                    {"params": params},
                    patches,
                    method=vae.decode).sample


# Convert back to [1, H, W, 3] and [0, 256]
decoded_x = jnp.transpose(decoded_x, (0, 2, 3, 1)) 
img_out = np.clip((np.array(decoded_x[0]) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
print(f"Final decoded shape: {decoded_x.shape}")

Image.fromarray(img_out).save("./vae/reconstruction.png")