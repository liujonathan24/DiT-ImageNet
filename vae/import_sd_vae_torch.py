import os
from functools import lru_cache
import jax.numpy as jnp
from diffusers import AutoencoderKL

def _has_flax_checkpoint(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "diffusion_flax_model.msgpack"))
        and os.path.isfile(os.path.join(path, "config.json"))
    )


def get_sd_vae(cache_dir: str = "./vae/sd-vae-ft-ema", dtype=jnp.float32):
    os.makedirs(cache_dir, exist_ok=True)

    # Use local copy if it exists
    if _has_flax_checkpoint(cache_dir):
        print("Local copy detected.")
        vae = AutoencoderKL.from_pretrained(cache_dir)
        return vae, None

    print("install")
    # If model architecture & weights are not downloaded, downloads it (converts pt -> flax)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
    )
    vae.save_pretrained(cache_dir)  # writes diffusion_flax_model.msgpack + config.json
    return vae, None


if __name__=="__main__":
    vae, params = get_sd_vae("./vae/sd-vae-ft-ema")

