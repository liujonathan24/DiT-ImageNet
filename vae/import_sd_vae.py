import os
from functools import lru_cache
import jax.numpy as jnp
from diffusers import FlaxAutoencoderKL

def _has_flax_checkpoint(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "diffusion_flax_model.msgpack"))
        and os.path.isfile(os.path.join(path, "config.json"))
    )

@lru_cache(maxsize=1) # Ensure download only occurs once
def get_sd_vae(cache_dir: str = "./vae/jax_sd-vae-ft-ema", dtype=jnp.float32):
    os.makedirs(cache_dir, exist_ok=True)

    # Use local copy if it exists
    if _has_flax_checkpoint(cache_dir):
        print("Local copy detected.")
        vae, params = FlaxAutoencoderKL.from_pretrained(cache_dir, dtype=dtype)
        return vae, params

    print("Converting PT weights of stabilityai/sd-vae-ft-ema to jax's .msgpack filetype and saving.")
    # If model architecture & weights are not downloaded, downloads it (converts pt -> flax)
    vae, params = FlaxAutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema",
        from_pt=True,              
        dtype=dtype,
    )
    vae.save_pretrained(cache_dir, params=params)  # writes diffusion_flax_model.msgpack + config.json
    return vae, params


if __name__=="__main__":
    vae, params = get_sd_vae("./vae/jax_sd-vae-ft-ema", jnp.float32)
