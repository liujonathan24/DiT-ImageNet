from diffusers import AutoencoderKL
from vae.import_sd_vae_torch import get_sd_vae
# from config import trainConfig

vae, params = get_sd_vae("./vae/sd-vae-ft-ema")


from ptflops import get_model_complexity_info
import torch


input_tensor = torch.randn(1, 3, 256, 256)

flops, params = get_model_complexity_info(
    vae,
    (3, 256, 256), # input_shape
)

print(f"FLOPs: {flops}")
print(f"Parameters: {params}")

