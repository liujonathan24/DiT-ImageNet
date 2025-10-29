import jax.numpy as jnp
import numpy as np

from .diffusion import GaussianDiffusion

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def create_diffusion(steps, noise_schedule="linear", model_mean_type="epsilon", model_var_type="learned_range"):
    betas = get_named_beta_schedule(noise_schedule, steps)
    
    if model_mean_type != "epsilon":
        raise ValueError("Only epsilon prediction is supported.")
    if model_var_type != "learned_range":
        raise ValueError("Only LEARNED_RANGE is supported.")

    return GaussianDiffusion(
        betas=betas,
        model_mean_type=model_mean_type, # "epsilon"
        model_var_type=model_var_type, # "learned_range"
    )
