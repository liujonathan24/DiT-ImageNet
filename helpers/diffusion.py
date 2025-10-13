import numpy as np
import jax.numpy as jnp

# This file is a JAX/NumPy port of the original PyTorch implementation in
# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = arr[timesteps].astype(jnp.float32)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return jnp.broadcast_to(res, broadcast_shape)


class GaussianDiffusion:
    def __init__(self, *, betas, model_mean_type, model_var_type):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, y, cfg_scale):
        # This is a simplified version for inference that includes CFG
        B, C = x.shape[:2]
        assert t.shape == (B,)

        # Setup for CFG
        x_in = jnp.concatenate([x, x], axis=0)
        t_in = jnp.concatenate([t, t], axis=0)
        y_null = jnp.array([model.config.num_classes] * B, dtype=jnp.int32)
        y_in = jnp.concatenate([y, y_null], axis=0)

        # Model call
        model_output = model(x_in, t_in, y_in)

        # CFG
        cond_output, uncond_output = jnp.split(model_output, 2, axis=0)
        cfg_output = uncond_output + cfg_scale * (cond_output - uncond_output)
        model_output, model_var_values = jnp.split(cfg_output, 2, axis=1)

        # Calculate variance
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log

        # Calculate mean
        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        pred_xstart = jnp.clip(pred_xstart, -1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "log_variance": model_log_variance,
        }

    def p_sample(self, model, x, t, y, cfg_scale, rng):
        out = self.p_mean_variance(model, x, t, y, cfg_scale)
        noise = jax.random.normal(rng, x.shape, dtype=x.dtype)
        nonzero_mask = (t != 0).astype(x.dtype).reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise
        return sample

    def p_sample_loop(self, model, shape, y, cfg_scale, rng, progress=True):
        # This is a simplified JAX version of the loop
        from tqdm.auto import tqdm

        latents = jax.random.normal(rng, shape, dtype=jnp.bfloat16)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = jnp.array([i] * shape[0])
            rng, sample_rng = jax.random.split(rng)
            latents = self.p_sample(model, latents, t, y, cfg_scale, sample_rng)
        
        return latents
