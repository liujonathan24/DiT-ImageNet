import enum
import jax
import jax.numpy as jnp

# === Enums for type safety and clarity ===
class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()

class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

# === Utility Functions ===

def mean_flat(x):
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    arr = jnp.array(arr)
    val = arr[timesteps]
    while len(val.shape) < len(broadcast_shape):
        val = val[..., None]
    return val + jnp.zeros(broadcast_shape, dtype=val.dtype)

def get_named_beta_schedule(schedule_name, num_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_timesteps
        betas = jnp.linspace(scale * 0.0001, scale * 0.02, num_timesteps, dtype=jnp.float32)
    elif schedule_name == "cosine":
        # Cosine schedule as in Nichol & Dhariwal 2021
        s = 0.008
        steps = num_timesteps + 1
        x = jnp.linspace(0, num_timesteps, steps) / num_timesteps
        alphas_cumprod = jnp.cos((x + s) / (1 + s) * jnp.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = jnp.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(schedule_name)
    return betas

# --- KL-divergence between Gaussians, used for VLB loss ---
def normal_kl(mean1, logvar1, mean2, logvar2):
    """Calculate KL divergence between two normal distributions."""
    return 0.5 * (
        logvar2 - logvar1 +
        (jnp.exp(logvar1) + (mean1 - mean2) ** 2) / jnp.exp(logvar2) - 1
    )

# === Main Class: GaussianDiffusion ===
class GaussianDiffusion:
    def __init__(self, betas, model_mean_type, model_var_type, loss_type):
        self.betas = jnp.array(betas, dtype=jnp.float32)
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.num_timesteps = self.betas.shape[0]
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior q(x_{t-1} | x_t, x_0) parameters
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clipped log for small variance
        self.posterior_log_variance_clipped = jnp.log(
            jnp.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * jnp.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise):
        '''Diffuse x_start to timestep t using noise.'''
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        '''Returns mean/var/logvar of q(x_{t-1} | x_t, x_0).'''
        mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, variance, log_variance

    def p_mean_variance(self, model, params, x, t, key, clip_denoised=True, return_pred_xstart=False, **kwargs):
        '''Compute predicted mean, variance, and optionally x_start from model output.'''
        # Model returns a tuple if learning variance, else just pred
        model_output = model.apply(params, x, t, **kwargs)  # Use Flax API
        if self.model_var_type in {ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE}:
            pred, var_values = jnp.split(model_output, 2, axis=1)
            # Use learned variance or interpolate if LEARNED_RANGE
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(jnp.log(self.betas), t, x.shape)
            frac = (var_values + 1) / 2  # [-1,1] to [0,1]
            model_log_variance = frac * max_log + (1 - frac) * min_log
        else:
            pred = model_output
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                model_log_variance = _extract_into_tensor(jnp.log(self.betas), t, x.shape)
            else:
                model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            model_mean = x - pred
        elif self.model_mean_type == ModelMeanType.START_X:
            model_mean = pred
        elif self.model_mean_type == ModelMeanType.EPSILON:
            model_mean = (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * pred
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        if clip_denoised:
            model_mean = jnp.clip(model_mean, -1., 1.)

        if return_pred_xstart:
            return model_mean, model_log_variance, pred
        return model_mean, model_log_variance

    def training_losses(self, model, params, x_start, t, key, **kwargs):
        '''Main loss for DDPM training (MSE or KL/VLB types).'''
        noise = jax.random.normal(key, x_start.shape)
        x_t = self.q_sample(x_start, t, noise)
        terms = {}

        model_output = model.apply(params, x_t, t, **kwargs)

        if self.loss_type in {LossType.KL, LossType.RESCALED_KL}:
            # VLB term: KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
            true_mean, true_var, true_log_var = self.q_posterior_mean_variance(x_start, x_t, t)
            model_mean, model_log_var = self.p_mean_variance(model, params, x_t, t, key)
            kl = normal_kl(true_mean, true_var, model_mean, jnp.exp(model_log_var))
            kl = mean_flat(kl)
            terms["loss"] = kl
            terms["kl"] = kl
        elif self.loss_type in {LossType.MSE, LossType.RESCALED_MSE}:
            if self.model_mean_type == ModelMeanType.EPSILON:
                loss = mean_flat((model_output - noise) ** 2)
            elif self.model_mean_type == ModelMeanType.START_X:
                x_recon = model_output
                loss = mean_flat((x_start - x_recon) ** 2)
            else:
                raise NotImplementedError(self.model_mean_type)
            terms["loss"] = loss
        else:
            raise NotImplementedError(self.loss_type)
        return terms

# --- Sampling Functions for JAX Diffusion ---

def sample_loop(gdiff, model, params, shape, rng, clip_denoised=True, **kwargs):
    """
    Standard sampling loop: iteratively sample x_{t-1} from x_t for all t, starting at pure noise.
    """
    x = jax.random.normal(rng, shape)
    def body_fn(i, val):
        x = val
        t = (jnp.ones(x.shape[0]) * (gdiff.num_timesteps - i - 1)).astype(jnp.int32)
        rng_step = jax.random.fold_in(rng, i)
        model_mean, model_log_var = gdiff.p_mean_variance(model, params, x, t, rng_step, clip_denoised)
        if i > 0:
            noise = jax.random.normal(rng_step, x.shape)
            x = model_mean + jnp.exp(0.5 * model_log_var) * noise
        else:
            x = model_mean
        return x
    x_final = jax.lax.fori_loop(0, gdiff.num_timesteps, body_fn, x)
    return x_final

def ddim_sample_loop(gdiff, model, params, shape, rng, eta=0.0, clip_denoised=True, **kwargs):
    """
    Deterministic DDIM sampling loop.
    Args:
        eta: controls stochasticity, eta=0 is pure DDIM, eta=1 is regular diffusion.
    """
    x = jax.random.normal(rng, shape)
    alphas_cumprod = gdiff.alphas_cumprod
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
    num_timesteps = gdiff.num_timesteps

    def body_fn(i, val):
        x = val
        t = (jnp.ones(x.shape[0]) * (num_timesteps - i - 1)).astype(jnp.int32)
        rng_step = jax.random.fold_in(rng, i)
        model_output = model.apply(params, x, t, **kwargs)
        if gdiff.model_mean_type == ModelMeanType.EPSILON:
            pred = (
                x - sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * model_output
            ) / sqrt_alphas_cumprod[t][:, None, None, None]
        else:
            pred = model_output
        if i > 0:
            sigma = eta * jnp.sqrt(
                (1.0 - alphas_cumprod[t] / alphas_cumprod[t - 1]) *
                (1.0 - alphas_cumprod[t - 1]) / (1.0 - alphas_cumprod[t])
            )
            noise = jax.random.normal(rng_step, x.shape)
            x = (
                sqrt_alphas_cumprod[t - 1][:, None, None, None] * pred +
                jnp.sqrt(1.0 - alphas_cumprod[t - 1])[:, None, None, None] * model_output +
                sigma * noise
            )
        else:
            x = pred
        return x
    x_final = jax.lax.fori_loop(0, num_timesteps, body_fn, x)
    return x_final
