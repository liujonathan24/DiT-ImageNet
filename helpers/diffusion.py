import jax.numpy as jnp

class Diffusion():
    def __init__(self, variance_min, variance_max, steps):
        self.steps = steps
        self.variance_min = variance_min
        self.variance_max = variance_max
        variances = jnp.linspace(variance_max, variance_min, steps)
        alpha = 1 - variances
        self.alpha_bar = jnp.cumprod(alpha)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t]