import jax.numpy as jnp

class Diffusion():
    def __init__(self, variance_min, variance_max, steps):
        self.steps = steps
        self.variance_min = variance_min
        self.variance_max = variance_max
        self.variances = jnp.linspace(variance_max, variance_min, steps)
        self.alphas = 1 - self.variances
        self.alpha_bars = jnp.cumprod(self.alphas)

    def get_alpha_bar(self, t):
        return self.alpha_bars[t]
