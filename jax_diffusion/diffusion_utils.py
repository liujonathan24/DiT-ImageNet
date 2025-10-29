import jax.numpy as jnp
import numpy as np

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = arr[timesteps].astype(jnp.float32)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return jnp.broadcast_to(res, broadcast_shape)