import jax 
import jax.numpy as jnp
import flax
from jax import grad
from flax import nnx
import optax
        
class Config:
    def __init__(self):
        self.model_type = 'DiT-S'
        self.input_size = 32
        self.n_layers = 12
        self.n_heads = 6
        self.DiT_hidden_size = 384
        self.MLP_hidden_size = self.hidden_size * 4
        self.patch_size = 8
        self.token_length = (self.input_size/self.patch_size)**2
        self.learning_rate = 1e-4
        self.ema = 0.9999
        self.rngs = nnx.Rngs(42)

class MHA(nnx.Module):
    def __init__(self, config: Config):
                 #length, hidden, num_heads, rngs):
      self.length = config.token_length
      self.hidden = config.DiT_hidden_size
      self.num_heads = config.n_heads
      self.rngs = config.rngs

      self.d_head = self.hidden // self.num_heads
      self.qkv_proj = nnx.Linear(self.hidden, 3 * self.hidden, rngs=self.rngs)
      self.out_proj = nnx.Linear(self.hidden, self.hidden, rngs=self.rngs)

    def forward(self, x):
        # x: [L, Hidden]
        L = x.shape[0]
        y = self.qkv_proj(x)                    # [L, 3H]
        y = y.reshape(L, 3, self.num_heads, self.d_head)   # [L, 3, head, d]
        y = jnp.transpose(y, (1, 2, 0, 3))       # [3, h, L, d]
        q, k, v = y[0], y[1], y[2]               # each [h, L, d]

        values, attention = self.scaled_dot_product(q, k, v)  # values: [h, L, d]
        values = jnp.transpose(values, (1, 0, 2)).reshape(L, self.hidden)  # [L, H]
        values = self.out_proj(values)              # [L, H]
        return values, attention               # attention: [h, L, L]
    
    def scaled_dot_product(self, q, k, v):
        """Implements scaled dot product with PyTorch's functionality"""
        # q, k, v: [num_heads, L, d_head]
        d = q.shape[-1]
        scale = 1.0 / jnp.sqrt(d)
        # [num_heads, L, L]
        attn_logits = jnp.einsum('hld,hmd->hlm', q, k) * scale
        attn = jax.nn.softmax(attn_logits, axis=-1)
        # [num_heads, L, d_head]
        out = jnp.einsum('hlm,hmd->hld', attn, v)
        return out, attn

# Architecture from Vision Transformer
class MLP(nnx.Module):
    def __init__(self, config:Config):
            # in_features: int,
            # hidden_features: int):
        """
        Inititializes a MLP block. Has reduced 
        options compared to ViT implementations, but 
        sufficient for the task. Uses GeLU activations.
        """
        self.fc1 = nnx.Linear(config.DiT_hidden_size, config.MLP_hidden_size)
        self.act = lambda t: nnx.gelu(t, approximate=True)
        self.fc2 = nnx.Linear(config.MLP_hidden_size, config.DiT_hidden_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nnx.Module):
    def __init__(self, config: Config):
            # length: int,
            # hidden_size: int,
            # num_heads: int,
            # rngs: nnx.rng):
        """
        Initialize a DiT block.
        """
        self.config = config
        self.LayerNorm1 = nnx.LayerNorm(config.DiT_hidden_size, rngs=config.rngs)
        self.LayerNorm2 = nnx.LayerNorm(config.n_heads, rngs=config.rngs)
        self.MHA = MHA(config)
        self.MLP = MLP(config) # Default in PyTorch implementation.

        # Conditional MLP
        self.cLinWeights = nnx.Linear(config.DiT_hidden_size, config.MLP_hidden_size)

    def forward(self, x, conditioning):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.cLinWeights.value
        tmp = self.LayerNorm1(x)
        tmp = gamma1*tmp + beta1*conditioning
        tmp = self.MHA(tmp)
        tmp = alpha1*tmp + conditioning

        tmp += x
        x = tmp

        tmp = self.LayerNorm2(tmp)
        tmp = gamma2*tmp + beta2*conditioning
        tmp = self.MLP(tmp)
        tmp = alpha2*tmp + conditioning

        x += tmp

        return x

class DiTFinalLayer(nnx.Module):
    def __init__(self, config: Config):
        pass
    def forward(self):
        pass


class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer"""
    def __init__(self, config: Config):
        self.config = config
        self.length = (32/self.config.patch_size)**2
        self.layers = [
                        DiTBlock(config) for _ in range(self.config.n_layer)
                      ]
        self.final_layer = DiTFinalLayer(config)



# class Model(nnx.Module):
#   def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
#     self.linear = nnx.Linear(din, dmid, rngs=rngs)
#     self.bn = nnx.BatchNorm(dmid, rngs=rngs)
#     self.dropout = nnx.Dropout(0.2, rngs=rngs)
#     self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

#   def __call__(self, x):
#     x = nnx.relu(self.dropout(self.bn(self.linear(x))))
#     return self.linear_out(x)

# model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # Eager initialization
# optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# @nnx.jit  # Automatic state management for JAX transforms.
# def train_step(model, optimizer, x, y):
#   def loss_fn(model):
#     y_pred = model(x)  # call methods directly
#     return ((y_pred - y) ** 2).mean()

#   loss, grads = nnx.value_and_grad(loss_fn)(model)
#   optimizer.update(model, grads)  # in-place updates

#   return loss



