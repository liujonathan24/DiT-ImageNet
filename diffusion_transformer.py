import jax 
import jax.numpy as jnp
import flax
from jax import grad
from flax import nnx
        

def MHA(length, hidden, num_heads, rngs):
    d_head = hidden // num_heads
    qkv_proj = nnx.Linear(hidden, 3 * hidden, rngs=rngs)
    out_proj = nnx.Linear(hidden, hidden, rngs=rngs)

    def forward(x):
        # x: [L, Hidden]
        L = x.shape[0]
        y = qkv_proj(x)                    # [L, 3H]
        y = y.reshape(L, 3, num_heads, d_head)   # [L, 3, head, d]
        y = jnp.transpose(y, (1, 2, 0, 3))       # [3, h, L, d]
        q, k, v = y[0], y[1], y[2]               # each [h, L, d]

        values, attention = scaled_dot_product(q, k, v)  # values: [h, L, d]
        values = jnp.transpose(values, (1, 0, 2)).reshape(L, hidden)  # [L, H]
        values = out_proj(values)              # [L, H]
        return values, attention               # attention: [h, L, L]

    return forward

# Architecture from Vision Transformer
def MLP(nnx.Module):
    def __init__(self,
            in_features: int,
            hidden_features: int):
        """
        Inititializes a MLP block. Has reduced 
        options compared to ViT implementations, but 
        sufficient for the task. Uses GeLU activations.
        """
        self.fc1 = nnx.Linear(in_features, hidden_features)
        self.act = nnx.gelu(x, approximate=True)
        self.fc2 = nnx.Linear(hidden_features, in_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def DiTBlock(nnx.Module):
    def __init__(self,
            length: int,
            hidden_size: int,
            num_heads: int,
            rngs: nnx.rng):
        """
        Initialize a DiT block.
        """
        self.LayerNorm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.LayerNorm2 = nnx.LayerNorm(num_heads, rngs=rngs)
        self.MHA = MHA(length, hidden_size, num_heads, rngs=rngs)
        self.MLP = MLP(hidden_size, hidden_size*4) # Default in PyTorch implementation.

        # Conditional MLP
        self.cLinWeights = nnx.Linear(hidden_size, hidden_size*6)

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

class FinalLayer():
    def __init__(self):
        pass
    def forward(self):
        pass

class Config:
    def __init__(self):
        self.model_type = 'DiT-S'
        self.n_layer = 12
        self.n_head = 6
        self.hidden_size = 384
        self.patch_size = 8
        self.learning_rate = 1e-4
        self.ema = 0.9999
        self.rngs = nnx.Rngs(42)


class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer"""
    def __init__(self, config: Config):
        self.config = config
        self.length = (32/self.config.patch_size)**2
        self.layers = [DiTBlock(self.length, self.config.hidden_size, self.config.n_head, self.config.rngs) for _ in range(self.config.n_layer)]
        self.final_layer = FinalLayer()



class Model(nnx.Module):
  def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
    self.linear = nnx.Linear(din, dmid, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.2, rngs=rngs)
    self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x):
    x = nnx.relu(self.dropout(self.bn(self.linear(x))))
    return self.linear_out(x)

model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # Eager initialization
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

@nnx.jit  # Automatic state management for JAX transforms.
def train_step(model, optimizer, x, y):
  def loss_fn(model):
    y_pred = model(x)  # call methods directly
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)  # in-place updates

  return loss



