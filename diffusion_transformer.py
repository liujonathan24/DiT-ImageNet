import jax 
import jax.numpy as jnp
import flax
from jax import grad

# DiT-S: 
# Layers = N = 12 
# Hidden Size = d = 384 
# heads = 6


####################################################################################
# Example:
####################################################################################

from flax import nnx
import optax


def MHA(hidden, num_heads, rngs):
    """
    Initializes a MHA layer
    """
    d_head = hidden//num_heads
    Ks = nnx.Linear() # Need to make "num_heads" of them, each with in & out being in: length, out: hidden

    def forward(x)

def attention(hidden, length):
    """
    Initializes an attention head
    """
    K = nnx.Linear(in_features=length, out_features=hidden, rngs=rngs)
    Q = nnx.Linear(in_features=length, out_features=hidden, rngs=rngs)
    V = nnx.Linear(in_features=length, out_features=hidden, rngs=rngs)

    def forward(x):
        key = K(x)
        query = Q(x)
        value = V(x)
        
        weight = nnx.softmax(nnx.swapaxes(key) @ query)
        output = value @ weight
        return output 

    return forward

def DiTBlock(nnx.Module):
    def __init__(self,
            hidden_size: int
            num_heads: int,
            time_emb_dim: int,
            rngs: nnx.rng):
        """
        Initialize a DiT block.
        """
        self.LayerNorm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.LayerNorm2 = nnx.LayerNorm(num_heads, rngs=rngs)
        self.MHA = MHA(hidden, features)
        self.linWeights = nnx.Param(jnp.ones(6, in_channels))
        
        self.FFN = FFN()

        # Conditional MLP
        self.MLP = MLP()

    def forward(self, x, conditioning):
        conditioning = self.MLP(conditioning)
        
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.linWeights.value
        tmp = self.LayerNorm1(x)
        tmp = gamma1*tmp + beta1*conditioning
        tmp = self.MHA(tmp)
        tmp = alpha1*tmp + conditioning

        tmp += x
        x = tmp

        tmp = self.LayerNorm2(tmp)
        tmp = gamma2*tmp + beta2*conditioning
        tmp = self.FFN(tmp)
        tmp = alpha2*tmp + conditioning

        x += tmp

        return x




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


####################################################################################
# Hyperparameters:
####################################################################################






class DiTBlock(nnx.Module):
    """Main iterated block for the diffusion transformer"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):


        self.norm1 = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nnx.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nnx.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nnx.Sequential(
            nnx.SiLU(),
            nnx.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer"""
    def __init__(self):
        pass
