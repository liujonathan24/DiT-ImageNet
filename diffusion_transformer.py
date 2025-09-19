import jax 
import jax.numpy as jnp
from jax import grad, vmap
from flax import nnx
from config import modelConfig
        

class MHA(nnx.Module):
    def __init__(self, config: modelConfig):
        
        self.length = config.token_length
        self.hidden = config.DiT_hidden_size
        self.num_heads = config.n_heads
        self.rngs = config.rngs

        self.d_head = self.hidden // self.num_heads
        self.qkv_proj = nnx.Linear(self.hidden, 3 * self.hidden, rngs=self.rngs)
        self.out_proj = nnx.Linear(self.hidden, self.hidden, rngs=self.rngs)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x: [L, Hidden]
        y = self.qkv_proj(x)                    # [L, 3H]
        y = y.reshape(self.length, 3, self.num_heads, self.d_head)   # [L, 3, head, d]
        y = jnp.transpose(y, (1, 2, 0, 3))       # [3, h, L, d]
        q, k, v = y[0], y[1], y[2]               # each [h, L, d]

        values, attention = self.scaled_dot_product(q, k, v)  # values: [h, L, d]
        values = jnp.transpose(values, (1, 0, 2)).reshape(self.length, self.hidden)  # [L, H]
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
    def __init__(self, config:modelConfig):
        """
        Inititializes a MLP block. Has reduced 
        options compared to ViT implementations, but 
        sufficient for the task. Uses GeLU activations.
        """
        self.fc1 = nnx.Linear(config.DiT_hidden_size, config.MLP_hidden_size, rngs=config.rngs)
        self.act = lambda t: nnx.gelu(t, approximate=True)
        self.fc2 = nnx.Linear(config.MLP_hidden_size, config.DiT_hidden_size, rngs=config.rngs)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class DiTBlock(nnx.Module):
    def __init__(self, config: modelConfig):
        """
        Initialize a DiT block.
        """
        
        self.config = config
        self.LayerNorm1 = nnx.LayerNorm(config.DiT_hidden_size, rngs=config.rngs)
        self.LayerNorm2 = nnx.LayerNorm(config.DiT_hidden_size, rngs=config.rngs)
        self.MHA = MHA(config)
        self.MLP = MLP(config) 

        # MLP for conditioning info
        self.cLinWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size * 6, rngs=config.rngs)
    
    def __call__(self, x, conditioning):
        """Uses vmap to process batched inputs."""
        func = lambda x, condit: self.forward(x, condit)
        return vmap(func, in_axes=(0, 0), out_axes=0)(x, conditioning)

    def forward(self, x, conditioning):
        print(x.shape, conditioning.shape)
        # assert 1 == 2
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.cLinWeights(conditioning).reshape((6, -1))
        tmp = self.LayerNorm1(x)
        tmp = gamma1*tmp + beta1 
        tmp, attn = self.MHA(tmp)
        tmp = alpha1*tmp

        tmp += x
        x = tmp

        tmp = self.LayerNorm2(tmp)
        tmp = gamma2*tmp + beta2 
        tmp = self.MLP(tmp)
        tmp = alpha2*tmp

        x += tmp

        return x

class DiTFinalLayer(nnx.Module):
    def __init__(self, config: modelConfig):
        
        self.LayerNorm = nnx.LayerNorm(config.DiT_hidden_size, rngs=config.rngs)
        self.linear = nnx.Linear(config.DiT_hidden_size, config.patch_size**2*4) 
        self.linWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size*2)

    def forward(self, x, conditioning):
        x = self.LayerNorm(x)
        alpha, beta = self.linWeights(conditioning).value
        x = alpha * x + beta
        x = self.linear(x)
        return x

class DiTPatch(nnx.Module):
    def __init__(self, config: modelConfig):
        self.config = config
    
    def convert_to_patches(self, input):
        input = input.reshape((self.patch_size, self.patch_size, 4))
        return input
    
    def convert_to_stream(self, input):
        input = input.reshape(-1)
        return input
        
class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer"""
    def __init__(self, config: modelConfig):
        
        self.config = config
        self.length = (32/self.config.patch_size)**2
        self.layers = [
                        DiTBlock(config) for _ in range(self.config.n_layer)
                      ]
        self.final_layer = DiTFinalLayer(config)
        self.mapper = DiTPatch(config)

    def forward(self, x, conditioning):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.mapper.convert_to_stream(x)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)


        # TODO: embedding parts,
        for layer in self.layers:
            x = layer.forward(x, conditioning)
        x = self.final_layer(x)
        return self.mapper.convert_to_patches(x)
            

    def __call__(self, something): # TODO: fix
        return self.forward(something)

# Testing
config = modelConfig()

batched = True
if not batched:
  test_input = jnp.ones((config.token_length, config.DiT_hidden_size))
  test_condit = jnp.ones((config.DiT_hidden_size))

  test_MHA = MHA(config)
  x, attn = test_MHA(test_input)
  print(test_input.shape, x.shape)


  test_MLP = MLP(config)
  x = test_MLP(test_input)
  print(test_input.shape, x.shape)

  test_DiTBlock = DiTBlock(config)
  x = test_DiTBlock(test_input, test_condit)
  print(test_input.shape, x.shape)

else:
  batch = 8
  test_input = jnp.ones((batch, config.token_length, config.DiT_hidden_size))
  test_condit = jnp.ones((batch, config.DiT_hidden_size))

  test_DiTBlock = DiTBlock(config)
  x = test_DiTBlock(test_input, test_condit)
  print(test_input.shape, x.shape)


# To consider: change all lin shifts to (1+scale)


# class Model(nnx.Module):
#   def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
#     
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



