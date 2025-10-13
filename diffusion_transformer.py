import jax 
import jax.numpy as jnp
from jax import vmap
from flax import nnx
from helpers.config import modelConfig

xavier_init = nnx.initializers.xavier_uniform()

class MHA(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        
        self.length = config.token_length
        self.hidden = config.DiT_hidden_size
        self.num_heads = config.n_heads
        
        self.d_head = self.hidden // self.num_heads
        qkv_rng, out_rng = jax.random.split(rng)
        self.qkv_proj = nnx.Linear(self.hidden, 3 * self.hidden, kernel_init=xavier_init, rngs=nnx.Rngs(qkv_rng))
        self.out_proj = nnx.Linear(self.hidden, self.hidden, kernel_init=xavier_init, rngs=nnx.Rngs(out_rng))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x: [L, Hidden]
        y = self.qkv_proj(x)                     # [L, 3H]
        y = y.reshape(self.length, 3, self.num_heads, self.d_head)   # [L, 3, head, d]
        y = jnp.transpose(y, (1, 2, 0, 3))       # [3, h, L, d]
        q, k, v = y[0], y[1], y[2]               # each [h, L, d]

        values, attention = self.scaled_dot_product(q, k, v)  # values: [h, L, d]
        values = jnp.transpose(values, (1, 0, 2)).reshape(self.length, self.hidden)  # [L, H]
        values = self.out_proj(values)           # [L, H]
        return values, attention                 # attention: [h, L, L]
  
    def scaled_dot_product(self, q, k, v):
        """Implements scaled dot product with jax.numpy's functionality"""
        # q, k, v: [num_heads, L, d_head]
        d = q.shape[-1]
        scale = 1.0 / jnp.sqrt(d)
        # [num_heads, L, L]
        attn_logits = jnp.einsum('hld,hmd->hlm', q, k) * scale
        attn = jax.nn.softmax(attn_logits, axis=-1)
        # [num_heads, L, d_head]
        out = jnp.einsum('hlm,hmd->hld', attn, v)
        return out, attn


class MLP(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        """
        Inititializes a MLP block. Has reduced 
        options compared to ViT implementations, but 
        sufficient for the task. Uses GeLU activations.
        """
        fc1_rng, fc2_rng = jax.random.split(rng)
        self.fc1 = nnx.Linear(config.DiT_hidden_size, config.MLP_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc1_rng))
        self.act = lambda t: nnx.gelu(t, approximate=True)
        self.fc2 = nnx.Linear(config.MLP_hidden_size, config.DiT_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc2_rng))
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TimeMLP(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        fc1_rng, fc2_rng = jax.random.split(rng)
        self.fc1 = nnx.Linear(config.time_embed_dim, config.DiT_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc1_rng))
        self.act = nnx.silu
        self.fc2 = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc2_rng))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)

def zero_init(key, shape, dtype):
    value = jnp.zeros(shape)
    return value

class DiTBlock(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        """
        Initialize a DiT block.
        """
        
        ln1_rng, ln2_rng, mha_rng, mlp_rng, cLin_rng = jax.random.split(rng, 5)
        self.LayerNorm1 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln1_rng), use_scale=False, use_bias=False)
        self.LayerNorm2 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln2_rng), use_scale=False, use_bias=False)
        self.MHA = MHA(config, mha_rng)
        self.MLP = MLP(config, mlp_rng) 

        # MLP for conditioning info
        self.cLinWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size * 6, rngs=nnx.Rngs(cLin_rng), kernel_init=zero_init)  
    
    def __call__(self, x, conditioning):
        """Uses vmap to process batched inputs."""
        return self.forward(x, conditioning)

    def forward(self, x, conditioning):
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.cLinWeights(nnx.silu(conditioning)).reshape((6, -1))

        tmp = self.LayerNorm1(x)
        tmp = (alpha1+1)*tmp + beta1
        tmp, attn = self.MHA(tmp)
        tmp = gamma1 * tmp

        x += tmp

        tmp = self.LayerNorm2(x)
        tmp = (alpha2+1)*tmp + beta2
        tmp = self.MLP(tmp)
        tmp = gamma2 * tmp

        x += tmp
        return x

class DiTFinalLayer(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        ln_rng, linear_rng, lin_weights_rng = jax.random.split(rng, 3)
        self.LayerNorm = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln_rng), use_scale=False, use_bias=False)
        self.linear = nnx.Linear(config.DiT_hidden_size, config.patch_size**2 * config.output_channels, kernel_init=xavier_init, rngs=nnx.Rngs(linear_rng))
        self.linWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size*2, rngs=nnx.Rngs(lin_weights_rng), kernel_init=zero_init)

    def forward(self, x, conditioning):
        x = self.LayerNorm(x)
        alpha, beta = self.linWeights(nnx.silu(conditioning)).reshape((2, -1))
        x = (1+alpha) * x + beta
        x = self.linear(x)
        return x

    def __call__(self, x, conditioning):
        return self.forward(x, conditioning)

class DiTPatch(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        self.output_dim = config.output_dim
        self.patch_size = config.patch_size
        self.output_channels = config.output_channels
        self.token_length = config.token_length
        self.DiT_hidden_size = config.DiT_hidden_size
        
        self.patch_embeddings = nnx.Conv(
            in_features=config.image_channels,         
            out_features=config.DiT_hidden_size,      
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding='VALID',                           
            rngs=nnx.Rngs(rng),
        )

    def convert_to_patches(self, input):
        x = input.reshape((self.output_dim, self.output_dim, self.patch_size, self.patch_size, self.output_channels))
        x = jnp.einsum('hwpqc->chpwq', x)
        imgs = x.reshape((self.output_channels, self.output_dim*self.patch_size, self.output_dim*self.patch_size))
        return imgs
    
    def convert_to_stream(self, input):
        input = jnp.einsum('chw->hwc', input)
        input = self.patch_embeddings(input)
        input = input.reshape(self.token_length, self.DiT_hidden_size)
        return input

class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer"""
    def __init__(self, config: modelConfig):
        self.config = config
        self.length = config.token_length 
        self.DiT_hidden_size = config.DiT_hidden_size
        self.n_layers = config.n_layers

        rng, layer_rng, final_rng, mapper_rng, time_mlp_rng = jax.random.split(config.rngs.params(), 5)
        layer_rngs = jax.random.split(layer_rng, self.n_layers)

        self.layers = nnx.List([
                 DiTBlock(config, layer_rngs[i]) for i in range(self.n_layers)
                 ])
        self.final_layer = DiTFinalLayer(config, final_rng)
        self.mapper = DiTPatch(config, mapper_rng)
        self.time_MLP = TimeMLP(config, time_mlp_rng)

        self.pos_embed = self.pos_embed()
    
    def pos_embed(self):
        position = jnp.arange(self.length)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.DiT_hidden_size, 2) * -(jnp.log(10000.0) / self.DiT_hidden_size))
        pe = jnp.zeros((self.length, self.DiT_hidden_size))
        pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return pe

    def time_embed(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        dim = self.config.time_embed_dim
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = jnp.float32(t) * freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros(1)], axis=-1)
        return embedding

    def forward(self, x, timestep):
        """
        Forward pass of DiT.
        """
        x = self.mapper.convert_to_stream(x)
        x = x + self.pos_embed
        conditioning = self.time_MLP(self.time_embed(timestep))

        for layer in range(self.n_layers):
            x = self.layers[layer].forward(x, conditioning)
        x = self.final_layer(x, conditioning)
        x = self.mapper.convert_to_patches(x)
        return x
            
    def get_weights(self):
        return nnx.state(self)

    def set_weights(self, weights):
        nnx.update(self, weights)

    def __call__(self, x, conditioning): 
        """Uses vmap to process batched inputs."""
        func = lambda x, conditioning: self.forward(x, conditioning)
        return vmap(func, in_axes=(0, 0), out_axes=0)(x, conditioning)

if __name__=="__main__":
  # Testing
  config = modelConfig(type='DiT-XL')
  rng = jax.random.PRNGKey(0)

  batch = 8
  test_input = jnp.ones((batch, 4, 32, 32))
  test_timesteps = jnp.ones(batch)
  
  test_DiT = DiffusionTransformer(config)
  x = test_DiT(test_input, test_timesteps)
  print(x.shape)
