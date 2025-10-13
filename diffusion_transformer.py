import jax 
import jax.numpy as jnp
from jax import vmap
from flax import nnx
from helpers.config import modelConfig

xavier_init = nnx.initializers.xavier_uniform()

def print_stats_jax(tensor, name="Tensor"):
    if jnp.any(jnp.isnan(tensor)):
        print(f"!!! {name} contains NaN values !!!")
    mean, std, min_val, max_val = float(jnp.mean(tensor)), float(jnp.std(tensor)), float(jnp.min(tensor)), float(jnp.max(tensor))
    print(
        f"{name} stats: "
        f"mean={mean:.4f}, std={std:.4f}, "
        f"min={min_val:.4f}, max={max_val:.4f}, shape={tensor.shape}, dtype={tensor.dtype}"
    )

class MHA(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        
        self.length = config.token_length
        self.hidden = config.DiT_hidden_size
        self.num_heads = config.n_heads
        self.dtype = config.dtype
        
        self.d_head = self.hidden // self.num_heads
        qkv_rng, out_rng = jax.random.split(rng)
        self.qkv_proj = nnx.Linear(self.hidden, 3 * self.hidden, kernel_init=xavier_init, rngs=nnx.Rngs(qkv_rng), dtype=self.dtype)
        self.out_proj = nnx.Linear(self.hidden, self.hidden, kernel_init=xavier_init, rngs=nnx.Rngs(out_rng), dtype=self.dtype)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        y = self.qkv_proj(x)
        y = y.reshape(self.length, 3, self.num_heads, self.d_head)
        y = jnp.transpose(y, (1, 2, 0, 3))
        q, k, v = y[0], y[1], y[2]

        values, attention = self.scaled_dot_product(q, k, v)
        values = jnp.transpose(values, (1, 0, 2)).reshape(self.length, self.hidden)
        values = self.out_proj(values)
        return values, attention
  
    def scaled_dot_product(self, q, k, v):
        d = q.shape[-1]
        scale = 1.0 / jnp.sqrt(d)
        attn_logits = jnp.einsum('hld,hmd->hlm', q, k)
        attn_logits = attn_logits * scale
        attn = jax.nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum('hlm,hmd->hld', attn, v)
        return out, attn


class MLP(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        fc1_rng, fc2_rng = jax.random.split(rng)
        self.dtype = config.dtype
        self.fc1 = nnx.Linear(config.DiT_hidden_size, config.MLP_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc1_rng), dtype=self.dtype)
        self.act = lambda t: nnx.gelu(t, approximate=True)
        self.fc2 = nnx.Linear(config.MLP_hidden_size, config.DiT_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc2_rng), dtype=self.dtype)
    
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
        self.dtype = config.dtype
        self.fc1 = nnx.Linear(config.time_embed_dim, config.DiT_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc1_rng), dtype=self.dtype)
        self.act = nnx.silu
        self.fc2 = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc2_rng), dtype=self.dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)

def zero_init(key, shape, dtype):
    value = jnp.zeros(shape, dtype=dtype)
    return value

class DiTBlock(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        self.dtype = config.dtype
        ln1_rng, ln2_rng, mha_rng, mlp_rng, cLin_rng = jax.random.split(rng, 5)
        self.LayerNorm1 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln1_rng), use_scale=False, use_bias=False, dtype=self.dtype)
        self.LayerNorm2 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln2_rng), use_scale=False, use_bias=False, dtype=self.dtype)
        self.MHA = MHA(config, mha_rng)
        self.MLP = MLP(config, mlp_rng) 
        self.cLinWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size * 6, rngs=nnx.Rngs(cLin_rng), kernel_init=zero_init, dtype=self.dtype)  
    
    def __call__(self, x, conditioning):
        return self.forward(x, conditioning)

    def forward(self, x, conditioning):
        beta1, alpha1, gamma1, beta2, alpha2, gamma2 = self.cLinWeights(nnx.silu(conditioning)).reshape((6, -1))
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
        self.dtype = config.dtype
        ln_rng, linear_rng, lin_weights_rng = jax.random.split(rng, 3)
        self.LayerNorm = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln_rng), use_scale=False, use_bias=False, dtype=self.dtype)
        self.linear = nnx.Linear(config.DiT_hidden_size, config.patch_size**2 * config.output_channels, kernel_init=xavier_init, rngs=nnx.Rngs(linear_rng), dtype=self.dtype)
        self.linWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size*2, rngs=nnx.Rngs(lin_weights_rng), kernel_init=zero_init, dtype=self.dtype)

    def forward(self, x, conditioning):
        x = self.LayerNorm(x)
        beta, alpha = self.linWeights(nnx.silu(conditioning)).reshape((2, -1))
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
        self.dtype = config.dtype
        
        self.patch_embeddings = nnx.Conv(
            in_features=config.image_channels,         
            out_features=config.DiT_hidden_size,      
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding='VALID',                           
            rngs=nnx.Rngs(rng),
            dtype=self.dtype
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
        self.dtype = config.dtype

        rng, layer_rng, final_rng, mapper_rng, time_mlp_rng, y_embed_rng = jax.random.split(config.rngs.params(), 6)
        layer_rngs = jax.random.split(layer_rng, self.n_layers)

        self.y_embedder = nnx.Embed(num_embeddings=config.num_classes, features=config.DiT_hidden_size, rngs=nnx.Rngs(y_embed_rng), dtype=self.dtype)

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
        pe = jnp.zeros((self.length, self.DiT_hidden_size), dtype=self.dtype)
        pe.at[:, 0::2].set(jnp.sin(position * div_term).astype(self.dtype))
        pe.at[:, 1::2].set(jnp.cos(position * div_term).astype(self.dtype))
        return pe

    def time_embed(self, t, max_period=10000):
        dim = self.config.time_embed_dim
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = jnp.float32(t) * freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros(1)], axis=-1)
        return embedding.astype(self.dtype)

    def forward(self, x, timestep, y):
        print("--- Start of Forward Pass ---")
        print_stats_jax(x, name="Input x")
        
        x = self.mapper.convert_to_stream(x)
        print_stats_jax(x, name="After convert_to_stream")
        
        x = x + self.pos_embed
        print_stats_jax(x, name="After pos_embed")
        
        time_embedding = self.time_MLP(self.time_embed(timestep))
        class_embedding = self.y_embedder(y)
        conditioning = time_embedding + class_embedding
        print_stats_jax(conditioning, name="Combined Conditioning")

        for i, layer in enumerate(self.layers):
            x = layer.forward(x, conditioning)
            print_stats_jax(x, name=f"After DiT Block {i}")
            
        x = self.final_layer(x, conditioning)
        print_stats_jax(x, name="After final_layer")
        
        x = self.mapper.convert_to_patches(x)
        print_stats_jax(x, name="Final Output")
        print("--- End of Forward Pass ---")
        return x
            
    def get_weights(self):
        return nnx.state(self)

    def set_weights(self, weights):
        nnx.update(self, weights)

    def __call__(self, x, timestep, y): 
        func = lambda x, timestep, y: self.forward(x, timestep, y)
        return vmap(func, in_axes=(0, 0, 0), out_axes=0)(x, timestep, y)

if __name__=="__main__":
  config = modelConfig(type='DiT-XL')
  rng = jax.random.PRNGKey(0)

  batch = 8
  test_input = jnp.ones((batch, 4, 32, 32), dtype=config.dtype)
  test_timesteps = jnp.ones(batch)
  test_labels = jnp.ones(batch, dtype=jnp.int32)
  
  test_DiT = DiffusionTransformer(config)
  x = test_DiT(test_input, test_timesteps, test_labels)
  print(x.shape)