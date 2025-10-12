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

def zero_init(key, shape, dtype):
    value = jnp.zeros(shape)
    return value

class DiTBlock(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        """
        Initialize a DiT block.
        """
        
        # self.config = config
        ln1_rng, ln2_rng, mha_rng, mlp_rng, cLin_rng, cScale_rng = jax.random.split(rng, 6)
        self.LayerNorm1 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln1_rng), use_scale=False, use_bias=False)
        self.LayerNorm2 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln2_rng), use_scale=False, use_bias=False)
        self.MHA = MHA(config, mha_rng)
        self.MLP = MLP(config, mlp_rng) 

        # MLP for conditioning info
        self.cLinWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size * 4, rngs=nnx.Rngs(cLin_rng), kernel_init=zero_init)  
        self.cScaleWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size * 2, rngs=nnx.Rngs(cScale_rng), kernel_init=xavier_init)
    
    def __call__(self, x, conditioning):
        """Uses vmap to process batched inputs."""
        return self.forward(x, conditioning)

    def forward(self, x, conditioning):
        alpha1, beta1, alpha2, beta2 = self.cLinWeights(nnx.silu(conditioning)).reshape((4, -1))
        gamma1, gamma2 = self.cScaleWeights(nnx.silu(conditioning)).reshape((2, -1))

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
        # print("init final layer")
        # print(config.patch_size, config.output_dim)
        self.linear = nnx.Linear(config.DiT_hidden_size, config.patch_size**2*config.output_channels, kernel_init=zero_init, rngs=nnx.Rngs(linear_rng)) 
        self.linWeights = nnx.Linear(config.DiT_hidden_size, config.DiT_hidden_size*2, rngs=nnx.Rngs(lin_weights_rng), kernel_init=zero_init)

    def forward(self, x, conditioning):
        x = self.LayerNorm(x)
        # print(f"layer norm: {x.shape}")
        alpha, beta = self.linWeights(nnx.silu(conditioning)).reshape((2, -1))
        x = (1+alpha) * x + beta
        # print(f"scaled: {x.shape}")
        x = self.linear(x)
        # print(f"x shape: {x.shape}")
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
        # print(input.shape)
        x = input.reshape((self.output_dim, self.output_dim, self.patch_size, self.patch_size, self.output_channels))
        x = jnp.einsum('hwpqc->chpwq', x)
        imgs = x.reshape((self.output_channels, self.output_dim*self.patch_size, self.output_dim*self.patch_size))
        return imgs
    
    def convert_to_stream(self, input):
        input = jnp.einsum('chw->hwc', input)
        input = self.patch_embeddings(input)
        input = input.reshape(self.token_length, self.DiT_hidden_size)
        # print(f"After stream: {input.shape}")
        return input

class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer"""
    def __init__(self, config: modelConfig):
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
        self.time_MLP = MLP(config, time_mlp_rng)

        self.pos_embed = self.pos_embed()
    
    def pos_embed(self):
        # Implements: pos / 10000^(2i/d_model)
        # Implementation from https://medium.com/thedeephub/positional-encoding-explained-a-deep-dive-into-transformer-pe-65cfe8cfe10b  
        position = jnp.arange(self.length)[:, jnp.newaxis]
        # The original formula pos / 10000^(2i/d_model) is equivalent to pos * (1 / 10000^(2i/d_model)).
        # I use the below version for numerical stability
        div_term = jnp.exp(jnp.arange(0, self.DiT_hidden_size, 2) * -(jnp.log(10000.0) / self.DiT_hidden_size))
        print(f"div_term: {div_term}")
        print(f"position: {position}")
        
        pe = jnp.zeros((self.length, self.DiT_hidden_size))
        pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        return pe

    def time_embed(self, t, max_period=10000):
        """
        From: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py 
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = self.DiT_hidden_size // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = jnp.float32(t) * freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if self.DiT_hidden_size % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros(1)], axis=-1)
        return embedding

    def forward(self, x, timestep):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # print(x.shape)
        x = self.mapper.convert_to_stream(x) # Convert [4, 32, 32] to [4*32*32] & applies MLP
        x = x + self.pos_embed # Adds sinusoidal PE
        # print(f"Added pos_embeds: {x.shape}")
        conditioning = self.time_MLP(self.time_embed(timestep)) # Embeds single timestep to [hidden_dim]
        # print(f"Conditioning shape: {conditioning.shape}")


        for layer in range(self.n_layers):
            x = self.layers[layer].forward(x, conditioning)
            # print(f"Shape after layer: {x.shape}")
        x = self.final_layer(x, conditioning)
        # print(f"Post final layer: {x.shape}")
        x = self.mapper.convert_to_patches(x)
        # print(f"Back to patches: {x.shape}")
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
  config = modelConfig()
  rng = jax.random.PRNGKey(0)

  batched = True
  if not batched:
    test_input = jnp.ones((config.token_length, config.DiT_hidden_size))
    test_condit = jnp.ones((config.DiT_hidden_size))

    mha_rng, mlp_rng, dit_block_rng = jax.random.split(rng, 3)
    test_MHA = MHA(config, mha_rng)
    x, attn = test_MHA(test_input)
    print(test_input.shape, x.shape)


    test_MLP = MLP(config, mlp_rng)
    x = test_MLP(test_input)
    print(test_input.shape, x.shape)

    test_DiTBlock = DiTBlock(config, dit_block_rng)
    x = test_DiTBlock(test_input, test_condit)
    print(test_input.shape, x.shape)

  else:
    batch = 8
    test_input = jnp.ones((batch, config.token_length, config.DiT_hidden_size))
    test_condit = jnp.ones((batch, config.DiT_hidden_size))
    test_timesteps = jnp.ones(batch)
    
    test_input = jnp.ones((batch, 4, 32, 32))
    test_DiT = DiffusionTransformer(config, rng)
    x = test_DiT(test_input, test_timesteps)
    print(x.shape)
