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
        """Implements scaled dot product with Pyjnp's functionality"""
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
        self.input_embeddings = MLP(config)

    def convert_to_patches(self, input):
        input = input.reshape((self.patch_size, self.patch_size, 4))
        input = self.input_embeddings(input)
        return input
    
    def convert_to_stream(self, input):
        input = input.reshape(-1)
        return input

class DiffusionTransformer(nnx.Module):
    """Diffusion Transformer"""
    def __init__(self, config: modelConfig):
        
        self.config = config
        self.length = config.token_length 
        self.layers = [
                        DiTBlock(config) for _ in range(self.config.n_layer)
                      ]
        self.final_layer = DiTFinalLayer(config)
        self.mapper = DiTPatch(config)
        self.time_MLP = MLP()

        self.pos_embed = self.pos_embed() #TODO: freeze these values.
    
    def pos_embed(self):
        # Implements: pos / 10000^(2i/d_model)
        # Implementation from https://medium.com/thedeephub/positional-encoding-explained-a-deep-dive-into-transformer-pe-65cfe8cfe10b  
        position = jnp.arange(self.config.token_length)[:, jnp.newaxis]
        # The original formula pos / 10000^(2i/d_model) is equivalent to pos * (1 / 10000^(2i/d_model)).
        # I use the below version for numerical stability
        div_term = jnp.exp(jnp.arange(0, self.config.DiT_hidden_size, 2) * -(jnp.log(10000.0) / self.config.DiT_hidden_size))
        
        pe = jnp.zeros((self.config.token_length, self.config.DiT_hidden_size))
        pe[:, 0::2] = jnp.sin(position * div_term)
        pe[:, 1::2] = jnp.cos(position * div_term)
        
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
        half = self.config.DiT_hidden_size // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, end=half, dtype=jnp.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = jnp.cat([jnp.cos(args), jnp.sin(args)], dim=-1)
        if self.config.DiT_hidden_size % 2:
            embedding = jnp.cat([embedding, jnp.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, timestep):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        print(x.shape, timestep.shape)
        x = self.mapper.convert_to_stream(x) # Convert [32, 32, 4] to [32*32*4] & applies MLP
        print(x.shape)
        x = x + self.pos_embed # Adds sinusoidal PE
        print(x.shape)
        
        conditioning = self.time_MLP(self.time_embed(timestep)) # Embeds single timestep to [hidden_dim]
        print(conditioning.shape)

        for layer in self.layers:
            x = layer.forward(x, conditioning)
        x = self.final_layer(x)
        return self.mapper.convert_to_patches(x)
            

    def __call__(self, something): # TODO: fix
        return self.forward(something)

if __name__=="__main__":
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



