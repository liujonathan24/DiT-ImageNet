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
        # print(x.shape) # N, C
        y = self.qkv_proj(x) # N, 3*C
        y = y.reshape(self.length, 3, self.num_heads, self.d_head) # N, 3, H, d_H
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
        self.fc1 = nnx.Linear(config.DiT_hidden_size, config.MLP_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc1_rng), dtype=self.dtype, use_bias=True)
        self.act = lambda t: nnx.gelu(t, approximate=True)
        self.fc2 = nnx.Linear(config.MLP_hidden_size, config.DiT_hidden_size, kernel_init=xavier_init, rngs=nnx.Rngs(fc2_rng), dtype=self.dtype, use_bias=True)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TimeMLP(nnx.Module):
    # TODO: init as
    # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
    # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

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

class LabelEmbed(nnx.Module):
    def __init__(self, config, rng):
        self.dropout_prob = config.dropout_prob
        self.num_classes = config.num_classes
        self.embedder = nnx.Embed(num_embeddings=config.num_classes + 1, features=config.DiT_hidden_size, rngs=nnx.Rngs(rng), dtype=config.dtype)
    
    def __call__(self, label, train: bool, *, rngs: dict | None = None):
        if train and self.dropout_prob > 0:
            assert rngs is not None, "RNGs must be provided for training with dropout."
            dropout_key = rngs['dropout']
            # Decide whether to drop for each item in the batch
            drop_mask = jax.random.uniform(dropout_key, shape=label.shape) < self.dropout_prob
            # The unconditional class label is num_classes
            unconditional_label = self.num_classes
            label = jnp.where(drop_mask, unconditional_label, label)
        
        return self.embedder(label)

def zero_init(key, shape, dtype):
    value = jnp.zeros(shape, dtype=dtype)
    return value

class DiTBlock(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        self.dtype = config.dtype
        ln1_rng, ln2_rng, mha_rng, mlp_rng, cLin_rng = jax.random.split(rng, 5)

        self.LayerNorm1 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln1_rng), epsilon=1e-6, use_scale=False, use_bias=False, dtype=self.dtype)
        self.LayerNorm2 = nnx.LayerNorm(config.DiT_hidden_size, rngs=nnx.Rngs(ln2_rng), epsilon=1e-6, use_scale=False, use_bias=False, dtype=self.dtype)
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
        x = x + tmp
        
        tmp = self.LayerNorm2(x)
        tmp = (alpha2+1)*tmp + beta2
        tmp = self.MLP(tmp)
        tmp = gamma2 * tmp
        x = x + tmp
        return x

class DiTFinalLayer(nnx.Module):
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        self.dtype = config.dtype
        ln_rng, linear_rng, lin_weights_rng = jax.random.split(rng, 3)
        self.LayerNorm = nnx.LayerNorm(config.DiT_hidden_size, epsilon=1e-6, rngs=nnx.Rngs(ln_rng), use_scale=False, use_bias=False, dtype=self.dtype)
        self.linear = nnx.Linear(config.DiT_hidden_size, config.patch_size**2 * config.output_channels, kernel_init=zero_init, rngs=nnx.Rngs(linear_rng), dtype=self.dtype)
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
    # TODO: initialize as: 
    # # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)
    def __init__(self, config: modelConfig, rng: jax.random.PRNGKey):
        self.output_dim = config.output_dim
        self.patch_size = (config.patch_size, config.patch_size)
        self.output_channels = config.output_channels
        self.token_length = config.token_length
        self.DiT_hidden_size = config.DiT_hidden_size
        self.dtype = config.dtype
        
        self.patch_embeddings = nnx.Conv(
            in_features=config.image_channels, # TODO: rename to in_channels         
            out_features=config.DiT_hidden_size,      
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='VALID',                           
            rngs=nnx.Rngs(rng),
            dtype=self.dtype
        )

    def convert_to_patches(self, input):
        x = input.reshape((self.output_dim, self.output_dim, self.patch_size[0], self.patch_size[1], self.output_channels))
        x = jnp.einsum('hwpqc->chpwq', x)
        imgs = x.reshape((self.output_channels, self.output_dim*self.patch_size[0], self.output_dim*self.patch_size[1]))
        return imgs
    
    def convert_to_stream(self, input):
        # Input = [4, 32, 32]
        input = jnp.einsum('chw->hwc', input)
        input = self.patch_embeddings(input) # 32/patch, 32/patch, Hidden
        input = input.reshape(self.token_length, self.DiT_hidden_size) # L, Hidden
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

        # Embedders:
        self.mapper = DiTPatch(config, mapper_rng)
        self.pos_embed = nnx.Param(self._get_pos_embed()) # TODO: make sure this guy is not trainable

        self.time_MLP = TimeMLP(config, time_mlp_rng)
        self.y_embedder = LabelEmbed(config, y_embed_rng)

        # DiT Blocks
        self.layers = nnx.List([
                 DiTBlock(config, layer_rngs[i]) for i in range(self.n_layers)
                 ])
        # Final layer
        self.final_layer = DiTFinalLayer(config, final_rng)
    
    def _get_pos_embed(self):
        """
        Build 2D sinusoidal positional embeddings.
        """
        embed_dim = self.DiT_hidden_size
        grid_size = int(self.length ** 0.5)
        
        def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
            assert embed_dim % 2 == 0
            omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
            omega /= embed_dim / 2.
            omega = 1. / 10000**omega  # (D/2,)

            pos = pos.reshape(-1)  # (M,)
            out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

            emb_sin = jnp.sin(out) # (M, D/2)
            emb_cos = jnp.cos(out) # (M, D/2)

            emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
            return emb

        assert embed_dim % 2 == 0
        
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, jnp.arange(grid_size, dtype=jnp.float32))
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, jnp.arange(grid_size, dtype=jnp.float32))
        
        grid_h = jnp.arange(grid_size, dtype=jnp.float32)
        grid_w = jnp.arange(grid_size, dtype=jnp.float32)
        grid = jnp.meshgrid(grid_w, grid_h)
        grid = jnp.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])

        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

        pos_embed = jnp.concatenate([emb_h, emb_w], axis=1)

        return pos_embed.astype(self.dtype)


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

    def forward(self, x, timestep, y, train: bool, *, rngs: dict | None = None):
        x = self.mapper.convert_to_stream(x)
        x = x + self.pos_embed
        
        time_embedding = self.time_MLP(self.time_embed(timestep))
        
        y_array = jnp.array([y])
        class_embedding = self.y_embedder(y_array, train=train, rngs=rngs).squeeze(axis=0)
        conditioning = time_embedding + class_embedding

        for i, layer in enumerate(self.layers):
            x = layer.forward(x, conditioning)
            
        x = self.final_layer(x, conditioning)
        x = self.mapper.convert_to_patches(x)
        return x
    
    def forward_with_cfg(self, x, t, y, cfg_scale, train=False, rngs=None):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = jnp.concatenate([half, half], axis=0)
        
        model_out = self(combined, t, y, train=train, rngs=rngs)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = jnp.split(eps, 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = jnp.concatenate([half_eps, half_eps], axis=0)
        return jnp.concatenate([eps, rest], axis=1)
            
    def get_weights(self):
        return nnx.state(self)

    def set_weights(self, weights):
        nnx.update(self, weights)

    def __call__(self, x, timestep, y, train: bool, *, rngs: dict | None = None):

        func = lambda x, timestep, y: self.forward(x, timestep, y, train=train, rngs=rngs)
        return vmap(func, in_axes=(0, 0, 0), out_axes=0)(x, timestep, y)

        print("!!! WARNING: Running without vmap for debugging. This will be slow. !!!")
        # Manually loop over the batch dimension for debugging prints
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            print(f"--- Debugging Batch Item {i+1}/{batch_size} ---")
            x_i = x[i]
            # The timestep is a single value, but the forward pass might expect it to be shaped like a batch of 1
            timestep_i = timestep[i]
            y_i = y[i]
            output_i = self.forward(x_i, timestep_i, y_i)
            outputs.append(output_i)
        return jnp.stack(outputs, axis=0)

if __name__=="__main__":
  config = modelConfig(type='DiT-XL')
  rng = jax.random.PRNGKey(0)

  batch = 8
  test_input = jax.random.uniform(rng, shape=(batch, 8, 32, 32), dtype=config.dtype)
  test_timesteps = jnp.ones(batch)
  test_labels = jnp.ones(batch, dtype=jnp.int32) * 0
  
  test_DiT = DiffusionTransformer(config)
  x = test_DiT(test_input, test_timesteps, test_labels)
  print(jnp.mean(x-test_input))
  print(x, test_input)
  print(x.shape)
