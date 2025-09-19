from flax import nnx
class modelConfig:
    def __init__(self):
        self.model_type = 'DiT-S'
        self.rngs = nnx.Rngs(42)

        # Patch details
        self.input_size = 32
        self.image_channels = 4
        self.patch_size = 8

        # Architecture details
        self.n_layers = 12
        self.n_heads = 6
        self.DiT_hidden_size = 384
        self.MLP_hidden_size = self.DiT_hidden_size * 4 # TODO: consider adjusting this. 
        self.token_length = int((self.input_size/self.patch_size)**2 * self.image_channels)

class trainConfig:
    def __init__(self):
        # Training details
        self.batch_size = 64 # 256
        self.epochs = 1 # 1m

        self.learning_rate = 1e-4
        self.ema = 0.9999

        self.tmax = 1000 
        self.linear_variance_min = 1e-4
        self.linear_variance_max = 2e-2
        
