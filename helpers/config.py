from flax import nnx
class modelConfig:
    def __init__(self):
        self.model_type = 'DiT-S'
        self.rngs = nnx.Rngs(params=42)

        # Patch details
        self.input_size = 32
        self.image_channels = 4
        self.patch_size = 8

        # output details
        self.output_dim = 4
        self.output_channels = 4

        # Architecture details
        self.n_layers = 12
        self.n_heads = 6
        self.DiT_hidden_size = 384
        self.MLP_hidden_size = self.DiT_hidden_size * 4 # TODO: consider adjusting this. 
        self.token_length = int((self.input_size/self.patch_size)**2)

    def to_dict(self):
        # Return a serializable dict representation, excluding non-serializable fields
        return {
            "model_type": self.model_type,
            # Optionally store a serializable version of rngs if needed, or exclude:
            #"rngs": None,
            "input_size": self.input_size,
            "image_channels": self.image_channels,
            "patch_size": self.patch_size,
            "output_dim": self.output_dim,
            "output_channels": self.output_channels,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "DiT_hidden_size": self.DiT_hidden_size,
            "MLP_hidden_size": self.MLP_hidden_size,
            "token_length": self.token_length,
        }

class trainConfig:
    def __init__(self):
        # Training details
        self.batch_size = 64 # 256
        self.epochs = 51 # 1m

        self.learning_rate = 1e-4
        self.ema = 0.9999

        self.tmax = 1000
        self.linear_variance_min = 1e-4
        self.linear_variance_max = 2e-2

        self.ckpt_frequency = 5
        self.log_frequency = 2000
    
    def to_dict(self):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "ema": self.ema,
            "tmax": self.tmax,
            "linear_variance_min": self.linear_variance_min,
            "linear_variance_max": self.linear_variance_max,
            "ckpt_frequency": self.ckpt_frequency,
            "log_frequency": self.log_frequency
        }
