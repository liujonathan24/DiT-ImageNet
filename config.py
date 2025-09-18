class Config:
    def __init__(self):
        self.model_type = 'DiT-S'
        self.n_layer = 12
        self.n_head = 6
        self.n_embd = 384
        self.learning_rate = 1e-4 
        self.ema = 0.9999
