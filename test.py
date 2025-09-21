from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from flax import nnx
from helpers.config import modelConfig, trainConfig
import optax
import argparse
from helpers.preprocess_data_torch import load_latents
from tqdm import tqdm
import os
from glob import glob
from copy import deepcopy
import optax
import orbax.checkpoint as ocp
import time
import jax_dataloader as jdl

modelconfig = modelConfig()
model = DiffusionTransformer(modelconfig)



checkpoint_path = "results/experiment-37/models/ckpt_2/"
checkpoint_path = os.path.abspath(checkpoint_path)
abstract_model = nnx.eval_shape(lambda: DiffusionTransformer(modelconfig))

# Split into graphdef and abstract state
_, abstract_state = nnx.split(abstract_model)

# Initialize the checkpointer
checkpointer = ocp.StandardCheckpointer()

# Restore the checkpoint while passing the target tree for matching types/shapes
restored_ckpt = checkpointer.restore(
    checkpoint_path,
    args=ocp.args.StandardRestore(abstract_state)
)



# Extract components
model_state_restored = restored_ckpt['model']
epoch_restored = restored_ckpt['epoch']
config_restored = restored_ckpt['config']


nnx.update(model, model_state_restored)
