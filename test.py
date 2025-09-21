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



#abstract_model = nnx.eval_shape(lambda: DiffusionTransformer(model_config))

# 3. Create a random key for initialization
key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(params=key)

# 4. Use nnx.init to initialize the entire model tree
# This single call initializes all parameters in the entire model hierarchy.
#initialized_model = nnx.init(lambda: DiffusionTransformer)(modelconfig)


#print("initial test done")



checkpoint_path = "/scratch/network/jl0796/DiT-ImageNet/results/experiment-v66/models/ckpt_0"
checkpoint_path = os.path.abspath(checkpoint_path)
#abstract_model = nnx.eval_shape(lambda: DiffusionTransformer(modelconfig))
abstract_model = nnx.eval_shape(lambda: model)


# abstract_state = jax.tree_util.tree_map(lambda x: nnx.Shape(x.shape, x.dtype) if hasattr(x, 'shape') else x, nnx.state(model))
abstract_state = jax.tree.map(lambda x: (x.shape, x.dtype), nnx.split(model))

# Initialize the checkpointer
checkpointer = ocp.StandardCheckpointer()

# Restore the checkpoint while passing the target tree for matching types/shapes
restored_ckpt = checkpointer.restore(
    checkpoint_path,
    # args=ocp.args.StandardRestore(abstract_state)
)
# Extract components
model_state_restored = restored_ckpt['model']
epoch_restored = restored_ckpt['epoch']
config_restored = restored_ckpt['config']
nnx.update(model, model_state_restored)
print("done")
config = modelConfig()
batch = 8
test_input = jnp.ones((batch, config.token_length, config.DiT_hidden_size))
test_condit = jnp.ones((batch, config.DiT_hidden_size))
test_timesteps = jnp.ones(batch)
test_input = jnp.ones((batch, 4, 32, 32))
#test_DiT = DiffusionTransformer(config)
x = model(test_input, test_timesteps)
print(x.shape)





"""
# Split into graphdef and abstract state
_, abstract_state = nnx.split(abstract_model)

# Initialize the checkpointer
checkpointer = ocp.StandardCheckpointer()

# Restore the checkpoint while passing the target tree for matching types/shapes
print(dir(checkpointer))
restored_ckpt = checkpointer.restore(
    checkpoint_path,
    # args=ocp.args.StandardRestore(abstract_state)
)



# Extract components
model_state_restored = restored_ckpt['model']
epoch_restored = restored_ckpt['epoch']
config_restored = restored_ckpt['config']


nnx.update(model, model_state_restored)
"""
