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

from helpers.porting import restore_checkpoint

def main(args):
    modelconfig = modelConfig()
    model = DiffusionTransformer(modelconfig)

    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
    mngr = ocp.CheckpointManager(args.checkpoint_path, options=options)

    extra_params = restore_checkpoint(mngr, model)

    print("Model restored from checkpoint")
    print(f"Restored epoch: {extra_params['epoch']}")

    config = modelConfig()
    batch = 8
    test_input = jnp.ones((batch, config.token_length, config.DiT_hidden_size))
    test_condit = jnp.ones((batch, config.DiT_hidden_size))
    test_timesteps = jnp.ones(batch)
    test_input = jnp.ones((batch, 4, 32, 32))
    x = model(test_input, test_timesteps)
    print(x.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", "-c", help="Path to the checkpoint directory to restore from.")
    
    args = parser.parse_args()
    main(args)
