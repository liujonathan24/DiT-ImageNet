from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from flax import nnx
from helpers.config import modelConfig, trainConfig
import optax
import argparse
import os
from glob import glob
from copy import deepcopy
import optax
import orbax.checkpoint as ocp
import numpy as np
from helpers.porting import restore_checkpoint, save_checkpoint


def main(args):
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        gpu_device = None
    assert gpu_device != None
    assert args.model_config == "DiT-S", "Currently, the only model config implemented is DiT-S."

    os.makedirs(args.results_dir, exist_ok=True)
    experiment_number = len(glob(f"{args.results_dir}/*"))
    experiment_path = os.path.join(args.results_dir, f"experiment-v{experiment_number}")
    models_dir = os.path.join(experiment_path, "models")
    os.makedirs(experiment_path)

    path = os.path.abspath(models_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
    mngr = ocp.CheckpointManager(
        path, options=options, item_names=('state', 'extra_params')
    )

    trainconfig = trainConfig()
    modelconfig = modelConfig()
    DiTmodel = DiffusionTransformer(modelconfig)
    opt = optax.adamw(learning_rate=1e-3)
    optimizer = nnx.Optimizer(DiTmodel, opt, wrt=nnx.Param)

    # Test forward pass of DiTmodel:
    config = modelConfig()
    batch = 8
    test_input = jnp.ones((batch, config.token_length, config.DiT_hidden_size))
    test_condit = jnp.ones((batch, config.DiT_hidden_size))
    test_timesteps = jnp.ones(batch)
    test_input = jnp.ones((batch, 4, 32, 32))
    #test_DiT = DiffusionTransformer(config)
    x = DiTmodel(test_input, test_timesteps)
    print(f"Passed initial test. Received {x.shape} shape output")
    
    for epoch in range(10):
        save_checkpoint(mngr, DiTmodel, optimizer, epoch, trainconfig, args)

    print(f"Files in models folder after 10 epochs/10 saves.")
    print(os.listdir(models_dir))

    # Restoration
    DiTmodel, extra_params = restore_checkpoint(path, modelconfig, trainconfig, gpu_device)

    # Test forward pass of DiTmodel:
    config = modelConfig()
    batch = 8
    test_input = jnp.ones((batch, config.token_length, config.DiT_hidden_size))
    test_condit = jnp.ones((batch, config.DiT_hidden_size))
    test_timesteps = jnp.ones(batch)
    test_input = jnp.ones((batch, 4, 32, 32))
    #test_DiT = DiffusionTransformer(config)
    x = DiTmodel(test_input, test_timesteps)
    print(f"Passed final test. Received {x.shape} shape output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    parser.add_argument("--results_dir", "-r", default="./results", help="Directory to save results to.")
    
    args = parser.parse_args()
    main(args)

