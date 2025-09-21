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
import json
import numpy as np
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        json_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(json_record)

def setup_logging(experiment_path):
    log_path = os.path.join(experiment_path, "training.log")
    json_log_path = os.path.join(experiment_path, "training.json.log")

    print(f"Logs will be saved to: {log_path} and {json_log_path}")

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler for traditional logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Create a file handler for JSON logging
    json_file_handler = logging.FileHandler(json_log_path)
    json_file_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)


def main(args):
    start_time = time.time()
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
    

    # Bundle states into checkpoint and save for later EMA.
    # model_state = nnx.state(deepcopy(DiTmodel))
    # ckpt = {'model': model_state, 'config': trainconfig.to_dict(), 'epoch': epoch} #, "args": vars(args)}
    # checkpointer = ocp.StandardCheckpointer()
    # checkpointer.save(os.path.abspath(os.path.join(models_dir, f'ckpt_{epoch}')), ckpt)
    # checkpointer.wait_until_finished()
    for epoch in range(10):
        state = nnx.state(deepcopy(DiTmodel))
        extra_params = {'config': trainconfig.to_dict(), 'epoch': epoch}
        mngr.save(
            epoch,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                extra_params=ocp.args.JsonSave(extra_params),
            ),
        )
        mngr.wait_until_finished()

    # Restoration
    train_state = jax.tree_util.tree_map(np.zeros_like, state)
    create_sharded_array = lambda x: jax.device_put(x, gpu_device)
    train_state = jax.tree_util.tree_map(create_sharded_array, train_state)
    abstract_train_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, train_state
    )

    
    path = os.path.abspath(models_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
    mngr = ocp.CheckpointManager(path, options=options)
    obj = ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                extra_params=ocp.args.JsonSave(extra_params),
            )
    restored = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_train_state),
            extra_params=ocp.args.JsonRestore(),
        )
    )
    
    state = nnx.update(DiTmodel, restored)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", "-m", default="DiT-S", help="The name of the model configurations")
    parser.add_argument("--data_directory", "-d", default="./data/", help="Directory to load ImageNet-100k from.")
    parser.add_argument("--results_dir", "-r", default="./results", help="Directory to save results to.")
    
    args = parser.parse_args()
    main(args)

