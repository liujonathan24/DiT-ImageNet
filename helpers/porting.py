import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
from flax import nnx
import os
from helpers.config import modelConfig, trainConfig
from diffusion_transformer import DiffusionTransformer
import numpy as np

# TODO: Make this into a class with init establishing mngr
def save_checkpoint(mngr, model, optimizer, epoch, trainconfig, args):
    state = nnx.state(model)
    extra_params = {'config': trainconfig.to_dict(), 'epoch': epoch, 'args': vars(args)}
    mngr.save(
        epoch,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            extra_params=ocp.args.JsonSave(extra_params),
        ),
    )
    mngr.wait_until_finished()

def restore_checkpoint(model_path, modelconfig: modelConfig, trainconfig: trainConfig, gpu_device):
    DiTmodel = DiffusionTransformer(modelconfig)
    status = nnx.state(DiTmodel)
    train_state = jax.tree_util.tree_map(np.zeros_like, status)
    create_sharded_array = lambda x: jax.device_put(x, gpu_device)
    train_state = jax.tree_util.tree_map(create_sharded_array, train_state)
    abstract_train_state = jax.tree_util.tree_map(
        ocp.utils.to_shape_dtype_struct, train_state
    )

    path = os.path.abspath(model_path)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
    mngr = ocp.CheckpointManager(path, options=options)
    extra_params = {'config': trainconfig.to_dict(), 'epoch': 0}
    restored = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_train_state),
            extra_params=ocp.args.JsonRestore(extra_params),
        )
    )
    
    nnx.update(DiTmodel, restored['state'])
    return DiTmodel, restored['extra_params']
