import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
from flax import nnx
import os

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

def restore_checkpoint(mngr, model):
    abstract_state = jax.tree_util.tree_map(lambda x: nnx.Shape(x.shape, x.dtype) if hasattr(x, 'shape') else x, nnx.state(model))
    restored = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            extra_params=ocp.args.JsonRestore(),
        )
    )
    nnx.update(model, restored['state'])
    return restored['extra_params']
