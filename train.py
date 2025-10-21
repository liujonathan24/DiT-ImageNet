import jax 
import jax.numpy as jnp
from flax import nnx
import optax
from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig, trainConfig
import argparse
import os
import orbax.checkpoint as ocp
import time
from helpers.create_diffusion import create_diffusion
from vae.import_sd_vae_torch import get_sd_vae
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from helpers.checkpoint import save_checkpoint, restore_checkpoint
from helpers.logging_utils import setup_logging
import logging
from helpers.save_latent_distr import create_latent_dataloader

@nnx.jit
def train_step(model, ema_model, optimizer, x, t, labels, epsilon, *, rngs, ema_decay=0.9999):
    """A single training step, JIT-compiled for performance."""
    def loss_fn(model):
        y_pred = model(x, t, labels, train=True, rngs=rngs)
        return optax.losses.squared_error(y_pred, epsilon).mean()

    # Get loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    # Update the main model
    optimizer.update(model, grads)

    # Update the EMA model
    new_params = nnx.state(model, nnx.Param)
    ema_params = nnx.state(ema_model, nnx.Param)
    
    updated_ema_params = jax.tree_util.tree_map(
        lambda ema, new: ema * ema_decay + (1 - ema_decay) * new,
        ema_params,
        new_params
    )
    nnx.update(ema_model, updated_ema_params)

    return loss

def main(args):
    start_time = time.time()
    
    # Set device
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        gpu_device = None
    assert gpu_device != None
    
    # Load configurations
    trainconfig = trainConfig()
    modelconfig = modelConfig()
    trainconfig.batch_size = 768 
    # trainconfig.log_frequency = 51
    trainconfig.ckpt_frequency = 10
    trainconfig.epochs = 135000

    diffusion = create_diffusion(trainconfig.tmax)

    # Load DiT Model and create EMA model
    if args.resume:
        experiment_path = args.resume
        models_dir = os.path.abspath(os.path.join(experiment_path, "models"))
        ema_model_dir = os.path.abspath(os.path.join(experiment_path, "ema_model"))
        # Restore the EMA model from its dedicated directory
        ema_model, extra_params = restore_checkpoint(ema_model_dir, modelconfig, trainconfig, gpu_device)
        DiTmodel, extra_params = restore_checkpoint(models_dir, modelconfig, trainconfig, gpu_device)
        start_epoch = extra_params['epoch'] + 1
        logging.info("Model and EMA model restored from checkpoint.")
        logging.info(f"Restored epoch: {extra_params['epoch']}")
    else:
        os.makedirs('results', exist_ok=True)
        experiment_number = len(glob(f"results/*"))
        experiment_path = os.path.join('results', f"experiment-v{experiment_number}")
        models_dir = os.path.abspath(os.path.join(experiment_path, "models"))
        ema_model_dir = os.path.abspath(os.path.join(experiment_path, "ema_model"))
        os.makedirs(models_dir)
        os.makedirs(ema_model_dir)

        DiTmodel = DiffusionTransformer(modelconfig)
        ema_model = DiffusionTransformer(modelconfig)
        nnx.update(ema_model, nnx.state(DiTmodel)) # Ensure they start identical
        start_epoch = 0
    
    # Set up logging
    setup_logging(experiment_path)
    logging.info(f"Starting experiment at {experiment_path}")
    logging.info(f"Arguments: {args}")
    
    # Setup optimizer and checkpointers
    options = ocp.CheckpointManagerOptions(max_to_keep=5)
    model_mngr = ocp.CheckpointManager(models_dir, options=options)
    ema_mngr = ocp.CheckpointManager(ema_model_dir, options=options)
    opt = optax.adamw(learning_rate=trainconfig.learning_rate)

    def is_trainable(path, node):
        if not isinstance(node, nnx.Param):
            return False
        for key in path:
            if hasattr(key, 'key') and key.key == 'pos_embed':
                return False
        return True

    optimizer = nnx.Optimizer(DiTmodel, opt, wrt=is_trainable)
    
    # Create the dataloader from pre-computed latent distributions
    train_dataloader = create_latent_dataloader("train", trainconfig.batch_size)

    random_key = jax.random.PRNGKey(0)
    for epoch in range(start_epoch, trainconfig.epochs):
        logging.info(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        running_loss = 0.0
        for i, (batch_labels, mean, std) in enumerate(tqdm(train_dataloader)):
            
            # --- Main Training Logic ---
            iter_key, random_key = jax.random.split(random_key)
            
            # 1. Sample from the latent distribution
            sampling_noise_key, diffusion_noise_key, t_key, dropout_key = jax.random.split(iter_key, 4)
            epsilon_for_sampling = jax.random.normal(key=sampling_noise_key, shape=mean.shape)
            batch_latents = mean + std * epsilon_for_sampling

            # 2. Sample a random timestep and noise for the diffusion process
            t = jax.random.randint(t_key, (batch_latents.shape[0],), 0, trainconfig.tmax)
            alpha_bar_t = diffusion.get_alpha_bar(t)[:, None, None, None]
            epsilon_for_diffusion = jax.random.normal(key=diffusion_noise_key, shape=batch_latents.shape)
            
            # 3. Create the noisy input for the model
            noisy_batch = batch_latents * jnp.sqrt(alpha_bar_t) + jnp.sqrt(1 - alpha_bar_t) * epsilon_for_diffusion

            # 4. Run the training step
            loss = train_step(DiTmodel, ema_model, optimizer, noisy_batch, t, batch_labels, epsilon_for_diffusion, rngs={'dropout': dropout_key})

            running_loss += loss.item()

        # Log the average loss for the entire epoch
        epoch_avg_loss = running_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch} Average Loss: {epoch_avg_loss:.4f}")
                
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} finished in {epoch_time:.2f} seconds")
        if (epoch + 1) % trainconfig.ckpt_frequency == 0:
            save_checkpoint(model_mngr, DiTmodel, optimizer, epoch, trainconfig, args)
            save_checkpoint(ema_mngr, ema_model, optimizer, epoch, trainconfig, args)
        logging.info(f"Total time passed is: {time.time() - start_time} seconds")
    logging.info(f"Training completed in {time.time() - start_time} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-re", help="Path to the experiment directory to resume training from.")
    args = parser.parse_args()
    main(args)
