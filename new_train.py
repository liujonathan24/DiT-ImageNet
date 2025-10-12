from jax_dataloader import DataLoader
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
from helpers.diffusion import Diffusion
from vae.import_sd_vae_torch import get_sd_vae
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from helpers.checkpoint import save_checkpoint, restore_checkpoint
from helpers.logging_utils import setup_logging
import logging
from jax_dataloader.datasets import ArrayDataset

@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, x, t, epsilon):
  def loss_fn(model):
    y_pred = model(x, t)  
    return optax.losses.squared_error(y_pred, epsilon).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)  # in-place updates

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
    trainconfig.batch_size = 256 #TODO: remove
    trainconfig.log_frequency = 51  # 635
    trainconfig.ckpt_frequency = 100 # 10
    trainconfig.epochs = 135000 # 1500

    diffusion = Diffusion(trainconfig.linear_variance_min, trainconfig.linear_variance_max, trainconfig.tmax)

    # Load DiT Model
    if args.resume:
        experiment_path = args.resume
        models_dir = os.path.abspath(os.path.join(experiment_path, "models"))

        DiTmodel, extra_params = restore_checkpoint(models_dir, modelconfig, trainconfig, gpu_device)
        start_epoch = extra_params['epoch'] + 1
        logging.info("Model restored from checkpoint.")
        logging.info(f"Restored epoch: {extra_params['epoch']}")
    else:
        os.makedirs('results', exist_ok=True)
        experiment_number = len(glob(f"results/*"))
        experiment_path = os.path.join('results', f"experiment-v{experiment_number}")
        models_dir = os.path.abspath(os.path.join(experiment_path, "models"))
        os.makedirs(models_dir)

        DiTmodel = DiffusionTransformer(modelconfig)
        start_epoch = 0
    
    # Set up logging and log initial experiment information.
    setup_logging(experiment_path)
    logging.info(f"Starting experiment at {experiment_path}")
    logging.info(f"Arguments: {args}")
    
    # Manage helper models, optimizers, checkpointers.
    options = ocp.CheckpointManagerOptions(max_to_keep=5)
    mngr = ocp.CheckpointManager(models_dir, options=options)
    opt = optax.adamw(learning_rate=trainconfig.learning_rate)
    optimizer = nnx.Optimizer(DiTmodel, opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sd_vae = get_sd_vae().to(device)
    
    train_dataset = jnp.array(np.load('data/shark_10_train_latents.npy')) # 'data/5_train_latents.npy'))
    print(train_dataset.shape)
    train_dataset = ArrayDataset(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        backend='jax',
        batch_size=trainconfig.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    ) 

    random_key = jax.random.PRNGKey(0)
    test_restore = False
    for epoch in range(start_epoch, trainconfig.epochs):
        logging.info(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        running_loss = 0.0
        for i, (batch) in enumerate(tqdm(train_dataloader)):
            batch = batch[0]  
            # print(batch.shape)
           
            # Encode using sd_vae:
            # batch = 0.18215 * sd_vae.encode(batch).latent_dist.sample()
            # assert batch.shape == (trainconfig.batch_size, 4, 32, 32)

            # IF testing restoration.
            if test_restore:
                restored = sd_vae.decode(batch/0.18215).sample.detach().cpu().numpy()
                im_arr = np.transpose(restored, (0, 2, 3, 1))
                print(im_arr.shape)
                im_arr = im_arr[0,:,:,:]
                im_arr = np.squeeze(im_arr)
                print(im_arr.shape)
        
                # VAE output is ~[-1, 1], convert to [0, 255] for saving
                im_arr = np.clip(im_arr, -1.0, 1.0)
                im_arr = (im_arr + 1) / 2.0
                im_arr = (im_arr * 255).astype(np.uint8)
                path = os.path.abspath(os.path.join(experiment_path, "tmp_restored.png"))
                Image.fromarray(im_arr).save(path) #os.path.join(experiment_path, "tmp_restored.png"))
                assert 1 == 2 

            batch = jnp.array(batch)
            # Sample noise & predict:
            iter_key, random_key = jax.random.split(random_key)
            t_key, noise_key = jax.random.split(iter_key)

            t = jax.random.randint(t_key, (batch.shape[0],), 0, 1000)
            alpha_bar_t = diffusion.get_alpha_bar(t)[:, None, None, None]
            # print(alpha_bar_t)
            epsilon = jax.random.normal(key=noise_key, shape=batch.shape)
            noisy_batch = batch * jnp.sqrt(alpha_bar_t) + jnp.sqrt(1 - alpha_bar_t) * epsilon

            loss = train_step(DiTmodel, optimizer, noisy_batch, t, epsilon)

            running_loss += loss.item()

            if (i + 1) % trainconfig.log_frequency == 0:
                avg_loss = running_loss / trainconfig.log_frequency
                logging.info(f"Epoch {epoch} | Step {i+1} | Loss: {avg_loss:.4f}")
                running_loss = 0.0
                
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} finished in {epoch_time:.2f} seconds")
        if (epoch + 1) % trainconfig.ckpt_frequency == 0:
            save_checkpoint(mngr, DiTmodel, optimizer, epoch, trainconfig, args)
        logging.info(f"Total time passed is: {time.time() - start_time} seconds")
    logging.info(f"Training completed in {time.time() - start_time} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-re", help="Path to the experiment directory to resume training from.")
    args = parser.parse_args()
    main(args)
