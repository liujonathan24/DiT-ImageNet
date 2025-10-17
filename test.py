from diffusion_transformer import DiffusionTransformer
import jax 
import jax.numpy as jnp
from helpers.config import modelConfig, trainConfig
import argparse
import os
from helpers.diffusion import Diffusion
from vae.import_sd_vae_torch import get_sd_vae
from PIL import Image
import numpy as np
import torch
from helpers.checkpoint import restore_checkpoint

def main(args):
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        gpu_device = jax.devices("cpu")[0]
    assert gpu_device != None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sd_vae = get_sd_vae()
    sd_vae.eval()
    sd_vae.to(device)

    

    modelconfig = modelConfig(type='DiT-XL')
    trainconfig = trainConfig()
    model = DiffusionTransformer(modelconfig)

    model, extra_params = restore_checkpoint(args.checkpoint_path, modelconfig, trainconfig, gpu_device)

    print("Model restored from checkpoint")
    print(f"Restored epoch: {extra_params['epoch']}")

    config = modelConfig()
    trainconfig = trainConfig()
    trainconfig.batch_size = 1 # 12
    diffusion = Diffusion(trainconfig.linear_variance_min, trainconfig.linear_variance_max, trainconfig.tmax)

    os.makedirs(args.output_dir, exist_ok=True)
    rngs = jax.random.PRNGKey(42)
    # Sample 1000 images for FID.
    for i in range(int(jnp.ceil(1000/trainconfig.batch_size))):
        # Diffusion process. Starts with [b, c, h, w] = [b, 4, 32, 32] ~ N(0, 1)
        x_t = jax.random.normal(rngs, shape=(trainconfig.batch_size, config.image_channels, config.input_size, config.input_size))
       
        for t in range(1000, 0, -1): # range(1000, 0, -1):
            
            # t = jnp.ones((trainconfig.batch_size)) * t
            # print(x_t.shape, t.shape)
            t_vec = jnp.ones((trainconfig.batch_size)) * t
            prediction = model(x_t, t_vec, t_vec*0, train=False)
            modified_x_t = x_t - prediction * (1-diffusion.alphas[t-1])/jnp.sqrt(1-diffusion.alpha_bars[t-1])

            modified_x_t *= 1/jnp.sqrt(diffusion.alphas[t-1])
            if t >1:
                z_t = jax.random.normal(rngs, shape=(trainconfig.batch_size, config.image_channels, config.input_size, config.input_size))
            else:
                z_t = jnp.zeros((trainconfig.batch_size, config.image_channels, config.input_size, config.input_size))
            
            noise_t = jnp.sqrt(diffusion.variances[t-1]) * z_t

            x_t = modified_x_t + noise_t
            # x_t = jnp.clip(x_t, min=-1, max=1)

            if t%200 == 0:
                restored = sd_vae.decode(torch.tensor(np.array(x_t)).to(device)/0.18215).sample.detach().cpu().numpy()
                im_arr = np.transpose(restored, (0, 2, 3, 1))
                print(im_arr.shape)
                im_arr = im_arr[0,:,:,:]
                im_arr = np.squeeze(im_arr)
                print(im_arr.shape)
        
                # VAE output is ~[-1, 1], convert to [0, 255] for saving
                im_arr = np.clip(im_arr, -1.0, 1.0)
                im_arr = (im_arr + 1) / 2.0
                im_arr = (im_arr * 255).astype(np.uint8)
                path = os.path.abspath(os.path.join(args.output_dir, "tmp_restored.png"))
                Image.fromarray(im_arr).save(path) #os.path.join(experiment_path, "tmp_restored.png"))
        # Decode:
        x_t /= 0.18215
        print(f"Final shape is {x_t.shape}") # (12, 4, 32, 32)
        print(f"Latent stats before decoding (mean, min, max, var):")
        print(jnp.mean(x_t), jnp.min(x_t), jnp.max(x_t), jnp.var(x_t))
        
        with torch.no_grad():
            img = sd_vae.decode(torch.tensor(np.array(x_t)).to(device)).sample
        img = img.cpu().numpy()

        print(f"Decoded image shape: {img.shape}")
        print(f"Image stats after decoding (mean, min, max, var):")
        print(jnp.mean(img), jnp.min(img), jnp.max(img), jnp.var(img))

        for j in range(img.shape[0]):
            # take one image (3, H, W)
            im_arr = img[j, :, :, :]

            # rearrange to (H, W, C)
            im_arr = np.transpose(im_arr, (1, 2, 0))
            
            # VAE output is ~[-1, 1], convert to [0, 255] for saving
            im_arr = np.clip(im_arr, -1.0, 1.0)
            im_arr = (im_arr + 1) / 2.0
            im_arr = (im_arr * 255).astype(np.uint8)

            # save
            im = Image.fromarray(im_arr)
            im.save(os.path.join(args.output_dir, f"sample_{i * trainconfig.batch_size + j}.jpeg"))

            
        break






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", "-c", help="Path to the checkpoint directory to restore from.")
    parser.add_argument("--output_dir", "-o", help="Path to the directory to save images to.")
    args = parser.parse_args()
    main(args)
