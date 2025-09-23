import numpy as np
from PIL import Image
import jax.numpy as jnp
from helpers.config import trainConfig, modelConfig
from jax_dataloader import Dataset, DataLoader
# from vae.import_sd_vae_torch import get_sd_vae
from vae.import_sd_vae import get_sd_vae
from helpers.preprocess_data_torch import load_latents, load_data, return_dataloader
import jax_dataloader as jdl
import torch


# Load model & weights
vae, params = get_sd_vae()

test_image = True
test_restore_image = True
test_load_and_restore_image = True

if test_load_and_restore_image:
    #TODO: fix.
    train, valid = load_data()
    config = trainConfig()
    dataloader = return_dataloader(train, valid, config)
    

    first_latent, first_label = next(iter(dataloader))
        
    batch_x = torch.from_numpy(np.array(first_latent).astype(np.float32))
    batch_x = torch.permute(batch_x, (0, 3, 1, 2))

    latent_dist = vae.encode(batch_x).latent_dist
    
    # Add print statements here to inspect latent distribution stats
    print(f"\nLatent distribution stats (mean, std, var) for batch {i}:")
    print(f"  Mean of means: {latent_dist.mean.mean().item():.4f}")
    print(f"  Mean of stds: {latent_dist.std.mean().item():.4f}")
    print(f"  Min of means: {latent_dist.mean.min().item():.4f}")
    print(f"  Max of means: {latent_dist.mean.max().item():.4f}")
    
    # Sample num_samples_per_image times
    # latent_samples will have shape (num_samples_per_image, batch_size, 4, 32, 32)
    # latent_samples = latent_dist.sample(sample_shape=(num_samples_per_image,)) 
    samples = [latent_dist.sample() for _ in range(1)]
    latent_samples = torch.stack(samples)

    # Reshape to (batch_size * num_samples_per_image, 4, 32, 32)
    latent_samples = latent_samples.permute(1, 0, 2, 3, 4).reshape(-1, 4, 32, 32)
    
    # RESTORE.
    decoded_x = vae.apply(# .decode(first_image).sample.detach().numpy()
                    {"params": params},
                    latent_samples[0],
                    method=vae.decode).sample


    # Convert back to [1, H, W, 3] and [0, 256]
    decoded_x = jnp.transpose(decoded_x, (0, 2, 3, 1)) 
    img_out = np.clip((np.array(decoded_x[0]) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    print(f"Final decoded shape: {decoded_x.shape}")

    Image.fromarray(img_out).save("./vae/restored_imagenet.png")

    print(f"Saved {len(latents)} latents to {output_dir}")

if test_restore_image:
    train_latents, train_labels = load_latents("./data/train_latent_5_samples_per_image")

    train_latents = jdl.ArrayDataset(train_latents, train_labels)
    train_latents = jdl.DataLoader(
        train_latents, 
        backend='pytorch', 
        batch_size=1,
        shuffle=True,
        drop_last=False,
    )
    first_image, _ = next(iter(train_latents))
    print(f"First Image latent shape: {first_image.shape}")
    print(f"Latent stats before decoding (mean, min, max, var):")
    print(jnp.mean(first_image), jnp.min(first_image), jnp.max(first_image), jnp.var(first_image))
    
    # # 
    # first_image = torch.tensor(first_image.astype(np.float32))
    decoded_x = vae.apply(# .decode(first_image).sample.detach().numpy()
                    {"params": params},
                    first_image,
                    method=vae.decode).sample


    # Convert back to [1, H, W, 3] and [0, 256]
    decoded_x = jnp.transpose(decoded_x, (0, 2, 3, 1)) 
    img_out = np.clip((np.array(decoded_x[0]) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    print(f"Final decoded shape: {decoded_x.shape}")

    Image.fromarray(img_out).save("./vae/restored_imagenet.png")


if test_image:
    path = "./vae/original.png"
    img = Image.open(path).convert("RGB")

    # Convert to [1, 3, H, W] format with range [-1, 1] 
    x = np.asarray(img).astype(np.float32) / 255.0        # [0,1]
    x = (x - 0.5) / 0.5                                   # [-1,1]
    x = jnp.asarray(x)[None, ...]                         # [1, H, W, 3] 
    x = jnp.transpose(x, (0, 3, 1, 2))               # [1, 3, H, W]
    print(f"Original shape: {x.shape}")


    # Encode image
    distr = vae.apply(
                        {"params": params}, 
                        x, 
                        method=vae.encode, 
                        deterministic=True)
    patches = distr.latent_dist.mean  

    print(f"Patches shape: {patches.shape}")
    print(f"Latent stats before decoding (mean, min, max, var):")
    print(jnp.mean(patches), jnp.min(patches), jnp.max(patches), jnp.var(patches))

    # Decode
    decoded_x = vae.apply(
                        {"params": params},
                        patches,
                        method=vae.decode).sample


    # Convert back to [1, H, W, 3] and [0, 256]
    decoded_x = jnp.transpose(decoded_x, (0, 2, 3, 1)) 
    img_out = np.clip((np.array(decoded_x[0]) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    print(f"Final decoded shape: {decoded_x.shape}")

    Image.fromarray(img_out).save("./vae/reconstruction.png")

