import os
import jax.numpy as jnp
import jax_dataloader as jdl
from jax_dataloader import Dataset, DataLoader
from PIL import Image
from jax import image as jimage
import jax
import numpy as np
from helpers.config import trainConfig
from vae.import_sd_vae import get_sd_vae
from tqdm import tqdm

jdl.manual_seed(1234)

# Data loading class (CustomImageDataset) inspired from https://www.kaggle.com/code/liucong12601/dinov2-imagenet-training
# Heavily modified the code to become load_data & return_dataloader functions, including the use of jax dataloader instead of torch
# The rest is my code

def resize_and_center_crop(
    img, # img is a NumPy array (H, W, C) with float32/uint8
    resize_short=256,
    crop_size=256,
    method="linear"
):
    """
    img: JAX or NumPy array with shape (H, W, C) and dtype float32/uint8
    returns: JAX array with shape (crop_size, crop_size, C) or (C, crop_size, crop_size)
    """
    # Convert to PIL for resizing/cropping, then back to JAX array
    x = Image.fromarray((img * 255).astype(np.uint8)) # Assuming img is [0,1] float or [0,255] uint8
    if x.mode != 'RGB':
        x = x.convert('RGB')

    # scale so that the shorter side == resize_short
    short = min(x.size)
    scale = float(resize_short) / float(short)
    new_h = max(1, int(round(x.size[1] * scale)))
    new_w = max(1, int(round(x.size[0] * scale)))

    x_resized = x.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # center crop crop_size x crop_size
    top = max(0, (new_h - crop_size) // 2)
    left = max(0, (new_w - crop_size) // 2)
    x_cropped = x_resized.crop((left, top, left + crop_size, top + crop_size))
    
    return jnp.asarray(x_cropped) # Return JAX array

class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
            x = jnp.asarray(sample).astype(jnp.float32) / 255.0 # JAX array [0, 1]
            x = resize_and_center_crop(x) # JAX array, cropped and resized
            x = (x - 0.5) / 0.5 # JAX array [-1, 1]
        return x, target

def load_data(number_classes=None):
    print("Loading data")
    BASE_PATH = "./data/"

    TRAIN_PATHS = [
        os.path.join(BASE_PATH, 'train.X1'),
        os.path.join(BASE_PATH, 'train.X2'),
        os.path.join(BASE_PATH, 'train.X3'),
        os.path.join(BASE_PATH, 'train.X4'),
    ]
    VALID_PATH = os.path.join(BASE_PATH, 'val.X')
    LABEL_PATH = os.path.join(BASE_PATH, 'Labels.json')

    all_class_dirs = [
        d for train_path in TRAIN_PATHS
        for d in os.listdir(train_path)
        if os.path.isdir(os.path.join(train_path, d))
    ]

    if number_classes:
        selected_class_dirs = all_class_dirs[:number_classes]
    else:
        number_classes = len(all_class_dirs)
        selected_class_dirs = all_class_dirs

    class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_class_dirs)}

    print(f"Efficiently loading the following {len(selected_class_dirs)} classes: {selected_class_dirs}")
    
    # --- Manually build the list of training samples (images, labels) ---
    train_samples = []
    for train_path_part in TRAIN_PATHS:
        for class_name in selected_class_dirs:
            class_idx = class_to_idx[class_name]
            class_dir = os.path.join(train_path_part, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, fname)
                        item = (path, class_idx)
                        train_samples.append(item)

    # --- Manually build the list of validation samples ---
    valid_samples = []
    for class_name in selected_class_dirs:
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(VALID_PATH, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, fname)
                    item = (path, class_idx)
                    valid_samples.append(item)

    train_dataset = CustomImageDataset(np.array(train_samples))
    valid_dataset = CustomImageDataset(np.array(valid_samples))


    print(f"Total training images ({number_classes} classes): {len(train_dataset)}")
    print(f"Total validation images ({number_classes} classes): {len(valid_dataset)}")

    return train_dataset, valid_dataset

def return_dataloader(train_dataset: CustomImageDataset, 
                      valid_dataset: CustomImageDataset, 
                      config: trainConfig):
    """
    To use the jax dataloaders:
    batch = next(iter(dataloader))
    """

    train_loader = DataLoader(
        dataset=train_dataset,
        backend='jax',
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        backend='jax',
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    steps_per_epoch = len(train_loader)
    print(f"DataLoaders for created successfully.")
    print(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")

def save_latents(dataset: CustomImageDataset, vae, params, config: trainConfig, output_dir="./data/", num_samples_per_image: int = 1):
    """
    Given a CustomImageDataset dataset, encodes images using the VAE
    saves the stored latents and classes. Allows every image to be 
    encoded in a single pass prior to training. 
    num_samples_per_image: Number of latent samples to draw per input image.
    """
    os.makedirs(output_dir, exist_ok=True)   
    print(f"Saving files to: {os.path.abspath(output_dir)}")

    dataloader = DataLoader(
        dataset=dataset,
        backend='jax',
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    total_latents_count = len(dataset) * num_samples_per_image
    latents = jnp.zeros((total_latents_count, 4, 32, 32), dtype=jnp.float16)
    labels = []

    current_latent_idx = 0
    rng = jax.random.PRNGKey(0) # For VAE sampling

    for i, (batch_images, batch_labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        rng, vae_rng = jax.random.split(rng)
        
        # batch_images are already preprocessed JAX arrays from CustomImageDataset
        batch_x = jnp.transpose(batch_images, (0, 3, 1, 2)) # (B, H, W, C) -> (B, C, H, W)

        # Encode image
        distr = vae.apply({"params": params}, batch_x, method=vae.encode, deterministic=True)
        
        # Add print statements here to inspect latent distribution stats
        print(f"\nLatent distribution stats (mean, std, var) for batch {i}:")
        print(f"  Mean of means: {distr.latent_dist.mean.mean().item():.4f}")
        print(f"  Mean of stds: {distr.latent_dist.std.mean().item():.4f}")
        print(f"  Mean of variances: {distr.latent_dist.variance().mean().item():.4f}")
        print(f"  Min of means: {distr.latent_dist.mean.min().item():.4f}")
        print(f"  Max of means: {distr.latent_dist.mean.max().item():.4f}")

        # Sample num_samples_per_image times
        # latent_samples will have shape (num_samples_per_image, batch_size, 4, 32, 32)
        latent_samples = distr.latent_dist.sample(seed=vae_rng, sample_shape=(num_samples_per_image,)) 
        
        # Reshape to (batch_size * num_samples_per_image, 4, 32, 32)
        latent_samples = latent_samples.transpose((1, 0, 2, 3, 4)).reshape(-1, 4, 32, 32) # JAX transpose
        
        num_current_latents = latent_samples.shape[0]
        
        latents = latents.at[current_latent_idx : current_latent_idx + num_current_latents].set(latent_samples)
        current_latent_idx += num_current_latents

        # Extend labels by repeating each label num_samples_per_image times
        for label in batch_labels:
            labels.extend([label] * num_samples_per_image)

    # Save latents and labels as .npy files
    jnp.save(os.path.join(output_dir, "latents.npy"), latents)
    jnp.save(os.path.join(output_dir, "labels.npy"), jnp.array(labels))
    print(f"Saved {len(latents)} latents to {output_dir}")


if __name__=="__main__":
    print(jax.devices())
    # Ensure JAX uses GPU if available
    if jax.devices("gpu"):
        jax.default_device = jax.devices("gpu")[0]
        print(f"JAX is running on: {jax.default_backend()} (GPU)")
    else:
        print(f"JAX is running on: {jax.default_backend()} (CPU)")

    train, val = load_data()
    vae, params  = get_sd_vae()
    config = trainConfig()
    
    print("Saving training set with 5 samples per image...")
    save_latents(train, vae, params, config, output_dir="./data/train_latent_5_samples_per_image", num_samples_per_image=5)
    
    print("Saving validation set with 5 samples per image...")
    save_latents(val, vae, params, config, output_dir="./data/test_latent_5_samples_per_image", num_samples_per_image=5)
