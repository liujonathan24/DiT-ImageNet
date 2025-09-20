import os

import os
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import collections

from helpers.config import trainConfig
from vae.import_sd_vae import get_sd_vae
from tqdm import tqdm
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# Data loading class (CustomImageDataset) inspired from https://www.kaggle.com/code/liucong12601/dinov2-imagenet-training
# Heavily modified the code to become load_data & return_dataloader functions, including the use of jax dataloader instead of torch
# The rest is my code

class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, 'rb') as f:
            sample = np.array(Image.open(f).convert('RGB'))
        if self.transform:
            sample = self.transform(sample)
        return sample, target

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

    train_dataset = CustomImageDataset(train_samples)
    valid_dataset = CustomImageDataset(valid_samples)


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
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    steps_per_epoch = len(train_loader)
    print(f"DataLoaders for created successfully.")
    print(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")

def resize_and_center_crop(
    img,
    resize_short=256,
    crop_size=256,
    method="linear"
):
    """
    img: JAX or NumPy array with shape (H, W, C) and dtype float32/uint8
    returns: JAX array with shape (crop_size, crop_size, C) or (C, crop_size, crop_size)
    """
    x = np.asarray(img)
    if x.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {x.shape}")
    H, W, C = x.shape

    # scale so that the shorter side == resize_short
    short = min(H, W)
    scale = float(resize_short) / float(short)
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))

    x_resized = torchvision.transforms.Resize((new_h, new_w, C)).forward(x)

    # center crop crop_size x crop_size
    top = max(0, (new_h - crop_size) // 2)
    left = max(0, (new_w - crop_size) // 2)
    x_cropped = x_resized[top:top + crop_size, left:left + crop_size, :]
    
    return x_cropped

def save_latents(dataset: CustomImageDataset, vae, params, config: trainConfig, output_dir="./data/"):
    """
    Given a CustomImageDataset dataset, encodes images using the VAE
    saves the stored latents and classes. Allows every image to be 
    encoded in a single pass prior to training. 
    """
    os.makedirs(output_dir, exist_ok=True)   
    print(f"Saving files to: {os.path.abspath(output_dir)}")

    dataloader = DataLoader(
        dataset=dataset,
        # backend='jax',
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    latents = np.zeros((len(dataset), 32, 32, 4), dtype=np.float16)
    labels = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_images, batch_labels = batch
        # Ensure batch_images is a list of PIL images
        # if not isinstance(batch_images, list):
        #    batch_images = [Image.fromarray(img) for img in batch_images]

        # Preprocess images in the batch
        processed_images = []
        for img in batch_images:
            x = np.asarray(img).astype(np.float32) / 255.0
            x = resize_and_center_crop(x)
            x = (x - 0.5) / 0.5
            print(x.shape)
            assert x.shape == (1, 256, 256, 3)
            processed_images.append(x)
        
        # Stack images into a single batch
        batch_x = np.stack(processed_images).astype(np.float16)
        batch_x = np.transpose(batch_x, (0, 3, 1, 2))

        # Encode image
        distr = vae.apply({"params": params}, batch_x, method=vae.encode, deterministic=True)
        latent = distr.latent_dist.mean

        # Calculate start and end indices for this batch
        start_idx = i * config.batch_size
        end_idx = start_idx + latent.shape[0]

        latents = latents.at[start_idx:end_idx].set(latent)
        labels.extend(batch_labels)

    # Save latents and labels as .npy files
    np.save(os.path.join(output_dir, "latents.npy"), latents)
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))



if __name__=="__main__":
    print(torch.cuda.is_available())
    # gpu0 = jax.devices("cuda")[0]
    # with jax.default_device(gpu0):
        # print(f"JAX is running on: {jax.default_backend()}")
    train, val = load_data()
    vae, params  = get_sd_vae()
    config = trainConfig()
    save_latents(train, vae, params, config, output_dir="./data/train_latent")
    save_latents(val, vae, params, config, output_dir="./data/test_latent")

