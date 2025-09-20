import os

import os
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import collections

from helpers.config import trainConfig
from vae.import_sd_vae_torch import get_sd_vae
from tqdm import tqdm
import numpy as np

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
torch.cuda.empty_cache()
print(f"✅ Using device: {DEVICE}")

# Data loading class (CustomImageDataset) inspired from https://www.kaggle.com/code/liucong12601/dinov2-imagenet-training
# Heavily modified the code to become load_data & return_dataloader functions, including the use of jax dataloader instead of torch
# The rest is my code


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
    x = Image.fromarray((img * 255).astype(np.uint8))
    if x.mode != 'RGB':
        x = x.convert('RGB')

    # scale so that the shorter side == resize_short
    short = min(x.size)
    scale = float(resize_short) / float(short)
    new_h = max(1, int(round(x.size[1] * scale)))
    new_w = max(1, int(round(x.size[0] * scale)))

    x_resized = torchvision.transforms.Resize((new_h, new_w)).forward(x)

    # center crop crop_size x crop_size
    top = max(0, (new_h - crop_size) // 2)
    left = max(0, (new_w - crop_size) // 2)
    x_cropped = x_resized.crop((left, top, left + crop_size, top + crop_size))
    
    return np.array(x_cropped)

class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
            x = np.asarray(sample).astype(np.float32) / 255.0
            x = resize_and_center_crop(x)
            x = (x - 0.5) / 0.5
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

    train_dataset = CustomImageDataset(train_samples)
    valid_dataset = CustomImageDataset(valid_samples)


    print(f"Total training images ({number_classes} classes): {len(train_dataset)}")
    print(f"Total validation images ({number_classes} classes): {len(valid_dataset)}")

    return train_dataset, valid_dataset

def return_dataloader(train_dataset: CustomImageDataset, 
                      valid_dataset: CustomImageDataset, 
                      config: trainConfig):
    """
    To use the dataloaders:
    batch = next(iter(dataloader))
    """

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    steps_per_epoch = len(train_loader)
    print(f"DataLoaders for created successfully.")
    print(f"{steps_per_epoch=}, val_steps: {len(valid_loader)}")

def save_latents(dataset: CustomImageDataset, vae, config: trainConfig, output_dir="./data/"):
    """
    Given a CustomImageDataset dataset, encodes images using the VAE
    saves the stored latents and classes. Allows every image to be 
    encoded in a single pass prior to training. 
    """
    os.makedirs(output_dir, exist_ok=True)   
    print(f"Saving files to: {os.path.abspath(output_dir)}")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    latents = np.zeros((len(dataset), 4, 32, 32), dtype=np.float16)
    labels = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_images, batch_labels = batch
        
        # Stack images into a single batch
        batch_x = torch.from_numpy(np.array(batch_images).astype(np.float32)).to(DEVICE)
        batch_x = torch.permute(batch_x, (0, 3, 1, 2))

        # Encode image
        latent_dist = vae.encode(batch_x).latent_dist
        latent = latent_dist.mean

        # Calculate start and end indices for this batch
        start_idx = i * config.batch_size
        end_idx = start_idx + latent.shape[0]

        latents[start_idx:end_idx] = latent.detach().cpu().numpy()
        labels.extend(batch_labels)

    # Save latents and labels as .npy files
    np.save(os.path.join(output_dir, "latents.npy"), latents)
    np.save(os.path.join(output_dir, "labels.npy"), np.array(labels))

def load_latents(input_dir):
    latents = np.load(os.path.join(input_dir, "latents.npy"))
    labels = np.load(os.path.join(input_dir, "labels.npy"))
    return latents, labels

if __name__=="__main__":
    train, val = load_data()
    vae, params = get_sd_vae()
    vae.to(DEVICE)
    config = trainConfig()
    save_latents(train, vae, config, output_dir="./data/train_latent")
    save_latents(val, vae, config, output_dir="./data/test_latent")

