import os
import jax.numpy as jnp
import jax_dataloader as jdl
from jax_dataloader import Dataset, DataLoader
from PIL import Image
from helpers.config import trainConfig
from vae.import_sd_vae import get_sd_vae

jdl.manual_seed(1234)

# Data loading inspired from https://www.kaggle.com/code/liucong12601/dinov2-imagenet-training
# Modified as functions and to use jax dataloader

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

def save_latents(dataset: CustomImageDataset, vae, params, output_dir="./data/"):
    """
    Given a CustomImageDataset dataset, encodes images using the VAE
    saves the stored latents and classes. Allows every image to be 
    encoded in a single pass prior to training. 
    """
    os.makedirs(output_dir, exist_ok=True)   

    latents = jnp.zeros((len(dataset), 32, 32, 4), dtype=jnp.float32)
    labels = []

    for idx, (path, target) in enumerate(dataset.samples):
        img = Image.open(path).convert("RGB")

        x = jnp.asarray(img).astype(jnp.float32) / 255.0        # [0,1]
        x = (x - 0.5) / 0.5                                     # [-1,1]
        x = x[None, ...]                                        # [1, H, W, 3] 
        x = jnp.transpose(x, (0, 3, 1, 2))                      # [1, 3, H, W]

        # Encode image
        distr = vae.apply({"params": params}, x, method=vae.encode, deterministic=True)
        latent = distr.latent_dist.mean  
        assert latent.shape == (1, 32, 32, 4)

        latents = latents.at[idx].set(latent[0])
        labels.append(int(target[2:])) # Previous two letters are n0.


    # Save latents and labels as .npy files
    jnp.save(os.path.join(output_dir, "latents.npy"), latents)
    jnp.save(os.path.join(output_dir, "labels.npy"), labels)

if __name__=="__main__":
    train, val = load_data()
