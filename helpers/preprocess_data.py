import os
import jax.numpy as jnp
import jax_dataloader as jdl
from jax_dataloader import Dataset, DataLoader
from PIL import Image
from jax import image as jimage
import jax
from helpers.config import trainConfig
from vae.import_sd_vae import get_sd_vae
from tqdm import tqdm

jdl.manual_seed(1234)

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
    x = jnp.asarray(img)
    if x.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {x.shape}")
    H, W, C = x.shape

    # scale so that the shorter side == resize_short
    short = min(H, W)
    scale = float(resize_short) / float(short)
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))

    x_resized = jimage.resize(x, (new_h, new_w, C), method=method)

    # pad if needed so that we can center-crop crop_size x crop_size
    # should not be necessary with default settings.
    pad_h = max(0, crop_size - new_h)
    pad_w = max(0, crop_size - new_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        # set padding to 0
        x_resized = jnp.pad(
            x_resized,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        new_h = x_resized.shape[0]
        new_w = x_resized.shape[1]

    # center crop crop_size x crop_size
    top = max(0, (new_h - crop_size) // 2)
    left = max(0, (new_w - crop_size) // 2)
    x_cropped = x_resized[top:top + crop_size, left:left + crop_size, :]
    
    return x_cropped

def save_latents(dataset: CustomImageDataset, vae, params, output_dir="./data/"):
    """
    Given a CustomImageDataset dataset, encodes images using the VAE
    saves the stored latents and classes. Allows every image to be 
    encoded in a single pass prior to training. 
    """

    os.makedirs(output_dir, exist_ok=True)   
    print(f"Saving files to: {os.path.abspath(output_dir)}")
    latents = jnp.zeros((len(dataset), 32, 32, 4), dtype=jnp.float32)
    labels = []
    for idx, (path, target) in tqdm(enumerate(dataset.samples), total=len(dataset.samples)):
        img = Image.open(path).convert("RGB")

        x = jnp.asarray(img).astype(jnp.float16) / 255.0        # [0,1]
        x = resize_and_center_crop(x)
        # print(x.shape)
        x = (x - 0.5) / 0.5                                     # [-1,1]
        x = x[None, ...]                                        # [1, H, W, 3] 
        x = jnp.transpose(x, (0, 3, 1, 2))                      # [1, 3, H, W]

        # Encode image
        distr = vae.apply({"params": params}, x, method=vae.encode, deterministic=True)
        latent = distr.latent_dist.mean  
        # print(latent.shape)
        assert latent.shape == (1, 32, 32, 4)

        latents = latents.at[idx].set(latent[0])
        # print(target)
        labels.append(target)
    
        if idx%100 == 0:
            print(f"{idx} images processed out of {len(dataset.samples)}.")

    # Save latents and labels as .npy files
    jnp.save(os.path.join(output_dir, "latents.npy"), latents)
    jnp.save(os.path.join(output_dir, "labels.npy"), labels)



if __name__=="__main__":
    print(jax.devices())
    gpu0 = jax.devices("cuda")[0]
    with jax.default_device(gpu0):
        print(f"JAX is running on: {jax.default_backend()}")
        train, val = load_data()
        vae, params  = get_sd_vae()
        save_latents(train, vae, params, output_dir="./data/train_latent")
        save_latents(val, vae, params, output_dir="./data/test_latent")
