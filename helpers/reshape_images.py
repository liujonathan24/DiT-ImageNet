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


def resize_and_center_crop(
        x,
        resize_short=256,
        crop_size=256,
    ):
    """
    img: JAX or NumPy array with shape (H, W, C) and dtype float32/uint8
    returns: JAX array with shape (crop_size, crop_size, C) or (C, crop_size, crop_size)
    """
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
    
    return np.array(x_cropped.convert("RGB"))

def crop_data(number_classes=None):
    print("Crop data")
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
                        # Crop and rewrite file.
                        with open(path, 'rb') as f:
                            sample = Image.open(f)
                            x = resize_and_center_crop(sample) 
                        #path2 = os.path.join(class_dir, "1"+fname)
                        Image.fromarray(x).save(path)
                        #assert 1 == 2
                        item = (path, class_idx)
                        train_samples.append(item)
    print("Processing validation samples")
    # --- Manually build the list of validation samples ---
    valid_samples = []
    for class_name in selected_class_dirs:
        class_idx = class_to_idx[class_name]
        class_dir = os.path.join(VALID_PATH, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, fname)
                    # Crop and rewrite file.
                    with open(path, 'rb') as f:
                        sample = Image.open(f)
                        x = resize_and_center_crop(sample)
                    Image.fromarray(x).save(path)
                    item = (path, class_idx)
                    valid_samples.append(item)

if __name__=="__main__":
    crop_data()

