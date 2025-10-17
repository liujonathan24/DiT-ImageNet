from jax_dataloader import DataLoader
from helpers.config import trainConfig
from tqdm import tqdm
import os
from vae.import_sd_vae_torch import get_sd_vae
from PIL import Image
import numpy as np
import torch
import optax
from helpers.preprocess_data_torch import load_data

def main():
    # Load configurations
    trainconfig = trainConfig()
    trainconfig.batch_size = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sd_vae = get_sd_vae()
    sd_vae.eval()
    sd_vae.to(device)


    train_dataset, valid_dataset = load_data()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=trainconfig.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=trainconfig.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
   

    # Hyperparameters for sampling test
    N_samples_per_image = 4  # Number of samples to draw from each distribution
    M_images_to_test = 2     # Number of images from the batch to test
    test_restore = False # True

    # Create directory for latent distributions if it doesn't exist
    latent_dir = "data/latent_distribution"
    os.makedirs(latent_dir, exist_ok=True)

    for source in ["train", "valid"]:
        latent_info = []
        all_labels = []
        dataloader = train_dataloader if source == "train" else valid_loader
        for i, (batch_orig, labels) in enumerate(tqdm(dataloader)):

            # Save the corresponding label for each mean/std batch.
            all_labels.append(labels.cpu().numpy())

            batch = batch_orig.clone().permute(0, 3, 1, 2).to(device)

            # Encode using sd_vae:
            with torch.no_grad():
                latent_dist = sd_vae.encode(batch).latent_dist
                mean = 0.18215 * latent_dist.mean
                std = latent_dist.std  
                # Stack mean and std into a single tensor of shape (B, 2, C, H, W)
                info = torch.stack((mean, std), dim=1)
                
            latent_info.append(info.cpu().numpy())

            # IF testing restoration.
            if test_restore:
                with torch.no_grad():
                    # De-interleave info to recover mean and std
                    mean_restored = info[:, 0, :, :, :]
                    std_restored = info[:, 1, :, :, :]

                    # Select M images to test
                    mean_to_sample = mean_restored[:M_images_to_test]
                    std_to_sample = std_restored[:M_images_to_test]

                    # Get shapes for noise generation
                    M, C, H, W = mean_to_sample.shape

                    # Generate N samples for each of the M distributions
                    noise = torch.randn((M, N_samples_per_image, C, H, W), device=device)

                    # Use broadcasting to create N samples for each of the M images
                    samples = mean_to_sample.unsqueeze(1) + std_to_sample.unsqueeze(1) * noise

                    # Reshape to a single batch for VAE decoding
                    latents_to_decode = samples.view(M * N_samples_per_image, C, H, W)

                    # Decode the batch of sampled latents
                    restored_images = sd_vae.decode(latents_to_decode / 0.18215).sample.cpu().numpy()

                # Save the M * N images
                print(f"\nSaving {M_images_to_test * N_samples_per_image} sampled images...")
                for img_idx in range(M_images_to_test):
                    for sample_idx in range(N_samples_per_image):
                        image_index_in_batch = img_idx * N_samples_per_image + sample_idx
                        im_arr = restored_images[image_index_in_batch]
                        im_arr = np.transpose(im_arr, (1, 2, 0))
                        im_arr = np.clip(im_arr, -1.0, 1.0)
                        im_arr = (im_arr + 1) / 2.0
                        im_arr = (im_arr * 255).astype(np.uint8)
                        base_img_idx = i * trainconfig.batch_size + img_idx
                        path = os.path.abspath(f"tmp_restored_img_{base_img_idx}_sample_{sample_idx}.png")
                        Image.fromarray(im_arr).save(path)
                
                print(f"Saved images. Set test_restore=False to process the full dataset.")
                break  # Stop after the first batch when testing 

        # Concatenate and save the collected data
        output_latents = np.concatenate(latent_info, axis=0)
        output_labels = np.concatenate(all_labels, axis=0)
        print(f"Shape of entire latent distribution for '{source}': {output_latents.shape}")
        print(f"Shape of entire labels array for '{source}': {output_labels.shape}")
        np.save(os.path.join(latent_dir, f"latent_{source}_distr.npy"), output_latents)
        np.save(os.path.join(latent_dir, f"latent_{source}_labels.npy"), output_labels)

def create_latent_dataloader(source, batch_size, shuffle=True, num_workers=2):
    """
    Loads the saved latent distribution and labels and creates a JAX DataLoader.

    Args:
        source (str): "train" or "valid" to specify which dataset to load.
        batch_size (int): The batch size for the dataloader.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        A JAX DataLoader that yields batches of (labels, mean, std).
    """
    from jax_dataloader.datasets import ArrayDataset
    import jax.numpy as jnp

    latent_dir = "data/latent_distribution"
    distr_path = os.path.join(latent_dir, f"latent_{source}_distr.npy")
    labels_path = os.path.join(latent_dir, f"latent_{source}_labels.npy")

    # Load the saved numpy arrays
    latent_distr = jnp.array(np.load(distr_path))
    labels = jnp.array(np.load(labels_path))

    # Split the distribution data into mean and std
    mean_data = latent_distr[:, 0, ...]
    std_data = latent_distr[:, 1, ...]

    # Create a JAX ArrayDataset
    # The dataloader will yield batches in the order of the arrays provided.
    dataset = ArrayDataset(labels, mean_data, std_data)

    # Create and return the DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        backend='jax',
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

if __name__ == "__main__":
    main()
