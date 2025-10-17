from torch.utils.data import DataLoader
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

    for source in ["train", "valid"]:
        latent_info = []
        dataloader = train_dataloader if source == "train" else valid_loader
        for i, (batch_orig, labels) in enumerate(tqdm(dataloader)):
            #batch = torch.squeeze(batch)
            batch = batch_orig.clone().permute(0, 3, 1, 2).to(device)
            # print(f"batch shape initially, after permute: {batch.shape}")
            # assert batch.shape == (trainconfig.batch_size, 3, 256, 256)

            # Encode using sd_vae:
            with torch.no_grad():
                latent_dist = sd_vae.encode(batch).latent_dist
                mean = 0.18215 * latent_dist.mean
                std = latent_dist.std  
                # Interleave mean and std along the channel dimension
                B, C, H, W = mean.shape
                info = torch.stack((mean, std), dim=1) #.view(B, C * 2, H, W)
                # print(f"Shapes of distributions (mean, std, interleaved): {mean.shape, std.shape, info.shape}")
                
            latent_info.append(info.cpu().numpy())

            # IF testing restoration.
            if test_restore:
                with torch.no_grad():
                    # De-interleave info to recover mean and std
                    # info shape is (B, 2, C, H, W)
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
                        # Get the correct image from the decoded batch
                        image_index_in_batch = img_idx * N_samples_per_image + sample_idx
                        im_arr = restored_images[image_index_in_batch]
                        
                        # Transpose from (C, H, W) to (H, W, C) for saving
                        im_arr = np.transpose(im_arr, (1, 2, 0))
                
                        # VAE output is ~[-1, 1], convert to [0, 255]
                        im_arr = np.clip(im_arr, -1.0, 1.0)
                        im_arr = (im_arr + 1) / 2.0
                        im_arr = (im_arr * 255).astype(np.uint8)

                        # Create a unique path for each sampled image
                        base_img_idx = i * trainconfig.batch_size + img_idx
                        path = os.path.abspath(f"tmp_restored_img_{base_img_idx}_sample_{sample_idx}.png")
                        Image.fromarray(im_arr).save(path)
                
                print(f"Saved images. Set test_restore=False to process the full dataset.")
                break  # Stop after the first batch when testing 

        output_latents = np.concatenate(latent_info, axis=0)
        print(f"Shape of entire latent distribution: {output_latents.shape}")
        np.save(f"data/latent_distribution/latent_{source}_distr.npy", output_latents)

if __name__ == "__main__":
    main()
