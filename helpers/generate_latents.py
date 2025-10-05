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
    num_copies = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sd_vae = get_sd_vae()
    sd_vae.eval()
    sd_vae.to(device)


    train_dataset, valid_dataset = load_data(number_classes=1)
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
   

    test_restore = False
    output_latents = []
    for i, (batch_orig, labels) in enumerate(tqdm(train_dataloader)):
        for j in range(num_copies):
            #batch = torch.squeeze(batch)
            batch = batch_orig.clone().permute(0, 3, 1, 2).to(device)
            # print(f"batch shape initially, after permute: {batch.shape}")
            # assert batch.shape == (trainconfig.batch_size, 3, 256, 256)

            # Encode using sd_vae:
            with torch.no_grad():
                batch = 0.18215 * sd_vae.encode(batch).latent_dist.sample()
            # print(batch.shape)
            # assert batch.shape == (trainconfig.batch_size, 4, 32, 32)
            output_latents.append(batch.cpu().numpy())

            # IF testing restoration.
            if test_restore:
                with torch.no_grad():
                    restored = sd_vae.decode(batch/0.18215).sample.cpu().numpy()
                print(f"restored version shape: {restored.shape}")
                im_arr = np.transpose(restored, (0, 2, 3, 1))
                print(im_arr.shape)
                im_arr = im_arr[5,:,:,:]
                im_arr = np.squeeze(im_arr)
                print(im_arr.shape)
        
                # VAE output is ~[-1, 1], convert to [0, 255] for saving
                im_arr = np.clip(im_arr, -1.0, 1.0)
                im_arr = (im_arr + 1) / 2.0
                im_arr = (im_arr * 255).astype(np.uint8)
                path = os.path.abspath("tmp_restored.png")
                Image.fromarray(im_arr).save(path) 
                # assert 1 == 2 
        # break
    output_latents = np.concatenate(output_latents, axis=0)
    print(output_latents.shape)
    np.save(f"data/{num_copies}_train_latents.npy", output_latents)
if __name__ == "__main__":
    main()
