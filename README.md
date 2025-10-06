# DiT-ImageNet Project

Implement a DiT-S model in JAX, and train it on ImageNet.


# Current status:
To download the data and process it, first run /scripts/download_imagenet.sh. Then, to resize the images, run 
``` python -m helpers.reshape_images```.

helpers/ contains many helper functions and classes.
checkpoint.py provides the functionality to save and load flax NNX models.
config.py and diffusion.py contain classes defining the model size, training configs, and generation configs.
generate_latents.py downloads or uses a local copy of the stabilityai/sd-vae-ft-ema VAE to save latent representations of the Imagenet data.
logging_utils creates a custom logging class.

preprocess_data_torch.py, preprocess_data.py are two variants (torch versus jax) that generate latent representations of Imagenet data. I created both versions to benchmark how quickly each ran.

In /vae, there are two files to import/download the VAE model in Jax (load as Pytorch but convert to Jax & save) or Pytorch. test_sd_vae.py allows these models to be tested. 







## Guide 

### Compute Resources
We will primarily use the Adroit cluster for this project.
1. Request an account: complete the [Adroit registration form](https://forms.rc.princeton.edu/registration/?q=adroit).
2. Read the scheduler docs: review the [SLURM guide](https://researchcomputing.princeton.edu/support/knowledge-base/slurm#gpus) to understand GPU job submission.
3. We recommend using MIG A100 or V100 on Adroit for development and debugging.

### JAX
You should use [JAX](https://github.com/jax-ml/jax). This community [learning guide](https://github.com/rcrowe-google/Learning-JAX) may be helpful.

### Dataset
Use the [ImageNet-100 dataset](https://www.kaggle.com/datasets/ambityga/imagenet100) (a subset of ImageNet-1k with 100 randomly selected classes).

### DiT
* You should understand the background knowledge, including [Transformers](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com) and [Diffusion models](https://arxiv.org/abs/2006.11239).
* You should understand DiT models [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).
* You should understand FID, the evaluation metric.

## Goal
* Tune training hyperparameters and other training settings for DiT-S to obtain an FID score lower than 20.
* More advanced: tweak the architectures to optimize FID, but stay within 33M parameters (the size of DiT-S).

## Instructions
* Invite the GitHub account @r01566525 to your working GitHub repo.
* Work independently.
* Feel free to refer to other resources or tutorials.
* Implement the codebase yourself as much as possible, including the attention module and the training loop; you may use AI or refer to others' code in an assisting capacity only, and you should be able to explain all the code.


## Weekly Reports
* Submit a weekly report (within 3 pages each week, keep in the same Google Doc) at the end of each week.
* Feel free to structure it yourself: you can include progress, issues / solutions, results, plots, or any other relevant things. Feel free to report negative results too, i.e., what has been tried but didn't work.
* Please clearly indicate in which parts of the code you used others' code (link source) or AI, and in what capacity.


## Contact
Please send your weekly reports and requests for help (if any) to r01566525@gmail.com. An initial meeting with a graduate student can be scheduled by emailing this address; further meetings can be arranged if necessary.
