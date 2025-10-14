import torch
import jax
import jax.numpy as jnp
import numpy as np
import argparse
import flax.nnx as nnx

# JAX model and helpers
from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig, trainConfig
from scripts.convert_weights import convert_weights
from helpers.checkpoint import restore_checkpoint

# PyTorch model and helpers
# Note: This assumes you have the original DiT repo's `models.py file
from torch_dit.models import DiT_models# , unpatchify

def print_comparison(pt_tensor, jax_tensor, name):
    """Compares a PyTorch and JAX tensor and prints the MAE."""
    # Move JAX tensor to CPU and convert to numpy
    jax_np = np.array(jax_tensor)
    # Move PyTorch tensor to CPU and convert to numpy
    pt_np = pt_tensor.detach().cpu().numpy()

    mae = np.mean(np.abs(jax_np - pt_np))
    print(f"MAE for {name}: {mae:.6f}")
    if mae > 1e-4: # Set a tolerance for divergence
        print(f"!!! High divergence detected at {name} !!!")
    return pt_np # Return the PyTorch output as the new ground truth

def main(args):
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)


    # --- Load PyTorch Model ---
    print("--- Loading PyTorch Model ---")
    pt_model = DiT_models[args.model](
        input_size=32,
        num_classes=1000
    ).to(device)

    
    # Load state dict directly, bypassing find_model
    state_dict = torch.load(args.pt_ckpt, map_location="cpu")
    pt_model.load_state_dict(state_dict)
    pt_model.eval()
    # pt_model.to(dtype=torch.bfloat16)
    print("PyTorch Model Loaded.")

    # --- Load and Convert JAX Model ---
    print("\n--- Loading and Converting JAX Model ---")
    # jax_config = modelConfig(type='DiT-XL')
    # jax_model = DiffusionTransformer(jax_config)
    if jax.devices("gpu"):
        gpu_device = jax.devices("gpu")[0]
    else:
        gpu_device = None
    model_config = modelConfig(type='DiT-XL')

    # Restore the JAX model from the converted checkpoint
    jax_model_o = DiffusionTransformer(model_config)
    jax_model, _ = restore_checkpoint(args.checkpoint_path, model_config, trainConfig(), gpu_device)

    # updated_jax_state = convert_weights(args.pt_ckpt, jax_model)
    # nnx.update(jax_model, updated_jax_state)
    print("JAX Model Loaded and Weights Converted.")

    # --- Create Identical Inputs ---
    print("\n--- Creating Inputs ---")
    seed = 42
    # PyTorch inputs
    pt_rng = torch.manual_seed(seed)
    pt_z = torch.randn(1, 4, 32, 32, device=device, dtype=torch.float32)
    pt_t = torch.tensor([249], device=device)
    pt_y = torch.tensor([207], device=device)

    # JAX inputs (from the same numpy arrays)
    jax_rng = jax.random.PRNGKey(seed)
    jax_z = jnp.array(pt_z.cpu().to(torch.float32).numpy(), dtype=jnp.float32)
    jax_t = jnp.array(pt_t.cpu().numpy())
    jax_y = jnp.array(pt_y.cpu().numpy())
    print("Inputs created.")

    # --- Model Surgery and Comparison ---
    print("\n--- Starting Layer-by-Layer Comparison ---")

    # 1. Embeddings (Patch, Position, Time, Class)
    # PyTorch
    pt_x = pt_model.x_embedder(pt_z) + pt_model.pos_embed
    pt_c = pt_model.t_embedder(pt_t) + pt_model.y_embedder(pt_y, train=False)
    # JAX
    # Note: JAX model is not vmapped here, so we operate on a single batch item
    jax_x = jax_model.mapper.convert_to_stream(jax_z[0]) + jax_model.pos_embed
    jax_c = jax_model.time_MLP(jax_model.time_embed(jax_t[0])) + jax_model.y_embedder(jax_y[0])
    
    # Comparison
    #import ipdb; ipdb.set_trace()
    print_comparison(pt_model.pos_embed, jax_model.pos_embed, "pos embed")
    print_comparison(pt_model.x_embedder(pt_z), jax_model.mapper.convert_to_stream(jax_z[0]), "x embed")
    
    print_comparison(pt_model.y_embedder(pt_y, train=False), jax_model.y_embedder(jax_y[0]), "c condition")
    print_comparison(pt_model.t_embedder(pt_t), jax_model.time_MLP(jax_model.time_embed(jax_t[0])), "time embed")
    print()

    pt_x_np = print_comparison(pt_x.squeeze(0), jax_x, "Initial Embeddings (x)")
    pt_c_np = print_comparison(pt_c.squeeze(0), jax_c, "Initial Conditioning (c)")

    # 2. Transformer Blocks
    # Use PyTorch outputs as input for the next stage to prevent error cascade
    jax_x = jnp.array(pt_x_np)
    jax_c = jnp.array(pt_c_np)
    # The PyTorch model expects a batch dimension, so we add it back
    pt_x = torch.from_numpy(pt_x_np).unsqueeze(0).to(device)
    pt_c = torch.from_numpy(pt_c_np).unsqueeze(0).to(device)

    for i in range(len(pt_model.blocks)):
        print(f"\n--- Block {i} ---")
        # PyTorch
        pt_x = pt_model.blocks[i](pt_x, pt_c)
        # JAX
        jax_x = jax_model.layers[i](jax_x, jax_c)

        # Comparison
        pt_x_np = print_comparison(pt_x.squeeze(0), jax_x, f"Block {i} Output")
        # Update inputs for next iteration with the PyTorch ground truth
        # jax_x = jnp.array(pt_x_np)
        # pt_x = torch.from_numpy(pt_x_np).unsqueeze(0).to(device)

    # 3. Final Layer
    print("\n--- Final Layer ---")
    # PyTorch
    pt_x = pt_model.final_layer(pt_x, pt_c)
    # JAX
    jax_x = jax_model.final_layer(jax_x, jax_c)
    pt_x_np = print_comparison(pt_x.squeeze(0), jax_x, "Final Layer Output")

    # 4. Unpatchify
    print("\n--- Unpatchify ---")
    # PyTorch 
    pt_final = pt_model.unpatchify(pt_x)#, pt_z.shape[2])
    # JAX
    jax_final = jax_model.mapper.convert_to_patches(jax_x)
    print_comparison(pt_final.squeeze(0), jax_final, "Final Image Output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--pt_ckpt", type=str, help="Path to the PyTorch DiT checkpoint (.pt file).", default="/scratch/network/jl0796/DiT-ImageNet/pretrained_models/DiT-XL-2-256x256_pretrained.pt")
    parser.add_argument("--checkpoint_path", type=str, default="pretrained/")
    args = parser.parse_args()
    main(args)
