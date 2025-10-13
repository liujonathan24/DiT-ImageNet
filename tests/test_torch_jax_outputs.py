import torch
import jax
import jax.numpy as jnp
import numpy as np
import argparse

# JAX model and helpers
from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig
from scripts.convert_weights import convert_weights

# PyTorch model and helpers
# Note: This assumes you have the original DiT repo's `models.py` file
# and a `download.py` file to load the checkpoint.
from torch.models import DiT_models
from torch.download import find_model

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
    state_dict = find_model(f"DiT-XL-2-256x256.pt")
    pt_model.load_state_dict(state_dict)
    pt_model.eval()
    pt_model.to(dtype=torch.bfloat16)
    print("PyTorch Model Loaded.")

    # --- Load and Convert JAX Model ---
    print("\n--- Loading and Converting JAX Model ---")
    jax_config = modelConfig(type='DiT-XL')
    jax_model = DiffusionTransformer(jax_config)
    updated_jax_state = convert_weights(f"DiT-XL-2-256x256.pt", jax_model)
    nnx.update(jax_model, updated_jax_state)
    print("JAX Model Loaded and Weights Converted.")

    # --- Create Identical Inputs ---
    print("\n--- Creating Inputs ---")
    seed = 42
    # PyTorch inputs
    pt_rng = torch.manual_seed(seed)
    pt_z = torch.randn(1, 4, 32, 32, device=device, dtype=torch.bfloat16)
    pt_t = torch.tensor([249], device=device)
    pt_y = torch.tensor([207], device=device)

    # JAX inputs (from the same numpy arrays)
    jax_rng = jax.random.PRNGKey(seed)
    jax_z = jnp.array(pt_z.cpu().numpy(), dtype=jnp.bfloat16)
    jax_t = jnp.array(pt_t.cpu().numpy())
    jax_y = jnp.array(pt_y.cpu().numpy())
    print("Inputs created.")

    # --- Model Surgery and Comparison ---
    print("\n--- Starting Layer-by-Layer Comparison ---")

    # 1. Embeddings (Patch, Position, Time, Class)
    # PyTorch
    pt_x = pt_model.x_embedder(pt_z) + pt_model.pos_embed
    pt_c = pt_model.t_embedder(pt_model.timestep_embedding(pt_t)) + pt_model.y_embedder(pt_y)
    # JAX
    jax_x = jax_model.mapper.convert_to_stream(jax_z[0]) + jax_model.pos_embed
    jax_c = jax_model.time_MLP(jax_model.time_embed(jax_t[0])) + jax_model.y_embedder(jax_y[0])
    
    # Comparison
    # Note: pt_x has a batch dim, jax_x does not. We squeeze pt_x.
    pt_x_np = print_comparison(pt_x.squeeze(0), jax_x, "Initial Embeddings (x)")
    pt_c_np = print_comparison(pt_c.squeeze(0), jax_c, "Initial Conditioning (c)")

    # 2. Transformer Blocks
    # Use PyTorch outputs as input for the next stage to prevent error cascade
    jax_x = jnp.array(pt_x_np)
    jax_c = jnp.array(pt_c_np)

    for i in range(len(pt_model.blocks)):
        print(f"\n--- Block {i} ---")
        # PyTorch
        pt_x = pt_model.blocks[i](pt_x, pt_c)
        # JAX
        jax_x = jax_model.layers[i](jax_x, jax_c)

        # Comparison
        pt_x_np = print_comparison(pt_x.squeeze(0), jax_x, f"Block {i} Output")
        # Update JAX input for next iteration with the PyTorch ground truth
        jax_x = jnp.array(pt_x_np)

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
    from models import unpatchify
    pt_final = unpatchify(pt_x, pt_z.shape[2])
    # JAX
    jax_final = jax_model.mapper.convert_to_patches(jax_x)
    print_comparison(pt_final.squeeze(0), jax_final, "Final Image Output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    args = parser.parse_args()
    main(args)
