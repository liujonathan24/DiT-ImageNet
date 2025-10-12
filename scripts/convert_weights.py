import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import os
import argparse

from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig
from helpers.checkpoint import save_checkpoint

def get_jax_state_dict(model):
    """Returns a flattened dictionary of the JAX model's state."""
    state = nnx.state(model)
    return nnx.graph.flatten(state)

def convert_weights(pytorch_weights_path, jax_model):
    """Converts and loads PyTorch weights into the JAX model."""
    
    # Load PyTorch weights
    pt_weights = torch.load(pytorch_weights_path, map_location="cpu")

    # Get JAX model state
    jax_state = nnx.state(jax_model)
    jax_flat_state_list = nnx.graph.flatten(jax_state)
    
    # Create a dictionary with string keys for easier lookup
    try:
        jax_flat_state_dict = {".".join(path): value for (path, kind), value in jax_flat_state_list}
    except Exception as e:
        print("Error processing flattened state. Printing raw state for debugging:")
        for i, item in enumerate(jax_flat_state_list[:5]):
            print(f"Item {i}: {item}")
        raise e

    print("Available JAX parameter keys:", list(jax_flat_state_dict.keys()))

    # --- Weight Mapping ---
    # This mapping is a starting point and may need adjustment
    # based on the exact names in the PyTorch checkpoint.
    weight_mapping = {
        # Patch Embedder
        "mapper.patch_embeddings.kernel": ("x_embedder.proj.weight", True),
        "mapper.patch_embeddings.bias": ("x_embedder.proj.bias", False),

        # Positional Embedding
        "pos_embed": ("pos_embed", False),

        # Time MLP
        "time_MLP.fc1.kernel": ("t_embedder.mlp.0.weight", True),
        "time_MLP.fc1.bias": ("t_embedder.mlp.0.bias", False),
        "time_MLP.fc2.kernel": ("t_embedder.mlp.2.weight", True),
        "time_MLP.fc2.bias": ("t_embedder.mlp.2.bias", False),

        # Final Layer
        "final_layer.linear.kernel": ("final_layer.linear.weight", True),
        "final_layer.linear.bias": ("final_layer.linear.bias", False),
        # TODO: Map final_layer.linWeights
    }

    # Add DiT block mappings
    for i in range(jax_model.n_layers):
        # MHA
        weight_mapping[f"layers.{i}.MHA.qkv_proj.kernel"] = (f"blocks.{i}.attn.qkv.weight", True)
        weight_mapping[f"layers.{i}.MHA.qkv_proj.bias"] = (f"blocks.{i}.attn.qkv.bias", False)
        weight_mapping[f"layers.{i}.MHA.out_proj.kernel"] = (f"blocks.{i}.attn.proj.weight", True)
        weight_mapping[f"layers.{i}.MHA.out_proj.bias"] = (f"blocks.{i}.attn.proj.bias", False)

        # MLP
        weight_mapping[f"layers.{i}.MLP.fc1.kernel"] = (f"blocks.{i}.mlp.fc1.weight", True)
        weight_mapping[f"layers.{i}.MLP.fc1.bias"] = (f"blocks.{i}.mlp.fc1.bias", False)
        weight_mapping[f"layers.{i}.MLP.fc2.kernel"] = (f"blocks.{i}.mlp.fc2.weight", True)
        weight_mapping[f"layers.{i}.MLP.fc2.bias"] = (f"blocks.{i}.mlp.fc2.bias", False)

        # Conditioning Parameters (adaLN)
        # TODO: These are complex and need careful mapping.
        # The original DiT combines these into a single `adaLN_modulation` layer.
        # You will need to investigate how to split the weights for your implementation.
        # weight_mapping[f"layers.{i}.cLinWeights.kernel"] = (f"blocks.{i}.adaLN_modulation.1.weight", True)
        # weight_mapping[f"layers.{i}.cScaleWeights.kernel"] = (f"...", True)

    # --- Conversion Loop ---
    new_jax_params = {}
    for jax_name, jax_value in jax_flat_state_dict.items():
        if jax_name in weight_mapping:
            pt_name, should_transpose = weight_mapping[jax_name]

            if pt_name not in pt_weights:
                print(f"Warning: PyTorch weight {pt_name} not found for JAX param {jax_name}. Using original JAX param.")
                new_jax_params[jax_name] = jax_value
                continue

            print(f"Converting: {pt_name} -> {jax_name}")

            # Convert tensor to numpy array
            value = pt_weights[pt_name].detach().cpu().numpy()

            # Transpose if necessary
            if should_transpose:
                if value.ndim == 2: # For Linear layers
                    value = value.T
                elif value.ndim == 4 and jax_value.ndim == 4: # For Conv layers
                    # PyTorch: (out_channels, in_channels, kernel_height, kernel_width)
                    # JAX/Flax: (kernel_height, kernel_width, in_channels, out_channels)
                    value = np.transpose(value, (2, 3, 1, 0))

            # Check shapes
            if value.shape != jax_value.shape:
                print(f"Shape mismatch for {jax_name}: JAX is {jax_value.shape}, PyTorch is {value.shape} after potential transpose. Skipping.")
                new_jax_params[jax_name] = jax_value
                continue

            new_jax_params[jax_name] = jnp.asarray(value)
        else:
            # Keep original parameter if no mapping is defined
            new_jax_params[jax_name] = jax_value

    # Reconstruct the state dict from the list of (key, value) tuples
    new_jax_state = nnx.graph.unflatten(list(new_jax_params.items()))

    print("\nWeight conversion process finished.")
    print("Please review the warnings and fill in the missing mappings (TODOs).")

    return new_jax_state
    print("Please review the warnings and fill in the missing mappings (TODOs).")

    return new_jax_state

def main(args):
    # 1. Create JAX model with DiT-XL config
    print("Creating DiT-XL model configuration...")
    model_config = modelConfig(type="DiT-XL")
    jax_model = DiffusionTransformer(model_config)

    # 2. Convert weights
    print(f"\nStarting weight conversion from: {args.pytorch_checkpoint_path}")
    updated_jax_state = convert_weights(args.pytorch_checkpoint_path, jax_model)

    # 3. Update the model with the new state
    nnx.update(jax_model, updated_jax_state)

    # 4. Save the converted JAX model
    if args.output_dir:
        print(f"\nSaving converted JAX model to: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        # We need a dummy optimizer and other objects for the save function
        import orbax.checkpoint as ocp
        from helpers.config import trainConfig
        mngr = ocp.CheckpointManager(args.output_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1))
        # The save_checkpoint function expects an optimizer, epoch, etc.
        # We can pass dummy values for this conversion script.
        class DummyArgs:
            pass
        dummy_args = DummyArgs()
        save_checkpoint(mngr, jax_model, optimizer=None, epoch=0, trainconfig=trainConfig(), args=dummy_args)
        print("Save complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch DiT weights to JAX.")
    parser.add_argument("--pytorch_checkpoint_path", "-pt", type=str, required=True, help="Path to the PyTorch .pt or .pth checkpoint file.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save the converted JAX checkpoint.")
    args = parser.parse_args()
    main(args)
