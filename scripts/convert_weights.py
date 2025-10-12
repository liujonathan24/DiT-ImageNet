import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import os
import argparse
import jax.tree_util

from diffusion_transformer import DiffusionTransformer
from helpers.config import modelConfig
from helpers.checkpoint import save_checkpoint

def create_dit_xl_config():
    """Creates a modelConfig for DiT-XL."""
    config = modelConfig()
    config.n_layers = 28
    config.n_heads = 16
    config.DiT_hidden_size = 1152
    config.patch_size = 2
    return config

def convert_weights(pytorch_weights_path, jax_model):
    """Converts and loads PyTorch weights into the JAX model."""
    
    # Load PyTorch weights
    pt_weights = torch.load(pytorch_weights_path, map_location="cpu")

    # Get the JAX model's state
    jax_state = nnx.state(jax_model)
    
    # Flatten the state using JAX's tree utility to get paths and values
    flat_state_with_paths, treedef = jax.tree_util.tree_flatten_with_path(jax_state)

    # --- Weight Mapping ---
    # Updated with the correct '..value' suffix
    weight_mapping = {
        # Patch Embedder
        "mapper.patch_embeddings.kernel..value": ("x_embedder.proj.weight", True),
        "mapper.patch_embeddings.bias..value": ("x_embedder.proj.bias", False),

        # Positional Embedding
        "pos_embed": ("pos_embed", False),

        # Time MLP
        "time_MLP.fc1.kernel..value": ("t_embedder.mlp.0.weight", True),
        "time_MLP.fc1.bias..value": ("t_embedder.mlp.0.bias", False),
        "time_MLP.fc2.kernel..value": ("t_embedder.mlp.2.weight", True),
        "time_MLP.fc2.bias..value": ("t_embedder.mlp.2.bias", False),

        # Final Layer
        "final_layer.linear.kernel..value": ("final_layer.linear.weight", True),
        "final_layer.linear.bias..value": ("final_layer.linear.bias", False),
        "final_layer.linWeights.kernel..value": ("final_layer.adaLN_modulation.1.weight", True),
        "final_layer.linWeights.bias..value": ("final_layer.adaLN_modulation.1.bias", False),
    }

    # Add DiT block mappings
    for i in range(jax_model.n_layers):
        # MHA
        weight_mapping[f"layers.{i}.MHA.qkv_proj.kernel..value"] = (f"blocks.{i}.attn.qkv.weight", True)
        weight_mapping[f"layers.{i}.MHA.qkv_proj.bias..value"] = (f"blocks.{i}.attn.qkv.bias", False)
        weight_mapping[f"layers.{i}.MHA.out_proj.kernel..value"] = (f"blocks.{i}.attn.proj.weight", True)
        weight_mapping[f"layers.{i}.MHA.out_proj.bias..value"] = (f"blocks.{i}.attn.proj.bias", False)

        # MLP
        weight_mapping[f"layers.{i}.MLP.fc1.kernel..value"] = (f"blocks.{i}.mlp.fc1.weight", True)
        weight_mapping[f"layers.{i}.MLP.fc1.bias..value"] = (f"blocks.{i}.mlp.fc1.bias", False)
        weight_mapping[f"layers.{i}.MLP.fc2.kernel..value"] = (f"blocks.{i}.mlp.fc2.weight", True)
        weight_mapping[f"layers.{i}.MLP.fc2.bias..value"] = (f"blocks.{i}.mlp.fc2.bias", False)

        # Conditioning Parameters (adaLN)
        # This is a guess, you may need to verify the PyTorch model structure
        weight_mapping[f"layers.{i}.cLinWeights.kernel..value"] = (f"blocks.{i}.adaLN_modulation.1.weight", True)
        weight_mapping[f"layers.{i}.cLinWeights.bias..value"] = (f"blocks.{i}.adaLN_modulation.1.bias", False)
        # NOTE: cScaleWeights might not have a direct mapping and could be part of the adaLN_modulation weights.
        # This requires inspecting the PyTorch model's adaLN_modulation layer.

    # --- Conversion Loop ---
    new_flat_state = []
    for key_path, jax_value in flat_state_with_paths:
        path_parts = []
        for k in key_path:
            if hasattr(k, 'idx'):
                path_parts.append(str(k.idx))
            elif hasattr(k, 'key'):
                path_parts.append(str(k.key))
            else:
                path_parts.append(str(k))
        path_str = ".".join(path_parts)

        if path_str.startswith('.'): path_str = path_str[1:]

        if path_str in weight_mapping:
            pt_name, should_transpose = weight_mapping[path_str]

            if pt_name not in pt_weights:
                print(f"Warning: PyTorch weight {pt_name} not found for JAX param {path_str}. Using original JAX param.")
                new_flat_state.append(jax_value)
                continue

            # print(f"Converting: {pt_name} -> {path_str}")

            # Convert tensor to numpy array
            value = pt_weights[pt_name].detach().cpu().numpy()

            # Special case for pos_embed shape
            if pt_name == 'pos_embed' and value.ndim == 3:
                # print("Squeezing extra dimension from pos_embed.")
                value = np.squeeze(value, axis=0)

            # Transpose if necessary
            if should_transpose:
                if value.ndim == 2: # For Linear layers
                    value = value.T
                elif value.ndim == 4 and hasattr(jax_value, 'ndim') and jax_value.ndim == 4: # For Conv layers
                    # PyTorch: (out_channels, in_channels, kernel_height, kernel_width)
                    # JAX/Flax: (kernel_height, kernel_width, in_channels, out_channels)
                    value = np.transpose(value, (2, 3, 1, 0))

            # Check shapes
            if hasattr(jax_value, 'shape') and value.shape != jax_value.shape:
                print(f"Converting: {pt_name} -> {path_str}")
                print(f"Shape mismatch for {path_str}: JAX is {jax_value.shape}, PyTorch is {value.shape} after potential transpose. Skipping.")
                new_flat_state.append(jax_value)
                continue

            new_flat_state.append(jnp.asarray(value))
        else:
            # Keep original parameter if no mapping is defined
            new_flat_state.append(jax_value)

    # Reconstruct the state using the new flattened list of values
    new_jax_state = jax.tree_util.tree_unflatten(treedef, new_flat_state)

    print("\nWeight conversion process finished.")
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
