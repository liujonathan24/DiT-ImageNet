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
    config = modelConfig(type='DiT-XL')
    return config

def convert_weights(pytorch_weights_path, jax_model):
    """Converts and loads PyTorch weights into the JAX model."""
    
    # Load PyTorch weights
    pt_weights = torch.load(pytorch_weights_path, map_location="cpu")
    # print(pt_weights.keys())
    # Get JAX model state
    jax_state = nnx.state(jax_model)
    flat_state_with_paths, treedef = jax.tree_util.tree_flatten_with_path(jax_state)
    print(treedef)
    # --- Weight Mapping ---
    weight_mapping = {
        # Class Embedder
        "y_embedder.embedding..value": ("y_embedder.embedding_table.weight", False),

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
        weight_mapping[f"layers.{i}.cLinWeights.kernel..value"] = (f"blocks.{i}.adaLN_modulation.1.weight", True)
        weight_mapping[f"layers.{i}.cLinWeights.bias..value"] = (f"blocks.{i}.adaLN_modulation.1.bias", False)

    # --- Verification Setup ---
    all_pt_keys = set(pt_weights.keys())
    updated_jax_keys = set()

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
        # print(path_str)
        if path_str in weight_mapping:
            #print("======" if path_str=="pos_embed")
            pt_name, should_transpose = weight_mapping[path_str]

            if pt_name not in pt_weights:
                print(f"Warning: PyTorch weight {pt_name} for JAX param {path_str} not found in checkpoint. Skipping.")
                new_flat_state.append(jax_value)
                continue

            value = pt_weights[pt_name].detach().cpu().numpy()

            if pt_name == 'pos_embed' and value.ndim == 3:
                print("hello?")
                value = np.squeeze(value, axis=0)

            if should_transpose:
                if value.ndim == 2:
                    value = value.T
                elif value.ndim == 4 and hasattr(jax_value, 'ndim') and jax_value.ndim == 4:
                    value = np.transpose(value, (2, 3, 1, 0))

            if hasattr(jax_value, 'shape') and value.shape != jax_value.shape:
                print(f"Shape mismatch for {path_str}: JAX is {jax_value.shape}, PyTorch is {value.shape}. Skipping.")
                new_flat_state.append(jax_value)
                continue
            
            # If we reach here, the conversion is successful for this key
            updated_jax_keys.add(path_str)
            # Remove the used key from the set of all PyTorch keys
            if pt_name in all_pt_keys:
                all_pt_keys.remove(pt_name)
            
            new_flat_state.append(jnp.asarray(value, dtype=jax_value.dtype))
        else:
            new_flat_state.append(jax_value)

    # Reconstruct the state
    new_jax_state = jax.tree_util.tree_unflatten(treedef, new_flat_state)

    # --- Verification Asserts ---
    print("\n--- Verification --- ")
    unupdated_jax_keys = set(weight_mapping.keys()) - updated_jax_keys
    assert not unupdated_jax_keys, (
        f"Error: {len(unupdated_jax_keys)} JAX parameters in the mapping were not updated (e.g., due to shape mismatch):\n"
        f"{sorted(list(unupdated_jax_keys))}"
    )
    print("Success: All mapped JAX parameters were successfully updated.")

    # Assert that all keys from the PyTorch file have been used
    assert not all_pt_keys, (
        f"Error: {len(all_pt_keys)} weights from the PyTorch file were not mapped to the JAX model:\n"
        f"{sorted(list(all_pt_keys))}"
    )
    print("Success: All weights from the PyTorch file were successfully converted.")
    print("--------------------\n")

    return new_jax_state

def main(args):
    print("Creating DiT-XL model configuration...")
    model_config = modelConfig(type="DiT-XL")
    jax_model = DiffusionTransformer(model_config)

    print(f"\nStarting weight conversion from: {args.pytorch_checkpoint_path}")
    updated_jax_state = convert_weights(args.pytorch_checkpoint_path, jax_model)

    nnx.update(jax_model, updated_jax_state)

    if args.output_dir:
        print(f"\nSaving converted JAX model to: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        import orbax.checkpoint as ocp
        from helpers.config import trainConfig
        mngr = ocp.CheckpointManager(args.output_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1))
        class DummyArgs:
            pass
        dummy_args = DummyArgs()
        save_checkpoint(mngr, jax_model, optimizer=None, epoch=0, trainconfig=trainConfig(), args=dummy_args)
        print("Save complete.")

if __name__ == "__main__":
    """Run: 
    python -m scripts.convert_weights -pt /scratch/network/jl0796/Torch_DiT/pretrained_models/DiT-XL-2-256x256.pt -o /scratch/network/jl0796/DiT-ImageNet/pretrained
    """
    parser = argparse.ArgumentParser(description="Convert PyTorch DiT weights to JAX.")
    parser.add_argument("--pytorch_checkpoint_path", "-pt", type=str, required=True, help="Path to the PyTorch .pt or .pth checkpoint file.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save the converted JAX checkpoint.")
    args = parser.parse_args()
    main(args)
