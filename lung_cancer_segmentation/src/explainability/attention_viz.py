# src/explainability/attention_viz.py
"""Attention map visualization for Vision Transformer and Hybrid CNN-ViT models.
Extracts and visualizes attention weights from transformer blocks.
"""
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def extract_attention_map(model, input_tensor: torch.Tensor, layer_idx: int = -1) -> np.ndarray:
    """Extract attention map from a ViT or Hybrid model.

    Args:
        model: A model with get_attention_maps() method.
        input_tensor: (1, C, H, W) tensor.
        layer_idx: Which transformer block to visualize (-1 = last).

    Returns:
        attention_map: (H_patches, W_patches) numpy array in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        attn_maps = model.get_attention_maps(input_tensor)

    attn = attn_maps[layer_idx]  # (B, heads, N, N) where N = 1+n_patches
    # Average across heads
    attn_avg = attn.mean(dim=1)  # (B, N, N)
    # Take CLS token attention to all patch tokens
    cls_attn = attn_avg[0, 0, 1:]  # (n_patches,)
    n = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(n, n).cpu().numpy()

    # Normalize to [0, 1]
    if attn_map.max() > 0:
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    return attn_map


def attention_rollout(model, input_tensor: torch.Tensor) -> np.ndarray:
    """Attention rollout – multiply attention matrices across all layers.

    This gives a more accurate picture of information flow from input patches
    to the CLS token across the full depth of the transformer.

    Returns:
        rollout_map: (H_patches, W_patches) numpy array in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        attn_maps = model.get_attention_maps(input_tensor)

    # Initialise with identity
    result = torch.eye(attn_maps[0].shape[-1])
    for attn in attn_maps:
        # Average across heads
        attn_avg = attn.mean(dim=1)[0]  # (N, N)
        # Add identity (residual connection)
        attn_with_res = 0.5 * attn_avg + 0.5 * torch.eye(attn_avg.shape[0])
        # Row-normalize
        attn_with_res = attn_with_res / attn_with_res.sum(dim=-1, keepdim=True)
        result = result @ attn_with_res

    # CLS token row → patch columns (skip CLS itself)
    cls_attn = result[0, 1:].cpu().numpy()
    n = int(cls_attn.shape[0] ** 0.5)
    rollout = cls_attn.reshape(n, n)
    if rollout.max() > 0:
        rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min())
    return rollout


def save_attention_figure(img: Image.Image, attn_map: np.ndarray, title: str, path: str):
    """Save a side-by-side figure of original image + attention heatmap."""
    w, h = img.size
    attn_resized = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((w, h)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(attn_resized, cmap="viridis")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attn_resized, cmap="viridis", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
