"""
backend/explainability/gradcam3d.py
=====================================
3D Grad-CAM and Grad-CAM++ for volumetric CNN classifiers.

Extends existing GradCAM / GradCAMPP from:
  lung_cancer_segmentation/src/explainability/gradcam.py

Additions:
  - 3D spatial pooling over (D, H, W) instead of (H, W)
  - Slice-wise projection to axial/coronal/sagittal 2D views
  - Works with ResNet3D50, DenseNet3D121 via get_feature_maps()
  - Enforces "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE" label
  - MultiScaleGradCAM3D: fuses a fine + coarse conv layer for sharper,
    less-blurred boundaries than single-layer Grad-CAM
  - AttentionRollout3D: rewritten to be window/shift-aware — the original
    version's rollout matmul silently mixed unrelated local windows
    together; see class docstring for why that was spatially invalid

Usage::
    cam = GradCAM3D(model, target_layer=model.layer4[-1].conv3)
    heatmap_3d = cam.generate(input_tensor, target_class=1)
    axial = cam.project_axial(heatmap_3d)   # max-intensity projection
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.models.classification.swin3d.model import window_reverse_3d

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# Attribution disclaimer — always visible in UI
ATTRIBUTION_DISCLAIMER = "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE"


# ─────────────────────────────────────────────────────────────
#  3D Grad-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM3D:
    """Standard 3D Grad-CAM for convolutional layers.

    Args:
        model:        PyTorch model (ResNet3D50, DenseNet3D121, etc.)
        target_layer: The nn.Module whose output we hook (last conv layer).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """Generate 3D Grad-CAM heatmap.

        Args:
            input_tensor: (1, 1, D, H, W) CT volume tensor.
            target_class: Class index (0=benign, 1=malignant).
                          If None, uses predicted class.

        Returns:
            heatmap_3d: (D, H, W) numpy array, values in [0, 1].
                        LABEL: MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Pool gradients over spatial dims (D, H, W)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)   # (1, C, 1, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()                             # (d, h, w)

        # Resize to input spatial size
        D, H, W = input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_resized = F.interpolate(cam_tensor, size=(D, H, W),
                                    mode="trilinear",
                                    align_corners=False).squeeze().numpy()

        if cam_resized.max() > 0:
            cam_resized = cam_resized / cam_resized.max()
        return cam_resized.astype(np.float32)

    # ── Projection helpers ─────────────────────────────────────

    @staticmethod
    def project_axial(heatmap_3d: np.ndarray) -> np.ndarray:
        """Max-intensity projection along depth → (H, W)."""
        return heatmap_3d.max(axis=0)

    @staticmethod
    def project_coronal(heatmap_3d: np.ndarray) -> np.ndarray:
        """MIP along height → (D, W)."""
        return heatmap_3d.max(axis=1)

    @staticmethod
    def project_sagittal(heatmap_3d: np.ndarray) -> np.ndarray:
        """MIP along width → (D, H)."""
        return heatmap_3d.max(axis=2)

    @staticmethod
    def get_slice(heatmap_3d: np.ndarray, axis: str, index: int) -> np.ndarray:
        """Extract a single slice from the 3D heatmap.

        Args:
            axis:  'axial' | 'coronal' | 'sagittal'
            index: slice index along that axis
        """
        if axis == "axial":
            return heatmap_3d[index, :, :]
        elif axis == "coronal":
            return heatmap_3d[:, index, :]
        elif axis == "sagittal":
            return heatmap_3d[:, :, index]
        raise ValueError(f"Unknown axis: {axis}")


# ─────────────────────────────────────────────────────────────
#  3D Grad-CAM++ (extends GradCAM3D)
# ─────────────────────────────────────────────────────────────

class GradCAMPP3D(GradCAM3D):
    """3D Grad-CAM++: improved localization via second-order gradients.

    Directly ports GradCAMPP from gradcam.py to 3D convolutions.
    Reference: Chattopadhay et al., WACV 2018.
    """

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        grads = self.gradients   # (1, C, d, h, w)
        acts  = self.activations

        # Grad-CAM++ alpha weights (3D)
        grads_sq = grads ** 2
        grads_cb = grads ** 3
        sum_acts = acts.sum(dim=(2, 3, 4), keepdim=True)
        alpha = grads_sq / (2 * grads_sq + sum_acts * grads_cb + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3, 4), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()

        D, H, W = input_tensor.shape[2:5]
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam_r = F.interpolate(cam_t, size=(D, H, W),
                              mode="trilinear",
                              align_corners=False).squeeze().numpy()

        if cam_r.max() > 0:
            cam_r = cam_r / cam_r.max()
        return cam_r.astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Swin Attention Rollout (3D) — window-aware, spatially correct
# ─────────────────────────────────────────────────────────────

class AttentionRollout3D:
    """Attention Rollout for 3D (shifted) Window Swin Transformer.

    IMPORTANT — why this is not the textbook ViT rollout:
    Swin's attention is *local*: each block's attention matrix only ever
    covers one window (e.g. 4x4x4 = 64 tokens), never the full token grid.
    There is also no CLS token — SwinTransformer3D pools the whole grid at
    the end. The original textbook rollout (treat each layer's attention as
    one global NxN matrix and chain-multiply, then read off the "CLS" row)
    is therefore invalid here: it silently averages together attention
    matrices from different, non-overlapping windows as if they were the
    same matrix, and the reshaped result has no correspondence to actual
    voxel positions.

    This version instead computes, per block, how much each token is
    attended to *within its own window* (column-sum of the row-normalised
    attention matrix), scatters that window-local relevance back to its
    true position in the (D, H, W) token grid with window_reverse_3d
    (undoing the same cyclic shift the block applied), and accumulates
    across blocks/stages in grid space — downsampling between stages to
    match each stage's patch-merged resolution. That keeps every value
    tied to a real spatial location throughout.

    Reference: Abnar & Zuidema, 2020 — "Quantifying Attention Flow"
    (rollout logic), adapted here to windowed/shifted local attention.
    """

    def __init__(self, model: nn.Module, discard_ratio: float = 0.0):
        self.model = model
        # Kept for API compatibility; window attention is already local and
        # sharp, so aggressive discarding tends to just delete signal.
        self.discard_ratio = discard_ratio

    @staticmethod
    def _block_relevance_to_grid(block_info: dict) -> torch.Tensor:
        """Map one block's window attention to a (1, D, H, W, 1) grid map.

        block_info comes from SwinTransformer3D.get_attention_maps().
        """
        attn = block_info["attn"]                 # (nW*B, heads, N, N)
        ws = block_info["window_size"]
        shift = block_info["shift_size"]
        D, H, W = block_info["resolution"]
        wd, wh, ww = ws
        N = wd * wh * ww

        head_avg = attn.mean(dim=1)                # (nW, N, N), assumes B=1
        row_sum = head_avg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        head_avg = head_avg / row_sum              # row-normalise (already ~softmax)

        # How much each token is attended TO by the rest of its window —
        # a legitimate "salience within this local context" score even
        # without a CLS token.
        importance = head_avg.sum(dim=1)           # (nW, N)

        windows = importance.view(-1, wd, wh, ww, 1)
        grid = window_reverse_3d(windows, ws, D, H, W)   # (1, D, H, W, 1)

        # Undo the pre-attention cyclic shift the block applied, so this
        # lands in the same unshifted coordinate frame every block uses.
        if any(s > 0 for s in shift):
            grid = torch.roll(grid, shifts=shift, dims=(1, 2, 3))

        g = grid.squeeze(0).squeeze(-1)            # (D, H, W)
        if g.max() > 0:
            g = g / g.max()
        return g

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Return rolled attention attribution volume.

        Args:
            input_tensor: (1, 1, D, H, W), e.g. (1, 1, 64, 64, 64)

        Returns:
            attribution: (D', H', W') normalised to [0, 1], at the coarsest
                         stage's grid resolution — upsample with
                         upsample_to_volume() for overlay onto the CT.
                         LABEL: MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE
        """
        self.model.eval()
        attn_info = self.model.get_attention_maps(input_tensor)
        if not attn_info:
            return np.zeros((4, 4, 4), dtype=np.float32)

        accum: Optional[torch.Tensor] = None
        current_res: Optional[Tuple[int, int, int]] = None

        for info in attn_info:
            res = info["resolution"]
            block_map = self._block_relevance_to_grid(info)   # (D, H, W) at `res`

            if accum is None:
                accum = block_map
                current_res = res
                continue

            if res != current_res:
                # A patch-merge happened between blocks: downsample the
                # running accumulation to the new (coarser) resolution
                # before continuing the chain.
                t = accum.unsqueeze(0).unsqueeze(0)
                accum = F.adaptive_avg_pool3d(t, res).squeeze(0).squeeze(0)
                current_res = res

            # Rollout-style residual combination, done elementwise in grid
            # space (valid here, unlike a full NxN matmul across windows
            # that don't share token indices).
            accum = accum * (0.5 + 0.5 * block_map)
            if accum.max() > 0:
                accum = accum / accum.max()

        rollout_3d = accum.cpu().numpy().astype(np.float32)
        return rollout_3d

    @staticmethod
    def upsample_to_volume(attribution_3d: np.ndarray,
                           target_shape: tuple) -> np.ndarray:
        """Upsample patch-grid attribution to full volume for overlay."""
        t = torch.from_numpy(attribution_3d).unsqueeze(0).unsqueeze(0).float()
        out = F.interpolate(t, size=target_shape, mode="trilinear",
                            align_corners=False)
        return out.squeeze().numpy()


# ─────────────────────────────────────────────────────────────
#  Multi-scale Grad-CAM (fuses a fine + coarse layer for sharper,
#  better-localized boundaries than single-layer Grad-CAM)
# ─────────────────────────────────────────────────────────────

class MultiScaleGradCAM3D:
    """Fuses Grad-CAM from an earlier (higher-resolution, finer boundary)
    layer with a later (lower-resolution, more class-discriminative) layer.

    Single-last-layer Grad-CAM on a 3D CNN is typically computed on an
    8x8x8 or 4x4x4 feature map and then trilinearly upsampled ~8-16x to
    the input volume — this necessarily produces a blurry blob, not a
    boundary that hugs the actual nodule. Combining with an earlier layer
    (larger feature map, less upsampling, more spatial detail) sharpens
    the localization while keeping the semantic weighting from the deep
    layer that actually drove the classification decision.

    Usage::
        cam = MultiScaleGradCAM3D(model, fine_layer=model.layer3[-1].conv3,
                                          coarse_layer=model.layer4[-1].conv3)
        heatmap = cam.generate(input_tensor, target_class=1)
    """

    def __init__(self, model: nn.Module, fine_layer: nn.Module,
                 coarse_layer: nn.Module, fine_weight: float = 0.4):
        self.model = model
        self.fine_cam = GradCAM3D(model, fine_layer)
        self.coarse_cam = GradCAM3D(model, coarse_layer)
        self.fine_weight = fine_weight

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        # Each GradCAM3D call does its own forward+backward; since both
        # hooks are registered on the same model this executes cleanly on
        # two separate passes rather than trying to share one graph.
        fine = self.fine_cam.generate(input_tensor, target_class)
        coarse = self.coarse_cam.generate(input_tensor, target_class)

        # Weighted geometric-mean fusion: a voxel only stays "hot" if both
        # scales agree it matters, which suppresses the coarse map's
        # blurred halo around the true boundary.
        eps = 1e-6
        fused = (np.clip(fine, eps, 1.0) ** self.fine_weight) * \
                (np.clip(coarse, eps, 1.0) ** (1.0 - self.fine_weight))
        if fused.max() > 0:
            fused = fused / fused.max()
        return fused.astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  2D slice overlay utility (reused from existing gradcam.py)
# ─────────────────────────────────────────────────────────────

def overlay_heatmap_2d(ct_slice: np.ndarray,
                        heatmap_2d: np.ndarray,
                        alpha: float = 0.4,
                        colormap: str = "jet") -> np.ndarray:
    """Overlay a 2D heatmap on a CT slice.

    Args:
        ct_slice:   (H, W) float32 HU or normalised CT slice
        heatmap_2d: (H, W) float32 in [0, 1]
        alpha:      blend factor
        colormap:   matplotlib colormap name

    Returns:
        (H, W, 3) uint8 overlay
    """
    if not HAS_PLOT:
        raise ImportError("matplotlib and PIL required for overlay")

    # Normalise CT to [0, 255]
    ct_norm = (ct_slice - ct_slice.min())
    denom = ct_norm.max()
    if denom > 0:
        ct_norm = ct_norm / denom
    ct_rgb = (np.stack([ct_norm]*3, axis=-1) * 255).astype(np.uint8)

    # Apply colormap to heatmap
    cmap    = cm.get_cmap(colormap)
    colored = (cmap(heatmap_2d)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (ct_rgb * (1 - alpha) + colored * alpha).astype(np.uint8)
    return overlay
