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
#  Swin Attention Rollout (3D)
# ─────────────────────────────────────────────────────────────

class AttentionRollout3D:
    """Attention Rollout for 3D Swin Transformer.

    Extends attention_viz.py pattern to 3D window attention.
    Reference: Abnar & Zuidema, 2020 — "Quantifying Attention Flow".

    Collects attention maps from SwinTransformer3D.get_attention_maps()
    and rolls them through the layer graph to get input-space attribution.
    """

    def __init__(self, model: nn.Module, discard_ratio: float = 0.9):
        self.model         = model
        self.discard_ratio = discard_ratio

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Return rolled attention attribution volume.

        Args:
            input_tensor: (1, 1, 64, 64, 64)

        Returns:
            attribution: (D', H', W') normalised to [0, 1]
                         in patch-grid resolution, needs upsampling for overlay
                         LABEL: MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE
        """
        self.model.eval()
        attn_maps = self.model.get_attention_maps(input_tensor)

        result = None
        for attn in attn_maps:
            # attn: (num_windows * B, num_heads, N_tokens, N_tokens)
            # Average over heads
            head_avg = attn.mean(dim=1)           # (nW, N, N)
            # Discard lowest attention weights (noise reduction)
            flat = head_avg.view(head_avg.shape[0], -1)
            threshold = flat.quantile(self.discard_ratio, dim=-1,
                                      keepdim=True).unsqueeze(-1)
            head_avg = head_avg * (head_avg >= threshold).float()

            # Add residual connection (identity + attention)
            eye = torch.eye(head_avg.shape[-1],
                            device=head_avg.device).unsqueeze(0)
            head_avg = head_avg + eye
            # Normalise rows
            row_sum = head_avg.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            head_avg = head_avg / row_sum

            if result is None:
                result = head_avg.mean(dim=0)       # (N, N)
            else:
                result = (head_avg.mean(dim=0) @ result)

        if result is None:
            return np.zeros((16, 16, 16), dtype=np.float32)

        # CLS-to-patch attention (first row if CLS exists, else mean)
        rollout = result[0] if result.shape[0] > 1 else result.mean(0)
        rollout = rollout.cpu().numpy()

        # Reshape to patch grid (16³ for 64³ input with patch_size=4)
        n_tokens = rollout.shape[0]
        grid_dim = int(round(n_tokens ** (1/3)))
        if grid_dim ** 3 == n_tokens:
            rollout_3d = rollout.reshape(grid_dim, grid_dim, grid_dim)
        else:
            # Fallback: fill a cube
            g = int(math.ceil(n_tokens ** (1/3)))
            tmp = np.zeros(g**3, dtype=np.float32)
            tmp[:n_tokens] = rollout
            rollout_3d = tmp.reshape(g, g, g)

        if rollout_3d.max() > 0:
            rollout_3d = rollout_3d / rollout_3d.max()

        return rollout_3d.astype(np.float32)

    @staticmethod
    def upsample_to_volume(attribution_3d: np.ndarray,
                           target_shape: tuple) -> np.ndarray:
        """Upsample patch-grid attribution to full volume for overlay."""
        t = torch.from_numpy(attribution_3d).unsqueeze(0).unsqueeze(0).float()
        out = F.interpolate(t, size=target_shape, mode="trilinear",
                            align_corners=False)
        return out.squeeze().numpy()


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


import math  # noqa: E402 — needed for AttentionRollout3D
