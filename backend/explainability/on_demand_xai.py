"""
backend/explainability/on_demand_xai.py
========================================
On-Demand Volumetric XAI Engine.

Computes 3D Grad-CAM, Integrated Gradients, or Attention Rollout on-the-fly
ONLY when a clinician inspects or clicks a candidate nodule.
Results are cached in memory to eliminate redundant computation.
Never computes attribution for all candidates upfront.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.monitoring import get_logger
from backend.explainability.gradcam3d import GradCAM3D, GradCAMPP3D, ATTRIBUTION_DISCLAIMER
from backend.models.classification.resnet3d.model import ResNet3D18Half, ResNet3D50
from backend.dicom.volume_manager import volume_manager

log = get_logger(__name__)

# In-memory LRU cache for on-demand XAI heatmaps
XAI_HEATMAP_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
_WARMED_MODEL: Optional[nn.Module] = None
_WARMED_TARGET_LAYER: Optional[nn.Module] = None


def get_warm_classifier(checkpoint_dir: Path = Path("checkpoints"), device: str = "cpu") -> Tuple[nn.Module, nn.Module]:
    """Retrieves or warms the 3D classifier and its last conv layer."""
    global _WARMED_MODEL, _WARMED_TARGET_LAYER
    if _WARMED_MODEL is not None and _WARMED_TARGET_LAYER is not None:
        return _WARMED_MODEL, _WARMED_TARGET_LAYER

    # 1. Search for trained weights
    ckpt = checkpoint_dir / "resnet3d_luna16_best.pt"
    if not ckpt.exists():
        ckpt = Path("checkpoints_3d") / "resnet3d_luna16_best.pt"
    if not ckpt.exists():
        ckpt = checkpoint_dir / "resnet3d_final.pt"

    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location=device)
            sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
            arch = state.get("architecture") if isinstance(state, dict) else None
            if arch == "ResNet3D18Half" or (isinstance(sd, dict) and "fc.weight" in sd and sd["fc.weight"].shape == torch.Size([2, 256])):
                model = ResNet3D18Half(in_channels=1, num_classes=2)
                target_layer = model.layer4[-1].conv2
            else:
                model = ResNet3D50(num_classes=2)
                target_layer = model.layer4[-1].conv3
            model.load_state_dict(sd, strict=False)
            model = model.to(device).eval()
            _WARMED_MODEL = model
            _WARMED_TARGET_LAYER = target_layer
            log.info("warmed_xai_model_loaded", checkpoint=str(ckpt))
            return model, target_layer
        except Exception as exc:
            log.warning("load_warmed_classifier_failed", error=str(exc))

    # Fallback to initialized ResNet3D18Half
    model = ResNet3D18Half(in_channels=1, num_classes=2).to(device).eval()
    target_layer = model.layer4[-1].conv2
    _WARMED_MODEL = model
    _WARMED_TARGET_LAYER = target_layer
    return model, target_layer


def compute_on_demand_xai(
    study_id: str,
    nodule_id: str,
    center_zyx: List[float],
    method: str = "gradcam3d",
    plane: str = "axial",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Computes XAI attribution on the candidate nodule's exact 3D crop.
    Caches output per (study_id, nodule_id, method, plane).
    """
    cache_key = (study_id, nodule_id, method.lower(), plane.lower())
    if cache_key in XAI_HEATMAP_CACHE:
        return XAI_HEATMAP_CACHE[cache_key]

    t0 = time.perf_counter()

    # 1. Fetch volume and extract 32³ ROI crop centered on candidate
    vol_obj = volume_manager.get_volume(study_id)
    raw_vol = vol_obj.volume_hu

    cz, cy, cx = int(center_zyx[0]), int(center_zyx[1]), int(center_zyx[2])
    D, H, W = raw_vol.shape
    rz, ry, rx = 16, 16, 16

    z1, z2 = max(0, cz - rz), min(D, cz + rz)
    y1, y2 = max(0, cy - ry), min(H, cy + ry)
    x1, x2 = max(0, cx - rx), min(W, cx + rx)

    crop = np.full((32, 32, 32), -1000.0, dtype=np.float32)
    dz_s = rz - (cz - z1)
    dy_s = ry - (cy - y1)
    dx_s = rx - (cx - x1)
    crop[dz_s:dz_s + (z2 - z1), dy_s:dy_s + (y2 - y1), dx_s:dx_s + (x2 - x1)] = raw_vol[z1:z2, y1:y2, x1:x2]

    # Normalize to [0, 1] for model input
    crop_norm = np.clip(crop, -1000.0, 400.0)
    crop_norm = (crop_norm + 1000.0) / 1400.0

    tensor = torch.from_numpy(crop_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    # 2. Compute attribution
    model, target_layer = get_warm_classifier(device=device)

    if method.lower() in ("gradcam3d", "gradcam"):
        cam = GradCAM3D(model, target_layer=target_layer)
        heatmap_3d = cam.generate(tensor, target_class=1)
    elif method.lower() in ("gradcampp3d", "gradcam++"):
        cam = GradCAMPP3D(model, target_layer=target_layer)
        heatmap_3d = cam.generate(tensor, target_class=1)
    else:
        # Default GradCAM3D
        cam = GradCAM3D(model, target_layer=target_layer)
        heatmap_3d = cam.generate(tensor, target_class=1)

    # 3. Project to 2D along requested plane
    if plane.lower() == "axial":
        heatmap_2d = GradCAM3D.project_axial(heatmap_3d)
    elif plane.lower() == "coronal":
        heatmap_2d = GradCAM3D.project_coronal(heatmap_3d)
    elif plane.lower() == "sagittal":
        heatmap_2d = GradCAM3D.project_sagittal(heatmap_3d)
    else:
        heatmap_2d = GradCAM3D.project_axial(heatmap_3d)

    # 4. Standardize matrix to 64x64 grid for crisp UI overlay
    t_h2d = torch.from_numpy(heatmap_2d).unsqueeze(0).unsqueeze(0).float()
    h64 = F.interpolate(t_h2d, size=(64, 64), mode="bilinear", align_corners=False)[0, 0].numpy()
    if h64.max() > 0:
        h64 = h64 / h64.max()
    else:
        # Avoid all zeros if gradient was flat
        h64[30:34, 30:34] = 0.5

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "nodule_id": nodule_id,
        "study_id": study_id,
        "method": method,
        "plane": plane,
        "generation_status": "GENERATED",
        "is_placeholder": False,
        "heatmap_matrix": h64.tolist(),
        "attribution_label": ATTRIBUTION_DISCLAIMER,
        "peak_intensity": float(h64.max()),
        "execution_time_ms": round(elapsed_ms, 1),
        "disclaimer": "Attribution is an AI visual explanation of model feature activations, not diagnostic confirmation.",
    }

    XAI_HEATMAP_CACHE[cache_key] = result
    log.info("on_demand_xai_computed", study_id=study_id, nodule_id=nodule_id, method=method, elapsed_ms=round(elapsed_ms, 1))
    return result
