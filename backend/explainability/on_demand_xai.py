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

# In-memory cache for on-demand XAI heatmaps
XAI_HEATMAP_CACHE: Dict[Tuple[str, str, str, str, Optional[int]], Dict[str, Any]] = {}
_WARMED_MODEL: Optional[nn.Module] = None
_WARMED_TARGET_LAYER: Optional[nn.Module] = None
_WARMED_CKPT_NAME: Optional[str] = None


def get_warm_classifier(
    checkpoint_path: Optional[Path] = None,
    checkpoint_dir: Path = Path("checkpoints_3d"),
    device: str = "cpu",
) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[str], Optional[str]]:
    """
    Retrieves or warms the 3D classifier and its last conv layer.
    Never returns untrained or randomly-initialized weights.
    Returns: (model, target_layer, error_reason, checkpoint_filename)
    """
    global _WARMED_MODEL, _WARMED_TARGET_LAYER, _WARMED_CKPT_NAME

    # 1. Resolve target checkpoint path
    ckpt: Optional[Path] = None
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.is_file():
            ckpt = p
        else:
            return None, None, f"Specified checkpoint path '{checkpoint_path}' does not exist", None
    else:
        # Search for trained weights in priority order
        candidates = [
            checkpoint_dir / "resnet3d_detector_finetuned.pt",
            checkpoint_dir / "resnet3d_luna16_best.pt",
            Path("checkpoints_3d") / "resnet3d_detector_finetuned.pt",
            Path("checkpoints_3d") / "resnet3d_luna16_best.pt",
            checkpoint_dir / "resnet3d_final.pt",
        ]
        for c in candidates:
            if c.is_file():
                ckpt = c
                break

    if ckpt is None or not ckpt.is_file():
        return None, None, f"Required model weights checkpoint not found in {checkpoint_dir}", None

    # Check cached warmed model if checkpoint matches
    if _WARMED_MODEL is not None and _WARMED_TARGET_LAYER is not None and _WARMED_CKPT_NAME == ckpt.name:
        return _WARMED_MODEL, _WARMED_TARGET_LAYER, None, _WARMED_CKPT_NAME

    try:
        state = torch.load(ckpt, map_location=device)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        arch = state.get("architecture") if isinstance(state, dict) else None
        if arch == "ResNet3D18Half" or (isinstance(sd, dict) and "fc.weight" in sd and sd["fc.weight"].shape == torch.Size([2, 256])):
            model = ResNet3D18Half(in_channels=1, num_classes=2)
            # For 32^3 inputs, layer4 has spatial resolution 1x1, which collapses Grad-CAM to a flat constant.
            # layer3 retains spatial resolution (8, 2, 2) providing genuine 3D spatial attribution.
            target_layer = model.layer2[-1].conv2
        else:
            model = ResNet3D50(num_classes=2)
            target_layer = model.layer4[-1].conv3

        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()
        _WARMED_MODEL = model
        _WARMED_TARGET_LAYER = target_layer
        _WARMED_CKPT_NAME = ckpt.name
        log.info("warmed_xai_model_loaded", checkpoint=str(ckpt))
        return model, target_layer, None, ckpt.name
    except Exception as exc:
        log.warning("load_warmed_classifier_failed", error=str(exc), checkpoint=str(ckpt))
        return None, None, f"Failed loading checkpoint '{ckpt.name}': {exc}", ckpt.name


def compute_on_demand_xai(
    study_id: str,
    nodule_id: str,
    center_zyx: List[float],
    method: str = "gradcam3d",
    plane: str = "axial",
    device: str = "cpu",
    slice_index: Optional[int] = None,
    checkpoint_name: Optional[str] = None,
    target_class_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes XAI attribution on the candidate nodule's exact 3D crop.
    Caches output per (study_id, nodule_id, method, plane, slice_index).
    """
    cache_key = (study_id, nodule_id, method.lower(), plane.lower(), slice_index)
    if cache_key in XAI_HEATMAP_CACHE:
        return XAI_HEATMAP_CACHE[cache_key]

    t0 = time.perf_counter()

    # 1. Fetch volume
    try:
        vol_obj = volume_manager.get_volume(study_id)
        raw_vol = vol_obj.volume_hu
        spacing = tuple(vol_obj.spacing_mm) if hasattr(vol_obj, "spacing_mm") else (1.0, 1.0, 1.0)
    except Exception as exc:
        return {
            "nodule_id": nodule_id,
            "study_id": study_id,
            "method": method,
            "plane": plane,
            "generation_status": "UNAVAILABLE",
            "status_reason": f"Volume unavailable for study '{study_id}': {exc}",
            "expected_checkpoint": checkpoint_name,
            "actual_checkpoint": None,
            "is_placeholder": True,
            "heatmap_matrix": None,
            "checkpoint": None,
            "attribution_label": ATTRIBUTION_DISCLAIMER,
            "disclaimer": "Attribution is unavailable because volume data could not be retrieved.",
        }

    # 2. Resolve the same checkpoint as the classification stage for that study
    resolved_ckpt_path: Optional[Path] = None
    stored_prob: Optional[float] = None
    try:
        from backend.api.service import STUDY_STORE
        s_data = STUDY_STORE.get(study_id, {})
        for f in s_data.get("nodules", []) + s_data.get("report", {}).get("findings", []):
            if f.get("nodule_id") == nodule_id:
                if f.get("checkpoint"):
                    resolved_ckpt_path = Path("checkpoints_3d") / f["checkpoint"]
                stored_prob = f.get("calibrated_probability") or f.get("raw_probability") or f.get("probability")
                break
        if resolved_ckpt_path is None and s_data.get("report", {}).get("classification_checkpoint"):
            resolved_ckpt_path = Path("checkpoints_3d") / s_data["report"]["classification_checkpoint"]
    except Exception:
        pass

    if checkpoint_name:
        resolved_ckpt_path = Path("checkpoints_3d") / checkpoint_name

    model, target_layer, err_reason, ckpt_filename = get_warm_classifier(
        checkpoint_path=resolved_ckpt_path,
        device=device,
    )

    if model is None or target_layer is None:
        return {
            "nodule_id": nodule_id,
            "study_id": study_id,
            "method": method,
            "plane": plane,
            "generation_status": "UNAVAILABLE",
            "status_reason": err_reason or "Trained model weights checkpoint missing or failed to load.",
            "expected_checkpoint": resolved_ckpt_path.name if resolved_ckpt_path else None,
            "actual_checkpoint": ckpt_filename,
            "is_placeholder": True,
            "heatmap_matrix": None,
            "checkpoint": ckpt_filename,
            "attribution_label": ATTRIBUTION_DISCLAIMER,
            "disclaimer": "Attribution is unavailable because trained model weights could not be loaded.",
        }

    # 3. Extract 32? ROI crop centered on candidate
    cz, cy, cx = int(round(center_zyx[0])), int(round(center_zyx[1])), int(round(center_zyx[2]))
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

    # 4. Predict probabilities & determine target class
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    p_non_nodule = float(probs[0])
    p_nodule = float(probs[1])
    predicted_class = int(np.argmax(probs))  # 0: non-nodule, 1: nodule
    cand_prob = float(stored_prob) if stored_prob is not None else p_nodule

    if target_class_override is not None:
        target_class = target_class_override
        target_class_label = "Class 1 (nodule)" if target_class == 1 else "Class 0 (non-nodule)"
        explanation_note = f"Explaining class {target_class} ({target_class_label})"
    elif predicted_class == 1:
        target_class = 1
        target_class_label = "Nodule (predicted class)"
        explanation_note = f"Explaining predicted class 'nodule' (confidence: {p_nodule:.1%})"
    else:
        target_class = 1
        target_class_label = "Evidence for nodule class (prediction: non-nodule)"
        explanation_note = f"Candidate predicted as non-nodule ({p_non_nodule:.1%}). Map displays positive visual evidence toward the 'nodule' class (nodule probability: {cand_prob:.1%})."

    # 5. Compute attribution
    if method.lower() in ("gradcampp3d", "gradcam++"):
        cam = GradCAMPP3D(model, target_layer=target_layer)
    else:
        cam = GradCAM3D(model, target_layer=target_layer)

    heatmap_3d = cam.generate(tensor, target_class=target_class)

    # 6. Map from 3D to 2D view plane
    if plane.lower() == "axial":
        if slice_index is not None:
            dz_rel = int(slice_index) - cz + 16
            if 0 <= dz_rel < 32 and heatmap_3d[dz_rel].max() > 0:
                heatmap_2d = heatmap_3d[dz_rel, :, :]
            else:
                heatmap_2d = GradCAM3D.project_axial(heatmap_3d)
        else:
            heatmap_2d = GradCAM3D.project_axial(heatmap_3d)
    elif plane.lower() == "coronal":
        heatmap_2d = GradCAM3D.project_coronal(heatmap_3d)
    elif plane.lower() == "sagittal":
        heatmap_2d = GradCAM3D.project_sagittal(heatmap_3d)
    else:
        heatmap_2d = GradCAM3D.project_axial(heatmap_3d)

    # 7. Check for flat or zero gradient signal
    raw_max = float(heatmap_2d.max())
    if raw_max <= 1e-7:
        # Never add an artificial patch. Return NO_ATTRIBUTION_SIGNAL without heatmap
        result = {
            "nodule_id": nodule_id,
            "study_id": study_id,
            "method": method,
            "plane": plane,
            "generation_status": "NO_ATTRIBUTION_SIGNAL",
            "status_reason": "Zero or flat gradient attribution signal across feature maps for this candidate.",
            "expected_checkpoint": resolved_ckpt_path.name if resolved_ckpt_path else None,
            "actual_checkpoint": ckpt_filename,
            "is_placeholder": False,
            "heatmap_matrix": None,
            "checkpoint": ckpt_filename,
            "attribution_label": ATTRIBUTION_DISCLAIMER,
            "target_class": target_class,
            "target_class_label": target_class_label,
            "predicted_class": predicted_class,
            "predicted_class_name": "nodule" if predicted_class == 1 else "non-nodule",
            "candidate_probability": round(cand_prob, 4),
            "nodule_probability": round(p_nodule, 4),
            "non_nodule_probability": round(p_non_nodule, 4),
            "explanation_note": explanation_note,
            "disclaimer": "Attribution signal is zero or flat; no salient features were activated.",
        }
        XAI_HEATMAP_CACHE[cache_key] = result
        return result

    # 8. Standardize matrix to 64x64 grid for UI overlay
    t_h2d = torch.from_numpy(heatmap_2d).unsqueeze(0).unsqueeze(0).float()
    h64 = F.interpolate(t_h2d, size=(64, 64), mode="bilinear", align_corners=False)[0, 0].numpy()
    h64 = h64 / h64.max()

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
        "checkpoint": ckpt_filename,
        "expected_checkpoint": resolved_ckpt_path.name if resolved_ckpt_path else None,
        "actual_checkpoint": ckpt_filename,
        "centroid_zyx": [cz, cy, cx],
        "crop_bbox_zyx": [[z1, z2], [y1, y2], [x1, x2]],
        "spacing_mm": list(spacing),
        "slice_index": slice_index,
        "target_class": target_class,
        "target_class_label": target_class_label,
        "predicted_class": predicted_class,
        "predicted_class_name": "nodule" if predicted_class == 1 else "non-nodule",
        "candidate_probability": round(cand_prob, 4),
        "nodule_probability": round(p_nodule, 4),
        "non_nodule_probability": round(p_non_nodule, 4),
        "explanation_note": explanation_note,
        "execution_time_ms": round(elapsed_ms, 1),
        "disclaimer": "Attribution is an AI visual explanation of model feature activations, not diagnostic confirmation.",
    }

    XAI_HEATMAP_CACHE[cache_key] = result
    log.info("on_demand_xai_computed", study_id=study_id, nodule_id=nodule_id, method=method, checkpoint=ckpt_filename, elapsed_ms=round(elapsed_ms, 1))
    return result
