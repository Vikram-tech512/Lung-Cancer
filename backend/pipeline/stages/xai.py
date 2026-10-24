"""
backend/pipeline/stages/xai.py
================================
Stage 7 — Explainability (XAI) Stage.

Runs all three XAI methods in sequence:
  1. 3D Grad-CAM            → volumetric attribution heatmap
  2. Integrated Gradients   → axiomatic pixel attributions
  3. Attention Rollout      → multi-head attention propagation (Swin3D only)

Reads:
    ctx.get(PREPROCESSING)  → {"volume": np.ndarray}
    ctx.get(CLASSIFICATION) → model result dicts

Writes:
    ctx.set(XAI, {
        "gradcam3d":  {"heatmap_shape": [...], "top_slices": [...], "saved_npy": "..."},
        "ig":         {"attributions_shape": [...], "top_voxels": [...]},
        "attention":  {"rollout_shape": [...]} | {"status": "not_available"},
        "summary":    {"most_activated_region": "upper_right_lobe", ...},
    })
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.monitoring import get_logger
from backend.pipeline.orchestrator import PipelineContext, PipelineStage

log = get_logger(__name__)

XAI_OUTPUT_DIR = Path(os.getenv("XAI_OUTPUT_DIR", "./results/xai"))


class XAIStage:
    """Runs all three XAI methods and persists heatmap arrays to disk."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Grad-CAM 3D
    # ------------------------------------------------------------------
    def _run_gradcam(
        self, volume: np.ndarray, ctx: PipelineContext
    ) -> Dict[str, Any]:
        try:
            import torch
            from backend.explainability.gradcam3d import GradCAM3D
            from backend.models.classification.resnet3d.model import ResNet3D50

            cls_out = ctx.get(PipelineStage.CLASSIFICATION) or {}
            model = cls_out.get("model")
            
            # If no active model was stored in context, attempt to instantiate ResNet3D50
            if model is None:
                ckpt_path = Path("checkpoints_3d/resnet3d_luna16_best.pt")
                if ckpt_path.exists():
                    model = ResNet3D50(in_channels=1, num_classes=2)
                    ckpt = torch.load(str(ckpt_path), map_location=self.device)
                    if "model_state_dict" in ckpt:
                        model.load_state_dict(ckpt["model_state_dict"])
                    model.to(self.device).eval()
            
            if model is None or not hasattr(model, "layer4"):
                return {
                    "status": "unavailable",
                    "detail": "GradCAM3D requires a loaded 3D CNN model with convolutional layers. Checkpoint not loaded."
                }

            target_layer = model.layer4[-1].conv3 if hasattr(model.layer4[-1], "conv3") else model.layer4[-1]
            gcam = GradCAM3D(model, target_layer=target_layer)

            # Ensure 5D tensor shape (1, 1, D, H, W)
            vol_tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            heatmap = gcam.generate(vol_tensor, target_class=1)
            if heatmap is None:
                raise ValueError("GradCAM returned None")
            
            # Persist to disk
            XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            npy_path = XAI_OUTPUT_DIR / f"{ctx.case_id}_gradcam.npy"
            np.save(str(npy_path), heatmap)
            
            # Top activated slices (axial plane)
            axial_scores = heatmap.mean(axis=(1, 2))
            top_slices = np.argsort(axial_scores)[-5:][::-1].tolist()
            return {
                "status": "computed",
                "heatmap_shape": list(heatmap.shape),
                "top_slices":    top_slices,
                "saved_npy":     str(npy_path),
                "max_activation": float(heatmap.max()),
                "disclaimer":    "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE"
            }
        except Exception as exc:
            log.warning("gradcam_failed", error=str(exc))
            return {"status": "error", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------
    def _run_ig(
        self, volume: np.ndarray, ctx: PipelineContext
    ) -> Dict[str, Any]:
        try:
            import torch
            from backend.explainability.integrated_gradients import IntegratedGradients3D
            from backend.models.classification.resnet3d.model import ResNet3D50

            cls_out = ctx.get(PipelineStage.CLASSIFICATION) or {}
            model = cls_out.get("model")
            if model is None:
                ckpt_path = Path("checkpoints_3d/resnet3d_luna16_best.pt")
                if ckpt_path.exists():
                    model = ResNet3D50(in_channels=1, num_classes=2)
                    ckpt = torch.load(str(ckpt_path), map_location=self.device)
                    if "model_state_dict" in ckpt:
                        model.load_state_dict(ckpt["model_state_dict"])
                    model.to(self.device).eval()

            if model is None:
                return {
                    "status": "unavailable",
                    "detail": "IntegratedGradients3D requires a loaded model. Checkpoint not loaded."
                }

            ig = IntegratedGradients3D(model=model, steps=20)
            vol_tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            res = ig.attribute(vol_tensor, target_class=1)
            attrs = res["attributions_3d"]

            if attrs is None:
                raise ValueError("IG returned None")
            
            # Top activated voxel coordinates
            flat_idx = np.argsort(np.abs(attrs).ravel())[-20:][::-1]
            coords = np.unravel_index(flat_idx, attrs.shape)
            top_voxels = [
                {"z": int(z), "y": int(y), "x": int(x), "attr": float(attrs[z, y, x])}
                for z, y, x in zip(*coords)
            ]
            return {
                "status": "computed",
                "attributions_shape": list(attrs.shape),
                "top_voxels": top_voxels,
                "l1_total": float(np.abs(attrs).sum()),
                "completeness_delta": float(res.get("completeness_delta", 0.0)),
                "disclaimer": "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE"
            }
        except Exception as exc:
            log.warning("ig_failed", error=str(exc))
            return {"status": "error", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Attention Rollout (Swin3D only)
    # ------------------------------------------------------------------
    def _run_attention(
        self, volume: np.ndarray, ctx: PipelineContext
    ) -> Dict[str, Any]:
        try:
            from backend.explainability.attention_viz3d import AttentionRollout3D
            
            cls_out = ctx.get(PipelineStage.CLASSIFICATION) or {}
            attention_matrices = cls_out.get("swin_attention_matrices")
            if not attention_matrices:
                return {
                    "status": "unavailable",
                    "detail": "Attention Rollout requires multi-layer attention matrices from a 3D Swin Transformer forward pass."
                }

            rollout_engine = AttentionRollout3D(discard_ratio=0.8, head_fusion="mean")
            rollout = rollout_engine.compute_rollout(attention_matrices)
            return {
                "status": "computed",
                "rollout_shape": list(rollout.shape),
                "mean_attention": float(rollout.mean()),
                "disclaimer": "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE"
            }
        except Exception as exc:
            log.warning("attention_rollout_failed", error=str(exc))
            return {"status": "unavailable", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------------
    @staticmethod
    def _anatomical_region(top_slice: int, total_slices: int) -> str:
        """Crude anatomical axial zone localization from axial slice index."""
        if total_slices <= 1:
            return "unavailable_2d_slice"
        ratio = top_slice / max(total_slices, 1)
        if ratio < 0.33:
            return "upper_lung_zone"
        elif ratio < 0.66:
            return "mid_lung_zone"
        else:
            return "lower_lung_zone"

    # ------------------------------------------------------------------
    # Stage entrypoint
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        prep = ctx.get(PipelineStage.PREPROCESSING)
        volume = prep.get("volume") if prep else None
        if volume is None:
            log.warning("xai_no_volume", case_id=ctx.case_id)
            return {"status": "no_volume"}

        gradcam_result = self._run_gradcam(volume, ctx)
        ig_result      = self._run_ig(volume, ctx)
        attn_result    = self._run_attention(volume, ctx)

        # Anatomical summary
        top_slice = (
            gradcam_result["top_slices"][0]
            if gradcam_result.get("top_slices")
            else volume.shape[0] // 2
        )
        region = self._anatomical_region(top_slice, volume.shape[0])

        result = {
            "gradcam3d": gradcam_result,
            "ig":        ig_result,
            "attention": attn_result,
            "summary": {
                "most_activated_region": region,
                "most_activated_axial_slice": top_slice,
            },
        }
        log.info(
            "xai_done",
            case_id=ctx.case_id,
            region=region,
            gradcam_ok="heatmap_shape" in gradcam_result,
            ig_ok="attributions_shape" in ig_result,
        )
        return result
