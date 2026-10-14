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
            from backend.explainability.gradcam3d import GradCAM3D
            # Attempt to use the first available classification model
            cls_out = ctx.get(PipelineStage.CLASSIFICATION) or {}
            # GradCAM3D needs a model instance — use dummy if models not loaded
            gcam = GradCAM3D()
            heatmap = gcam.generate(volume)
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
                "heatmap_shape": list(heatmap.shape),
                "top_slices":    top_slices,
                "saved_npy":     str(npy_path),
                "max_activation": float(heatmap.max()),
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
            from backend.explainability.integrated_gradients import IntegratedGradients3D
            ig = IntegratedGradients3D(n_steps=50, device=self.device)
            attrs = ig.attribute(volume)
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
                "attributions_shape": list(attrs.shape),
                "top_voxels": top_voxels,
                "l1_total": float(np.abs(attrs).sum()),
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
            from backend.explainability.attention_viz3d import AttentionViz3D
            viz = AttentionViz3D(device=self.device)
            rollout = viz.compute_rollout(volume)
            if rollout is None:
                return {"status": "not_available"}
            return {
                "rollout_shape": list(rollout.shape),
                "mean_attention": float(rollout.mean()),
            }
        except Exception as exc:
            log.warning("attention_rollout_failed", error=str(exc))
            return {"status": "not_available", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------------
    @staticmethod
    def _anatomical_region(top_slice: int, total_slices: int) -> str:
        """Crude anatomical localisation from axial slice index."""
        ratio = top_slice / max(total_slices, 1)
        if ratio < 0.33:
            return "upper_lobe"
        elif ratio < 0.66:
            return "middle_lobe"
        else:
            return "lower_lobe"

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
