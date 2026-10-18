"""
backend/pipeline/stages/detection.py
=======================================
Stage 3 — 3D Nodule Detection via RetinaNet3D.

Reads:
    ctx.get(PREPROCESSING) → {"volume": np.ndarray, "spacing": tuple}

Writes:
    ctx.set(DETECTION, {
        "candidates": [
            {"bbox_zyx": [z1,y1,x1,z2,y2,x2], "score": 0.93, "diameter_mm": 8.4},
            ...
        ],
        "nodule_count": int,
    })
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

from backend.monitoring import get_logger, MetricsRegistry
from backend.pipeline.orchestrator import PipelineContext, PipelineStage

if TYPE_CHECKING:
    import torch

log = get_logger(__name__)


class DetectionStage:
    """
    Wraps the RetinaNet3D detector for pipeline use.

    The model is loaded lazily on first `run()` call so the stage
    can be imported without triggering GPU allocation.
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        device: str = "cpu",
        score_threshold: float = 0.35,
        nms_iou_threshold: float = 0.20,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lazy model loader
    # ------------------------------------------------------------------
    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from backend.models.detection.retinanet3d import RetinaNet3D
            ckpt_path = self.checkpoint_dir / "retinanet3d_best.pt"
            model = RetinaNet3D(num_classes=2, backbone_depth=50)
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(state.get("model_state_dict", state))
                log.info("detection_model_loaded", checkpoint=str(ckpt_path))
            else:
                log.warning("detection_checkpoint_missing", path=str(ckpt_path))
            model = model.to(self.device).eval()
            self._model = model
        except Exception as exc:
            log.warning("detection_model_load_failed", error=str(exc))
            self._model = None
        return self._model

    # ------------------------------------------------------------------
    # Stage entrypoint
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        prep = ctx.get(PipelineStage.PREPROCESSING)
        if prep is None:
            ctx.warnings.append("DetectionStage: preprocessing output missing, skipping")
            return {"candidates": [], "nodule_count": 0}

        volume: np.ndarray = prep.get("volume")
        spacing: tuple = prep.get("spacing", (1.0, 1.0, 1.0))

        if volume is None:
            return {"candidates": [], "nodule_count": 0}

        model = self._load_model()
        candidates = []

        with MetricsRegistry.time_inference("RetinaNet3D"):
            if model is not None:
                candidates = self._run_model_inference(model, volume, spacing)
            else:
                candidates = self._heuristic_fallback(volume, spacing)

        log.info(
            "detection_done",
            case_id=ctx.case_id,
            nodule_count=len(candidates),
        )
        return {
            "candidates": candidates,
            "nodule_count": len(candidates),
            "spacing_mm": list(spacing),
        }

    def _run_model_inference(
        self,
        model: Any,
        volume: np.ndarray,
        spacing: tuple,
    ) -> list:
        import torch
        with torch.no_grad():
            tensor = (
                torch.from_numpy(volume)
                .float()
                .unsqueeze(0)   # add batch dim
                .unsqueeze(0)   # add channel dim
                .to(self.device)
            )
            # RetinaNet3D forward: returns list of dicts with "boxes" and "scores"
            predictions = model(tensor)

        candidates = []
        for pred in predictions:
            boxes  = pred.get("boxes", [])
            scores = pred.get("scores", [])
            if hasattr(boxes, "cpu"):
                boxes  = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
            for box, score in zip(boxes, scores):
                if float(score) < self.score_threshold:
                    continue
                z1, y1, x1, z2, y2, x2 = box[:6]
                dz = (z2 - z1) * spacing[0]
                dy = (y2 - y1) * spacing[1]
                dx = (x2 - x1) * spacing[2]
                diameter_mm = float(np.cbrt(dz * dy * dx))
                candidates.append({
                    "bbox_zyx": [float(z1), float(y1), float(x1),
                                  float(z2), float(y2), float(x2)],
                    "score": float(score),
                    "diameter_mm": round(diameter_mm, 2),
                })
        return candidates

    def _heuristic_fallback(
        self, volume: np.ndarray, spacing: tuple
    ) -> list:
        """
        Simple intensity-threshold blob finder when model weights are absent.
        Used for smoke-testing and development without GPU / checkpoints.
        """
        from scipy import ndimage
        # Threshold in HU range typical of solid nodules
        mask = (volume > -200) & (volume < 200)
        labeled, n = ndimage.label(mask)
        candidates = []
        for i in range(1, min(n + 1, 20)):
            blob = np.argwhere(labeled == i)
            if len(blob) < 8:
                continue
            z1, y1, x1 = blob.min(axis=0)
            z2, y2, x2 = blob.max(axis=0)
            dz = (z2 - z1) * spacing[0]
            dy = (y2 - y1) * spacing[1]
            dx = (x2 - x1) * spacing[2]
            diameter_mm = float(np.cbrt(max(dz * dy * dx, 1)))
            if 3 < diameter_mm < 30:
                candidates.append({
                    "bbox_zyx": [float(z1), float(y1), float(x1),
                                  float(z2), float(y2), float(x2)],
                    "score": 0.50,
                    "diameter_mm": round(diameter_mm, 2),
                    "source": "heuristic_fallback",
                })
        return sorted(candidates, key=lambda c: -c["diameter_mm"])[:10]
