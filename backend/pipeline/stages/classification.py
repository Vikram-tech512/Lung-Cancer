"""
backend/pipeline/stages/classification.py
==========================================
Stage 4 — Multi-model 3D Classification.

Reads:
    ctx.get(PREPROCESSING) → {"volume": np.ndarray}
    ctx.get(SEGMENTATION)  → {"nodule_mask": np.ndarray}  (optional)

Writes:
    ctx.set(CLASSIFICATION, {
        "ResNet3D":       {"malignant_prob": 0.82, "benign_prob": 0.18, "logit": 1.51},
        "DenseNet3D":     {"malignant_prob": 0.79, ...},
        "SwinTransformer3D": {...},
        "consensus_prob": 0.80,
        "consensus_label": "malignant",
    })
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.monitoring import get_logger, MetricsRegistry
from backend.pipeline.orchestrator import PipelineContext, PipelineStage

log = get_logger(__name__)

# Models enabled by default (all if checkpoint exists, else fallback)
_DEFAULT_MODELS = ["ResNet3D", "DenseNet3D", "SwinTransformer3D"]


class ClassificationStage:
    """
    Runs all three 3D classification heads and fuses their probabilities
    using a simple weighted average ensemble at the classification stage
    (distinct from the full multimodal ensemble in Stage 6).
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        device: str = "cpu",
        enabled_models: Optional[list] = None,
        input_size: tuple = (96, 96, 96),
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.enabled_models = enabled_models or _DEFAULT_MODELS
        self.input_size = input_size
        self._models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------
    def _load_resnet3d(self) -> Optional[Any]:
        try:
            import torch
            from backend.models.classification.resnet3d.model import resnet3d_50
            model = resnet3d_50(num_classes=2)
            ckpt = self.checkpoint_dir / "resnet3d_best.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
                log.info("loaded_resnet3d", checkpoint=str(ckpt))
            return model.to(self.device).eval()
        except Exception as exc:
            log.warning("resnet3d_load_failed", error=str(exc))
            return None

    def _load_densenet3d(self) -> Optional[Any]:
        try:
            import torch
            from backend.models.classification.densenet3d.model import densenet3d_121
            model = densenet3d_121(num_classes=2)
            ckpt = self.checkpoint_dir / "densenet3d_best.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
                log.info("loaded_densenet3d", checkpoint=str(ckpt))
            return model.to(self.device).eval()
        except Exception as exc:
            log.warning("densenet3d_load_failed", error=str(exc))
            return None

    def _load_swin3d(self) -> Optional[Any]:
        try:
            import torch
            from backend.models.classification.swin3d.model import SwinTransformer3D
            model = SwinTransformer3D(num_classes=2)
            ckpt = self.checkpoint_dir / "swin3d_best.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device)
                model.load_state_dict(state.get("model_state_dict", state), strict=False)
                log.info("loaded_swin3d", checkpoint=str(ckpt))
            return model.to(self.device).eval()
        except Exception as exc:
            log.warning("swin3d_load_failed", error=str(exc))
            return None

    def _ensure_models(self) -> None:
        if not self._models:
            loaders = {
                "ResNet3D":          self._load_resnet3d,
                "DenseNet3D":        self._load_densenet3d,
                "SwinTransformer3D": self._load_swin3d,
            }
            for name in self.enabled_models:
                if name in loaders:
                    m = loaders[name]()
                    if m is not None:
                        self._models[name] = m

    # ------------------------------------------------------------------
    # Preprocessing helper
    # ------------------------------------------------------------------
    def _prepare_tensor(self, volume: np.ndarray):
        """Resize to input_size, normalise, add batch+channel dims."""
        import torch
        from skimage.transform import resize as sk_resize
        vol = sk_resize(
            volume,
            self.input_size,
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)
        # Clip & normalise to [0, 1]
        vol = np.clip(vol, -1000, 400)
        vol = (vol + 1000) / 1400.0
        return (
            torch.from_numpy(vol)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _infer_one(self, model_name: str, tensor) -> Dict[str, float]:
        import torch
        import torch.nn.functional as F
        model = self._models[model_name]
        with MetricsRegistry.time_inference(model_name):
            with torch.no_grad():
                logits = model(tensor)          # [1, 2]
                probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return {
            "malignant_prob": float(probs[1]),
            "benign_prob":    float(probs[0]),
            "logit":          float(logits[0, 1].item()),
        }

    def _mock_prediction(self) -> Dict[str, float]:
        """Returned when no model weights are available (dev/CI mode)."""
        return {
            "malignant_prob": 0.50,
            "benign_prob":    0.50,
            "logit":          0.0,
            "source":         "mock_no_checkpoint",
        }

    # ------------------------------------------------------------------
    # Stage entrypoint
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        prep = ctx.get(PipelineStage.PREPROCESSING)
        if prep is None:
            ctx.warnings.append("ClassificationStage: preprocessing missing, using zeros")
            volume = np.zeros((128, 128, 128), dtype=np.float32)
        else:
            volume = prep.get("volume", np.zeros((128, 128, 128), dtype=np.float32))

        self._ensure_models()
        tensor = self._prepare_tensor(volume)

        results: Dict[str, Any] = {}
        probs_list = []

        for model_name in self.enabled_models:
            if model_name in self._models:
                pred = self._infer_one(model_name, tensor)
            else:
                pred = self._mock_prediction()
            results[model_name] = pred
            probs_list.append(pred["malignant_prob"])

            MetricsRegistry.record_inference(
                model=model_name,
                outcome="malignant" if pred["malignant_prob"] > 0.5 else "benign",
                latency_ms=0,
            )

        # Consensus
        consensus_prob = float(np.mean(probs_list)) if probs_list else 0.5
        results["consensus_prob"]  = round(consensus_prob, 4)
        results["consensus_label"] = "malignant" if consensus_prob > 0.5 else "benign"

        log.info(
            "classification_done",
            case_id=ctx.case_id,
            consensus_prob=results["consensus_prob"],
            consensus_label=results["consensus_label"],
        )
        return results
