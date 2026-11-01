"""
backend/pipeline/stages/classification.py
==========================================
Stage 4 — Multi-model 3D Classification.

Reads:
    ctx.get(PREPROCESSING) → {"volume": np.ndarray}
    ctx.get(SEGMENTATION)  → {"nodule_mask": np.ndarray}  (optional)

Writes:
    ctx.set(CLASSIFICATION, {
        "ResNet3D":       {"nodule_prob": 0.82, "non_nodule_prob": 0.18, "logit": 1.51},
        "DenseNet3D":     {"nodule_prob": 0.79, ...},
        "SwinTransformer3D": {...},
        "consensus_prob": 0.80,
        "consensus_label": None,
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

# Models are enabled only when a trained checkpoint is available.
_DEFAULT_MODELS = ["ResNet3D", "DenseNet3D", "SwinTransformer3D"]


def infer_nodule_crop(
    crop_volume: np.ndarray,
    model: Optional[Any] = None,
    device: str = "cpu",
    checkpoint_dir: Path = Path("checkpoints_3d"),
    input_size: tuple = (32, 32, 32),
) -> Dict[str, float]:
    """
    Estimates the candidate nodule class on a 3D crop volume of size 32x32x32.
    Resizes if crop dimensions differ.
    """
    import torch
    import torch.nn.functional as F
    from skimage.transform import resize as sk_resize

    if model is None:
        from backend.models.classification.resnet3d.model import resnet3d_50, ResNet3D18Half
        ckpt = checkpoint_dir / "resnet3d_luna16_best.pt"
        if not ckpt.exists():
            ckpt = Path("checkpoints_3d/resnet3d_luna16_best.pt")
        if not ckpt.exists():
            ckpt = checkpoint_dir / "resnet3d_best.pt"

        if ckpt.exists():
            state = torch.load(ckpt, map_location=device)
            sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
            arch = state.get("architecture") if isinstance(state, dict) else None
            if arch == "ResNet3D18Half" or (isinstance(sd, dict) and "fc.weight" in sd and sd["fc.weight"].shape == torch.Size([2, 256])):
                model = ResNet3D18Half(in_channels=1, num_classes=2)
            else:
                model = resnet3d_50(num_classes=2)
            model.load_state_dict(sd, strict=False)
        else:
            raise RuntimeError("A trained ResNet3D checkpoint is required for inference")
        model = model.to(device).eval()

    if crop_volume.shape != input_size:
        vol = sk_resize(
            crop_volume,
            input_size,
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)
    else:
        vol = crop_volume.astype(np.float32)

    vol = np.clip(vol, -1000, 400)
    vol = (vol + 1000) / 1400.0

    tensor = (
        torch.from_numpy(vol)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    p_pos = float(probs[1])
    p_neg = float(probs[0])
    logit_val = float(logits[0, 1].item())

    return {
        "nodule_prob": p_pos,
        "non_nodule_prob": p_neg,
        "probability_type": "uncalibrated nodule-candidate score",
        "logit": logit_val,
    }


class ClassificationStage:
    """
    Runs all three 3D classification heads and fuses their probabilities
    using a simple weighted average ensemble at the classification stage
    (distinct from the full multimodal ensemble in Stage 6).
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints_3d"),
        device: str = "cpu",
        enabled_models: Optional[list] = None,
        input_size: tuple = (32, 32, 32),
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
            from backend.models.classification.resnet3d.model import resnet3d_50, ResNet3D18Half
            ckpt = self.checkpoint_dir / "resnet3d_best.pt"
            if not ckpt.exists():
                ckpt = self.checkpoint_dir / "resnet3d_luna16_best.pt"
            if not ckpt.exists():
                ckpt = Path("checkpoints_3d/resnet3d_luna16_best.pt")

            if not ckpt.exists():
                return None

            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device)
                sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
                arch = state.get("architecture") if isinstance(state, dict) else None
                if arch == "ResNet3D18Half" or (isinstance(sd, dict) and "fc.weight" in sd and sd["fc.weight"].shape == torch.Size([2, 256])):
                    model = ResNet3D18Half(in_channels=1, num_classes=2)
                else:
                    model = resnet3d_50(num_classes=2)
                model.load_state_dict(sd, strict=False)
                log.info("loaded_resnet3d", checkpoint=str(ckpt), arch=model.__class__.__name__)
            return model.to(self.device).eval()
        except Exception as exc:
            log.warning("resnet3d_load_failed", error=str(exc))
            return None

    def _load_densenet3d(self) -> Optional[Any]:
        try:
            import torch
            from backend.models.classification.densenet3d.model import densenet3d_121
            ckpt = self.checkpoint_dir / "densenet3d_best.pt"
            if not ckpt.exists():
                return None
            model = densenet3d_121(num_classes=2)
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
            ckpt = self.checkpoint_dir / "swin3d_best.pt"
            if not ckpt.exists():
                return None
            model = SwinTransformer3D(num_classes=2)
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
        p_pos = float(probs[1])
        p_neg = float(probs[0])
        return {
            "nodule_prob":    p_pos,
            "non_nodule_prob": p_neg,
            "probability_type": "uncalibrated nodule-candidate score",
            "logit":          float(logits[0, 1].item()),
        }

    # ------------------------------------------------------------------
    # Stage entrypoint
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        prep = ctx.get(PipelineStage.PREPROCESSING)
        volume = prep.get("volume") if prep else None
        if volume is None:
            ctx.warnings.append("ClassificationStage: input volume is not available")
            return {
                "status": "not_available",
                "models": {},
                "consensus_prob": None,
                "consensus_label": None,
            }

        self._ensure_models()
        if not self._models:
            return {
                "status": "not_available",
                "models": {},
                "consensus_prob": None,
                "consensus_label": None,
                "message": "No trained classification checkpoint is available.",
            }
        tensor = self._prepare_tensor(volume)

        results: Dict[str, Any] = {}
        probs_list = []

        for model_name in self.enabled_models:
            if model_name not in self._models:
                continue
            pred = self._infer_one(model_name, tensor)
            results[model_name] = pred
            probs_list.append(pred["nodule_prob"])

            MetricsRegistry.record_inference(
                model=model_name,
                outcome="positive_candidate" if pred["nodule_prob"] >= pred["non_nodule_prob"] else "negative_candidate",
                latency_ms=0,
            )

        # Consensus
        consensus_prob = float(np.mean(probs_list))
        results["consensus_prob"]  = round(consensus_prob, 4)
        results["consensus_label"] = None
        results["probability_type"] = "uncalibrated nodule-candidate score"
        results["status"] = "available"

        log.info(
            "classification_done",
            case_id=ctx.case_id,
            consensus_prob=results["consensus_prob"],
            status=results["status"],
        )
        return results
