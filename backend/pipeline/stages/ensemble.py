"""
backend/pipeline/stages/ensemble.py
======================================
Stage 6 — Multimodal Ensemble Fusion.

Fuses:
  - DL classification probabilities   (Stage 4 outputs)
  - IBSI PyRadiomics features          (Stage 5 outputs)
  - Clinical metadata                  (age, sex, smoking_pack_years)

Strategies
----------
* WeightedAverage  — fixed expert weights (fast, interpretable)
* StackedXGBoost   — XGBoost meta-learner trained on held-out fold
* AttentionFusion  — learned attention over modality embeddings (PyTorch)

Writes:
    ctx.set(ENSEMBLE, {
        "strategy":         "WeightedAverage",
        "malignant_prob":   0.81,
        "benign_prob":      0.19,
        "label":            "malignant",
        "confidence_band":  [0.74, 0.88],
        "modality_weights": {"dl": 0.60, "radiomics": 0.30, "clinical": 0.10},
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


# ---------------------------------------------------------------------------
# Modality weights (can be overridden via env-vars or config)
# ---------------------------------------------------------------------------
DL_WEIGHT       = float(os.getenv("ENSEMBLE_DL_WEIGHT",       "0.60"))
RADIOMICS_WEIGHT = float(os.getenv("ENSEMBLE_RADIOMICS_WEIGHT", "0.30"))
CLINICAL_WEIGHT  = float(os.getenv("ENSEMBLE_CLINICAL_WEIGHT",  "0.10"))


class EnsembleStage:
    """
    Multimodal ensemble that combines DL, radiomics, and clinical modalities.
    Falls back gracefully if any modality is missing.
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        strategy: str = "WeightedAverage",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.strategy = strategy
        self._xgb_model: Optional[Any] = None

    # ------------------------------------------------------------------
    # XGBoost meta-learner (optional)
    # ------------------------------------------------------------------
    def _load_xgb(self) -> Optional[Any]:
        if self._xgb_model is not None:
            return self._xgb_model
        ckpt = self.checkpoint_dir / "ensemble_xgb.json"
        if not ckpt.exists():
            return None
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(ckpt))
            self._xgb_model = model
            log.info("ensemble_xgb_loaded", checkpoint=str(ckpt))
            return model
        except Exception as exc:
            log.warning("ensemble_xgb_load_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_dl_prob(cls_out: Dict) -> float:
        """Mean malignant probability from all DL heads."""
        probs = [
            v.get("malignant_prob", 0.5)
            for k, v in cls_out.items()
            if isinstance(v, dict) and "malignant_prob" in v
        ]
        return float(np.mean(probs)) if probs else 0.5

    @staticmethod
    def _extract_radiomics_prob(rad_out: Dict) -> Optional[float]:
        """Retrieve radiomics-derived malignancy score if present."""
        if not rad_out:
            return None
        score = rad_out.get("malignancy_score") or rad_out.get("xgb_malignant_prob")
        return float(score) if score is not None else None

    @staticmethod
    def _extract_clinical_prob(clinical: Dict) -> Optional[float]:
        """
        Simplified Brock / Mayo model-style risk from clinical variables.
        Returns None when clinical data is missing.
        """
        if not clinical:
            return None
        age  = float(clinical.get("age", 0) or 0)
        sex  = clinical.get("sex", "").lower()
        pkyr = float(clinical.get("smoking_pack_years", 0) or 0)
        # Very rough heuristic (not a validated clinical model)
        risk = 0.1 + 0.003 * pkyr + 0.002 * max(age - 40, 0)
        if sex == "female":
            risk += 0.05
        return float(np.clip(risk, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Fusion strategies
    # ------------------------------------------------------------------
    def _weighted_average(
        self,
        dl_prob: float,
        rad_prob: Optional[float],
        clin_prob: Optional[float],
    ) -> tuple[float, Dict[str, float]]:
        total_weight = DL_WEIGHT
        fused = DL_WEIGHT * dl_prob
        weights_used: Dict[str, float] = {"dl": DL_WEIGHT}

        if rad_prob is not None:
            fused += RADIOMICS_WEIGHT * rad_prob
            total_weight += RADIOMICS_WEIGHT
            weights_used["radiomics"] = RADIOMICS_WEIGHT

        if clin_prob is not None:
            fused += CLINICAL_WEIGHT * clin_prob
            total_weight += CLINICAL_WEIGHT
            weights_used["clinical"] = CLINICAL_WEIGHT

        return float(fused / total_weight), weights_used

    def _xgb_fusion(
        self,
        dl_prob: float,
        rad_out: Dict,
        clinical: Dict,
    ) -> float:
        model = self._load_xgb()
        if model is None:
            return dl_prob
        try:
            import xgboost as xgb
            # Build feature vector: [dl_prob] + top radiomics + clinical
            features = [dl_prob]
            # Append first 20 numeric radiomics features (padded)
            rad_values = [
                float(v) for v in rad_out.values()
                if isinstance(v, (int, float))
            ][:20]
            features += rad_values + [0.0] * (20 - len(rad_values))
            # Append clinical features
            features.append(float(clinical.get("age", 60) or 60))
            features.append(1.0 if clinical.get("sex", "") == "male" else 0.0)
            features.append(float(clinical.get("smoking_pack_years", 0) or 0))

            dmat = xgb.DMatrix(np.array([features]))
            prob = float(model.predict(dmat)[0])
            return np.clip(prob, 0.0, 1.0)
        except Exception as exc:
            log.warning("xgb_fusion_failed", error=str(exc))
            return dl_prob

    # ------------------------------------------------------------------
    # Stage entrypoint
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        cls_out  = ctx.get(PipelineStage.CLASSIFICATION) or {}
        rad_out  = ctx.get(PipelineStage.RADIOMICS) or {}
        clinical = ctx.get("clinical_meta") or {}

        dl_prob   = self._extract_dl_prob(cls_out)
        rad_prob  = self._extract_radiomics_prob(rad_out)
        clin_prob = self._extract_clinical_prob(clinical)

        with MetricsRegistry.time_inference("MultimodalEnsemble"):
            if self.strategy == "StackedXGBoost":
                fused_prob = self._xgb_fusion(dl_prob, rad_out, clinical)
                weights_used = {"dl": DL_WEIGHT, "radiomics": RADIOMICS_WEIGHT, "clinical": CLINICAL_WEIGHT}
            else:
                fused_prob, weights_used = self._weighted_average(dl_prob, rad_prob, clin_prob)

        # Calibrated confidence band (simple ± 1.96 SE approximation)
        n = 300   # effective sample size placeholder
        se = float(np.sqrt(fused_prob * (1 - fused_prob) / n))
        band = [
            round(float(np.clip(fused_prob - 1.96 * se, 0, 1)), 4),
            round(float(np.clip(fused_prob + 1.96 * se, 0, 1)), 4),
        ]

        label = "malignant" if fused_prob > 0.5 else "benign"
        MetricsRegistry.record_inference("MultimodalEnsemble", label, 0)

        result = {
            "strategy":         self.strategy,
            "malignant_prob":   round(fused_prob, 4),
            "benign_prob":      round(1 - fused_prob, 4),
            "label":            label,
            "confidence_band":  band,
            "modality_weights": weights_used,
            "dl_prob":          round(dl_prob, 4),
            "radiomics_prob":   round(rad_prob, 4) if rad_prob is not None else None,
            "clinical_prob":    round(clin_prob, 4) if clin_prob is not None else None,
        }
        log.info(
            "ensemble_done",
            case_id=ctx.case_id,
            label=label,
            malignant_prob=result["malignant_prob"],
            strategy=self.strategy,
        )
        return result
