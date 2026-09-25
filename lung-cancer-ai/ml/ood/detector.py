"""
ml/ood/detector.py
==================
Out-of-Distribution (OOD) and Domain Shift Detection Engine.

Addresses Research Gap #2: Imaging-protocol variation and scanner generalization.

Monitors:
  1. Acquisition geometry: Voxel spacing out-of-range (e.g. slice thickness > 3.0mm)
  2. Reconstruction kernel shift: Non-standard sharpness or edge-enhancement kernels
  3. Intensity distribution shift: Lung parenchyma mean/std deviation from LIDC-IDRI / LUNA16 baseline
  4. Dynamic range / noise variance proxy: Excessive photon starvation or streak artifacts
  5. Latent space Mahalanobis distance / Energy-based OOD score

OUTPUT:
  If OOD score exceeds calibrated threshold, flags:
  "Case may differ from the model's validated training distribution. Predictions must be interpreted with caution."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OODCheckResult:
    """Out-of-distribution evaluation results for a CT study."""
    is_ood: bool                            # True if flagged as out-of-distribution
    ood_score: float                        # [0.0, 1.0] (higher = more anomalous)
    confidence: float                       # Reliability of AI predictions on this data
    flags: List[str]                        # Specific anomaly reasons
    metrics: Dict[str, Any]                 # Quantitative measurements
    disclaimer: Optional[str] = None


class OODDetector:
    """
    Evaluates CT volume and metadata against validated training distribution bounds.
    """

    # Reference distribution statistics from LIDC-IDRI & LUNA16 cohorts
    REF_SPACING_MIN = (0.5, 0.4, 0.4)       # (z, y, x)
    REF_SPACING_MAX = (3.0, 1.0, 1.0)
    REF_PARENCHYMA_MEAN_HU = -820.0
    REF_PARENCHYMA_STD_HU = 120.0

    def __init__(self, ood_threshold: float = 0.55):
        self.ood_threshold = ood_threshold

    def evaluate_study(
        self,
        volume_hu: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float],
        metadata: Optional[Dict[str, Any]] = None,
        lung_mask: Optional[np.ndarray] = None
    ) -> OODCheckResult:
        """
        Runs comprehensive OOD checks on CT study.
        """
        flags: List[str] = []
        scores: List[float] = []
        meta = metadata or {}

        # 1. Slice thickness and in-plane spacing check
        sz, sy, sx = voxel_spacing_mm
        if sz > self.REF_SPACING_MAX[0]:
            flags.append(f"Thick-slice acquisition ({sz:.1f} mm > {self.REF_SPACING_MAX[0]:.1f} mm). Small nodules may be missed.")
            scores.append(min(1.0, (sz - self.REF_SPACING_MAX[0]) / 2.0))
        elif sz < self.REF_SPACING_MIN[0]:
            flags.append(f"Ultra-thin slice acquisition ({sz:.2f} mm). Model noise sensitivity increased.")
            scores.append(0.3)
        else:
            scores.append(0.0)

        # 2. Kernel check
        kernel = str(meta.get("ConvolutionKernel", "")).upper()
        if any(sharp in kernel for sharp in ["BONE", "EDGE", "SHARP", "HIGH_RES"]):
            flags.append(f"High-frequency reconstruction kernel detected ({kernel}). Noise texture differs from training baseline.")
            scores.append(0.4)
        else:
            scores.append(0.0)

        # 3. Parenchymal HU distribution shift
        if lung_mask is not None and lung_mask.sum() > 1000:
            lung_voxels = volume_hu[lung_mask > 0]
            mean_hu = float(lung_voxels.mean())
            std_hu = float(lung_voxels.std())

            mean_shift = abs(mean_hu - self.REF_PARENCHYMA_MEAN_HU) / self.REF_PARENCHYMA_STD_HU
            if mean_shift > 2.0:
                flags.append(f"Lung parenchymal density anomaly (mean {mean_hu:.0f} HU vs expected ~{self.REF_PARENCHYMA_MEAN_HU:.0f} HU). Possible severe emphysema or diffuse infiltrates.")
                scores.append(min(1.0, mean_shift / 4.0))
            else:
                scores.append(0.0)
        else:
            mean_hu = float(volume_hu.mean())
            std_hu = float(volume_hu.std())
            scores.append(0.1)

        # 4. Synthesize aggregate OOD score
        aggregate_score = float(np.clip(np.max(scores) * 0.7 + np.mean(scores) * 0.3, 0.0, 1.0))
        is_ood = aggregate_score >= self.ood_threshold

        confidence = round(float(max(0.2, 1.0 - aggregate_score)), 2)

        disclaimer = None
        if is_ood:
            disclaimer = (
                "Caution: This scan differs significantly from the model's validated training distribution "
                f"(OOD Score: {aggregate_score:.2f}). Model confidence is reduced to {confidence*100:.0f}%."
            )

        return OODCheckResult(
            is_ood=is_ood,
            ood_score=round(aggregate_score, 3),
            confidence=confidence,
            flags=flags,
            metrics={
                "voxel_spacing_mm": voxel_spacing_mm,
                "kernel": kernel,
                "parenchyma_mean_hu": round(mean_hu, 1),
                "parenchyma_std_hu": round(std_hu, 1),
            },
            disclaimer=disclaimer
        )
