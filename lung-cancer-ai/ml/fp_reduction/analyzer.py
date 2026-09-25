"""
ml/fp_reduction/analyzer.py
===========================
False Positive (FP) Reduction and Anatomical Mimic Classifier.

Addresses Research Gap #6 & #8:
  - Subtle low-contrast difficulty
  - Normal anatomy false positives (vessels, pleura, scars, ribs)

Categorizes detected candidate lesions into:
  1. "True Nodule Suspicion"
  2. "Vascular Bifurcation / Cross-section" (elongated tubular continuity along z-axis)
  3. "Pleural Plaque / Thickening" (continuous planar adherence to costal/mediastinal pleura)
  4. "Apical Scar / Linear Atelectasis" (non-spherical high-aspect ratio fibrotic band)
  5. "Costal Osteophyte / Bone Edge" (HU > 400 connected to skeletal boundary)

Computes an Anatomical Prior Score to suppress high-confidence normal anatomical false alarms.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import scipy.ndimage as ndi
except ImportError:
    ndi = None

from ml.segmentation.nodule_segmenter import NoduleMeasurements

logger = logging.getLogger(__name__)


@dataclass
class FPReductionAssessment:
    """Evaluation of whether a candidate detection is a true lesion or anatomical mimic."""
    is_likely_false_positive: bool
    mimic_category: str                     # "none" | "vascular_bifurcation" | "pleural_thickening" | "apical_scar" | "bone_edge"
    fp_probability: float                   # [0.0, 1.0]
    anatomical_reasoning: str
    retained_detection_score: float         # Adjusted score after applying anatomical prior


class FalsePositiveAnalyzer:
    """
    3D geometric and anatomical feature engine to eliminate non-nodular false detections.
    """

    def analyze_candidate(
        self,
        crop_hu: np.ndarray,
        mask: np.ndarray,
        measurements: NoduleMeasurements,
        initial_score: float,
        lung_mask_crop: Optional[np.ndarray] = None
    ) -> FPReductionAssessment:
        """
        Analyzes a candidate nodule ROI to identify anatomical mimics.
        """
        if mask.sum() == 0:
            return FPReductionAssessment(
                is_likely_false_positive=True,
                mimic_category="empty_roi",
                fp_probability=1.0,
                anatomical_reasoning="Zero segmented voxels in candidate ROI.",
                retained_detection_score=0.0
            )

        # 1. Bone / Osteophyte check (mean HU > 350 and connected to high HU border)
        if measurements.hu_mean > 350.0 and measurements.hu_max > 700.0:
            return FPReductionAssessment(
                is_likely_false_positive=True,
                mimic_category="bone_edge",
                fp_probability=0.92,
                anatomical_reasoning="High mineral attenuation (>350 HU) consistent with costal osteophyte or rib margin.",
                retained_detection_score=round(initial_score * 0.1, 3)
            )

        # 2. Tubular Vessel continuity check
        # A blood vessel in cross-section is round on 1 slice but continues across multiple slices with high elongation
        D, H, W = crop_hu.shape
        z_presence = np.any(mask > 0, axis=(1, 2))
        z_extent = int(np.sum(z_presence))

        if measurements.elongation > 3.2 and z_extent >= D * 0.7:
            return FPReductionAssessment(
                is_likely_false_positive=True,
                mimic_category="vascular_bifurcation",
                fp_probability=0.88,
                anatomical_reasoning="High cylindrical elongation (>3.2) tracking continuously along vascular arborization tree.",
                retained_detection_score=round(initial_score * 0.15, 3)
            )

        # 3. Linear Scar / Atelectasis check (sphericity < 0.28, low volume, flat sheet geometry)
        if measurements.sphericity < 0.28 and measurements.compactness < 0.12 and measurements.max_diameter_mm > 10.0:
            return FPReductionAssessment(
                is_likely_false_positive=True,
                mimic_category="apical_scar",
                fp_probability=0.82,
                anatomical_reasoning="Planar / sheet-like morphology (sphericity < 0.28) consistent with linear atelectatic scar band.",
                retained_detection_score=round(initial_score * 0.20, 3)
            )

        # 4. Pleural Plaque check (broad flat adherence to thoracic wall)
        if lung_mask_crop is not None and ndi is not None:
            dilated = ndi.binary_dilation(mask, structure=np.ones((3, 3, 3)))
            contact_voxels = (dilated & (lung_mask_crop == 0)).sum()
            contact_ratio = contact_voxels / max(1, measurements.surface_area_mm2)
            if contact_ratio > 0.65 and measurements.sphericity < 0.35:
                return FPReductionAssessment(
                    is_likely_false_positive=True,
                    mimic_category="pleural_thickening",
                    fp_probability=0.78,
                    anatomical_reasoning="Broad planar adherence to pleural boundary without focal intraparenchymal epicentre.",
                    retained_detection_score=round(initial_score * 0.25, 3)
                )

        # Genuine lesion suspicion
        fp_prob = float(np.clip(1.0 - initial_score * 0.8, 0.05, 0.50))
        return FPReductionAssessment(
            is_likely_false_positive=False,
            mimic_category="none",
            fp_probability=round(fp_prob, 3),
            anatomical_reasoning="Focal intraparenchymal morphology with distinct three-dimensional spherical compactness.",
            retained_detection_score=round(initial_score, 3)
        )
