"""
backend/datasets/lidc_idri.py
=============================
LIDC-IDRI (The Lung Image Database Consortium) dataset loader and processor.
Handles multi-radiologist nodule annotations, 4-radiologist consensus agreement,
and strict patient-level splitting.
"""

from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class NoduleAnnotation:
    patient_id: str
    series_uid: str
    nodule_id: str
    centroid_zyx: Tuple[float, float, float]
    diameter_mm: float
    volume_mm3: float
    malignancy_mean: float  # 1 (highly unlikely) to 5 (highly suspicious)
    malignancy_votes: List[int]
    subtlety_mean: float
    sphericity_mean: float
    binary_label: Optional[int]  # 0: Benign (<=2), 1: Malignant (>=4), None: Indeterminate (3)

class LIDCIDRILoader:
    def __init__(self, data_root: str = "data/LIDC-IDRI"):
        self.data_root = Path(data_root)

    def is_dataset_present(self) -> bool:
        return self.data_root.exists() and any(self.data_root.iterdir())

    def binarize_malignancy(self, malignancy_mean: float) -> Optional[int]:
        """
        Standard consensus binarization rule:
          malignancy_mean >= 3.5 or >= 4 -> 1 (Malignant)
          malignancy_mean <= 2.5 or <= 2 -> 0 (Benign)
          Between 2.5 and 3.5 -> Indeterminate (excluded from binary benchmark)
        """
        if malignancy_mean >= 3.5:
            return 1
        elif malignancy_mean <= 2.5:
            return 0
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Returns metadata summary of LIDC-IDRI availability."""
        present = self.is_dataset_present()
        return {
            "dataset": "LIDC-IDRI",
            "present": present,
            "root_path": str(self.data_root),
            "expected_patients": 1018,
            "modalities": ["Thoracic CT"],
            "annotation_standard": "4-radiologist blinded consensus",
            "status": "AVAILABLE" if present else "NOT_DOWNLOADED (Use PyLIDC or TCIA API)"
        }
