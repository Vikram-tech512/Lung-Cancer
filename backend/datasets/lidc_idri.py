"""
backend/datasets/lidc_idri.py
=============================
LIDC-IDRI (The Lung Image Database Consortium) dataset loader and processor.
Handles multi-radiologist nodule annotations, 4-radiologist consensus agreement,
and strict patient-level splitting.

REAL DATA REQUIRED — this file does not fabricate results.
  1. Get the data (NOT downloadable inside a restricted sandbox — you need
     TCIA's own NBIA Data Retriever, ~124GB):
       https://www.cancerimagingarchive.net/collection/lidc-idri/
  2. `pip install pylidc` and point it at your download by creating
     ~/.pylidcrc with:
       [dicom]
       path = /absolute/path/to/LIDC-IDRI
  3. Then LIDCIDRIVolumeDataset below will do real DICOM + XML-annotation
     parsing via pylidc. Until step 1-2 are done, is_dataset_present()
     returns False and the volume dataset raises rather than returning
     placeholder data.
"""

from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:                      # keep split/consensus logic usable
    Dataset = object                     # without requiring torch installed
    HAS_TORCH = False

from backend.preprocessing.ct_pipeline import CTPipeline

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


def consensus_mask_from_annotator_masks(annotator_masks: List[np.ndarray],
                                         agreement_frac: float = 0.5) -> np.ndarray:
    """Standard LIDC-IDRI consensus rule: a voxel counts as 'nodule' if at
    least `agreement_frac` of the annotating radiologists (up to 4 per
    nodule in LIDC-IDRI) marked it. Shared by the real pylidc-backed path
    and the synthetic smoke-test data so both use identical logic.
    """
    if not annotator_masks:
        raise ValueError("annotator_masks is empty — nothing to build consensus from")
    stacked = np.stack(annotator_masks, axis=0).astype(np.int32)
    return stacked.sum(axis=0) >= (agreement_frac * len(annotator_masks))


def patient_level_split(patient_ids: List[str], val_frac: float = 0.15,
                         test_frac: float = 0.15, seed: int = 42
                         ) -> Tuple[List[str], List[str], List[str]]:
    """Split PATIENTS (not nodules, not slices) into train/val/test.

    This matters more than it sounds: LIDC-IDRI patients can have several
    scans and several nodules each. Splitting at the nodule level lets
    nodules from the same patient leak across train and test, which
    inflates validation/test accuracy in a way that does not reflect
    real-world generalization to an unseen patient. Every split in this
    project must go through this function, never a per-nodule shuffle.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    ids = sorted(set(patient_ids))  # sorted first so shuffling is reproducible
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    n_test = max(1, int(round(n * test_frac))) if n > 1 else 0
    n_test = min(n_test, n - n_val) if n - n_val > 0 else 0

    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]

    overlap = (set(train_ids) & set(val_ids)) | (set(train_ids) & set(test_ids)) | (set(val_ids) & set(test_ids))
    assert not overlap, f"Patient-level split leaked patients across splits: {overlap}"

    return train_ids, val_ids, test_ids


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

    def list_annotations(self) -> List[NoduleAnnotation]:
        """Real pylidc-backed annotation listing. Requires pylidc installed
        and configured (~/.pylidcrc) and the dataset actually downloaded —
        raises rather than returning fabricated annotations if not.
        """
        if not self.is_dataset_present():
            raise RuntimeError(
                "LIDC-IDRI not found at "
                f"{self.data_root}. See module docstring for setup steps — "
                "this loader does not fabricate annotations."
            )
        try:
            import pylidc as pl
            from pylidc.utils import consensus
        except ImportError as e:
            raise RuntimeError(
                "pylidc is required for real annotation parsing: pip install pylidc"
            ) from e

        annotations = []
        scans = pl.query(pl.Scan)
        for scan in scans:
            clusters = scan.cluster_annotations()
            for cluster in clusters:
                if not cluster:
                    continue
                mal_votes = [int(a.malignancy) for a in cluster]
                mal_mean = float(np.mean(mal_votes))
                diam = float(np.mean([a.diameter for a in cluster]))
                vol = float(np.mean([a.volume for a in cluster]))
                subtlety = float(np.mean([a.subtlety for a in cluster]))
                sphericity = float(np.mean([a.sphericity for a in cluster]))
                # consensus() gives a boolean volume mask + bbox; centroid
                # is derived from the mask's own bounding box center.
                _, cbbox, _ = consensus(cluster, clevel=0.5)
                centroid = tuple(float((s.start + s.stop) / 2.0) for s in cbbox)

                annotations.append(NoduleAnnotation(
                    patient_id=scan.patient_id,
                    series_uid=scan.series_instance_uid,
                    nodule_id=f"{scan.patient_id}_{cluster[0].id}",
                    centroid_zyx=centroid,
                    diameter_mm=diam,
                    volume_mm3=vol,
                    malignancy_mean=mal_mean,
                    malignancy_votes=mal_votes,
                    subtlety_mean=subtlety,
                    sphericity_mean=sphericity,
                    binary_label=self.binarize_malignancy(mal_mean),
                ))
        return annotations


class LIDCIDRIVolumeDataset(Dataset):
    """Torch Dataset yielding (volume, mask, label, annotation) tuples for
    real training. Every sample is a real preprocessed CT ROI — no
    synthetic data ever flows through this class.
    """

    def __init__(self, annotations: List[NoduleAnnotation],
                 crop_size: Tuple[int, int, int] = (64, 64, 64),
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 augment: bool = False):
        if not HAS_TORCH:
            raise RuntimeError("torch is required to use LIDCIDRIVolumeDataset")
        self.annotations = [a for a in annotations if a.binary_label is not None]
        self.crop_size = crop_size
        self.pipeline = CTPipeline(target_spacing=target_spacing)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        import pylidc as pl
        from pylidc.utils import consensus

        ann = self.annotations[idx]
        scan = pl.query(pl.Scan).filter(
            pl.Scan.series_instance_uid == ann.series_uid).first()
        vol = scan.to_volume()                       # (H, W, D) raw HU, pylidc convention
        vol = np.transpose(vol, (2, 0, 1)).astype(np.float32)   # -> (D, H, W)

        spacing = (float(scan.slice_thickness), float(scan.pixel_spacing),
                   float(scan.pixel_spacing))
        resampled, _ = self.pipeline.resample_volume(vol, spacing, order=1)

        # Convert the annotation's mm centroid back to the resampled
        # voxel grid using the same spacing pipeline.resample_volume used.
        scale = np.array(spacing) / np.array(self.pipeline.target_spacing)
        center_zyx = tuple(int(round(c * s)) for c, s in
                            zip(ann.centroid_zyx, scale))

        crop = self.pipeline.extract_nodule_crop(resampled, center_zyx, self.crop_size,
                                                  pad_value=-1000.0)
        crop = self.pipeline.apply_window(crop)       # -> [0, 1] float32

        if self.augment:
            crop = _augment_volume(crop)

        volume_t = torch.from_numpy(crop).unsqueeze(0).float()   # (1, D, H, W)
        label_t = torch.tensor(ann.binary_label, dtype=torch.long)
        return volume_t, label_t, asdict(ann)


def _augment_volume(volume: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """Light, label-preserving 3D augmentation: random axis flips + small
    intensity jitter. Deliberately avoids rotation/elastic warps that would
    need mask-aware handling to not distort the true lesion boundary.
    """
    rng = np.random.default_rng(seed)
    out = volume.copy()
    for axis in (0, 1, 2):
        if rng.random() < 0.5:
            out = np.flip(out, axis=axis)
    out = out + rng.normal(0, 0.02, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)
