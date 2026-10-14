"""
backend/datasets/luna16.py
==========================
Dataset loader for LUNA16 (LUng Nodule Analysis 2016).

LUNA16 is freely downloadable (~120 GB total, but subsets are usable) and
is the standard public benchmark for 3D pulmonary nodule detection &
classification. It is derived from LIDC-IDRI with:
  - 10 cross-validation folds (pre-defined patient-level splits)
  - A curated candidate list (candidates_V2.csv) with radiologist-consensus labels
  - Raw DICOM replaced by MetaImage (.mhd/.raw) for simpler I/O

HOW TO DOWNLOAD (Free, Kaggle):
  Option A — Kaggle (recommended, no TCIA account needed):
    kagglehub.dataset_download("fanbyprinciple/luna-lung-cancer")
    OR via CLI:
      kaggle datasets download -d fanbyprinciple/luna-lung-cancer

  Option B — Official (full dataset, 10 subsets, ~120 GB):
    https://zenodo.org/record/3723295  (subsets 0-9)
    Download at least subset0 to start (~12 GB).

FOLDER STRUCTURE EXPECTED:
  <luna16_root>/
    subset0/ ... subset9/     # .mhd + .raw files (one pair per CT scan)
    candidates_V2.csv         # or candidates.csv  — nodule candidate coords
    annotations.csv           # confirmed nodule coords + diameter

LABELS:
  candidates_V2.csv: class=1 (nodule, malignant/benign combined) or 0 (non-nodule)
  annotations.csv:   lists confirmed nodule centroids + diameter_mm (all positive)

This module provides:
  - LUNA16Dataset  : torch.utils.data.Dataset yielding (volume_crop, label, meta)
  - get_luna16_loaders() : factory for train/val/test DataLoaders

Integrates with existing:
  - backend.preprocessing.ct_pipeline.CTPipeline  (HU windowing, resampling)
  - backend.models.classification.resnet3d.model.ResNet3D50
  - backend.models.classification.densenet3d.model.DenseNet3D121
  - backend.models.classification.swin3d.model.SwinTransformer3D
"""

from __future__ import annotations

import csv
import os
import glob
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    Dataset = object
    HAS_TORCH = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

from backend.preprocessing.ct_pipeline import CTPipeline


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LUNA16Candidate:
    """One row from candidates_V2.csv."""
    series_uid:    str
    coord_x:       float   # mm, world space
    coord_y:       float
    coord_z:       float
    label:         int     # 1 = confirmed nodule, 0 = non-nodule
    diameter_mm:   float   # 0 if not an annotated nodule
    subset_id:     int     # 0-9 (used for cross-validation folds)


# ─────────────────────────────────────────────────────────────────────────────
#  CSV parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_candidates_csv(csv_path: Path) -> List[Dict]:
    """Parse candidates_V2.csv → list of dicts."""
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "series_uid": row["seriesuid"],
                "coord_x":    float(row["coordX"]),
                "coord_y":    float(row["coordY"]),
                "coord_z":    float(row["coordZ"]),
                "label":      int(row["class"]),
            })
    return records


def _parse_annotations_csv(csv_path: Path) -> Dict[str, List[Dict]]:
    """Parse annotations.csv → {series_uid: [{coord_x, coord_y, coord_z, diameter_mm}]}"""
    by_uid: Dict[str, List[Dict]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["seriesuid"]
            entry = {
                "coord_x":     float(row["coordX"]),
                "coord_y":     float(row["coordY"]),
                "coord_z":     float(row["coordZ"]),
                "diameter_mm": float(row["diameter_mm"]),
            }
            by_uid.setdefault(uid, []).append(entry)
    return by_uid


def _find_mhd_file(luna_root: Path, series_uid: str) -> Optional[Path]:
    """Search all subset directories for <series_uid>.mhd."""
    for subset_dir in sorted(luna_root.glob("subset*")):
        mhd = subset_dir / f"{series_uid}.mhd"
        if mhd.exists():
            return mhd
    return None


def _series_uid_to_subset(luna_root: Path, series_uid: str) -> int:
    """Determine which subset (0-9) a series belongs to."""
    for subset_dir in sorted(luna_root.glob("subset*")):
        mhd = subset_dir / f"{series_uid}.mhd"
        if mhd.exists():
            try:
                return int(subset_dir.name.replace("subset", ""))
            except ValueError:
                return -1
    return -1


# ─────────────────────────────────────────────────────────────────────────────
#  Loader class  (reads .mhd/.raw via SimpleITK)
# ─────────────────────────────────────────────────────────────────────────────

class LUNA16Loader:
    """
    High-level helper for loading LUNA16 metadata and checking availability.
    """

    def __init__(self, luna_root: str | Path):
        self.root = Path(luna_root)

    def is_available(self) -> bool:
        has_subsets  = any(self.root.glob("subset*"))
        has_csv      = (
            (self.root / "candidates_V2.csv").exists() or
            (self.root / "candidates.csv").exists()
        )
        return has_subsets and has_csv

    def candidates_csv(self) -> Path:
        for name in ("candidates_V2.csv", "candidates.csv"):
            p = self.root / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No candidates CSV found in {self.root}. "
            "Expected candidates_V2.csv or candidates.csv."
        )

    def annotations_csv(self) -> Optional[Path]:
        p = self.root / "annotations.csv"
        return p if p.exists() else None

    def summary(self) -> Dict:
        available = self.is_available()
        n_mhd = len(list(self.root.rglob("*.mhd"))) if available else 0
        return {
            "dataset":         "LUNA16",
            "root":            str(self.root),
            "available":       available,
            "n_ct_scans":      n_mhd,
            "expected_scans":  888,
            "download_url":    "https://zenodo.org/record/3723295",
            "kaggle_dataset":  "fanbyprinciple/luna-lung-cancer",
        }

    def load_candidates(
        self,
        max_neg_per_pos: int = 10,
        min_diameter_mm: float = 0.0,
        seed: int = 42,
    ) -> List[LUNA16Candidate]:
        """
        Load & balance candidates.

        Args:
            max_neg_per_pos:  Maximum ratio of negative to positive samples.
                              LUNA16 has ~1,351 positives vs ~550K negatives;
                              without balancing the negatives dominate.
            min_diameter_mm:  Skip confirmed nodules smaller than this.
            seed:             For reproducible negative sub-sampling.
        """
        if not HAS_SITK:
            raise RuntimeError(
                "SimpleITK required: pip install SimpleITK"
            )

        cand_path = self.candidates_csv()
        ann_path  = self.annotations_csv()

        raw = _parse_candidates_csv(cand_path)
        ann_by_uid = _parse_annotations_csv(ann_path) if ann_path else {}

        # Enrich positives with diameter from annotations.csv
        def _diameter_for(uid, cx, cy, cz) -> float:
            for a in ann_by_uid.get(uid, []):
                dist = ((a["coord_x"] - cx)**2 +
                        (a["coord_y"] - cy)**2 +
                        (a["coord_z"] - cz)**2) ** 0.5
                if dist < 5.0:   # within 5 mm → same nodule
                    return a["diameter_mm"]
            return 0.0

        positives = []
        negatives = []
        for r in raw:
            uid  = r["series_uid"]
            diam = _diameter_for(uid, r["coord_x"], r["coord_y"], r["coord_z"]) if r["label"] else 0.0
            if r["label"] == 1 and diam < min_diameter_mm:
                continue
            subset = _series_uid_to_subset(self.root, uid)
            if subset < 0:
                continue   # .mhd file not downloaded yet
            cand = LUNA16Candidate(
                series_uid  = uid,
                coord_x     = r["coord_x"],
                coord_y     = r["coord_y"],
                coord_z     = r["coord_z"],
                label       = r["label"],
                diameter_mm = diam,
                subset_id   = subset,
            )
            (positives if r["label"] == 1 else negatives).append(cand)

        # Under-sample negatives
        rng = random.Random(seed)
        max_neg = len(positives) * max_neg_per_pos
        if len(negatives) > max_neg:
            negatives = rng.sample(negatives, max_neg)

        all_candidates = positives + negatives
        rng.shuffle(all_candidates)

        print(
            f"[LUNA16] Loaded {len(positives)} positives + {len(negatives)} negatives "
            f"= {len(all_candidates)} total candidates"
        )
        return all_candidates

    def patient_level_split(
        self,
        candidates: List[LUNA16Candidate],
        val_frac:  float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[LUNA16Candidate], List[LUNA16Candidate], List[LUNA16Candidate]]:
        """
        Split at the SCAN level (not candidate level) to prevent leakage.
        Uses LUNA16's pre-defined subset IDs to avoid same-patient leakage.
        """
        series_ids = sorted({c.series_uid for c in candidates})
        rng = random.Random(seed)
        rng.shuffle(series_ids)

        n = len(series_ids)
        n_val  = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))

        test_ids  = set(series_ids[:n_test])
        val_ids   = set(series_ids[n_test:n_test + n_val])
        train_ids = set(series_ids[n_test + n_val:])

        train = [c for c in candidates if c.series_uid in train_ids]
        val   = [c for c in candidates if c.series_uid in val_ids]
        test  = [c for c in candidates if c.series_uid in test_ids]

        print(
            f"[LUNA16 Split] train: {len(train)} | val: {len(val)} | test: {len(test)} candidates"
        )
        return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
#  3D Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _augment_3d(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Label-preserving 3D augmentation:
      - Random axis flips (x3)
      - 90° rotations in the axial plane
      - Gaussian intensity noise
      - Random intensity scaling (simulates scanner variability)
    """
    out = volume.copy()

    # Random flips
    for axis in range(3):
        if rng.random() < 0.5:
            out = np.flip(out, axis=axis)

    # Random 90° rotation in axial plane (axes 1,2)
    k = rng.integers(0, 4)
    if k > 0:
        out = np.rot90(out, k=k, axes=(1, 2))

    # Gaussian noise
    noise_sigma = rng.uniform(0.0, 0.02)
    out = out + rng.normal(0, noise_sigma, size=out.shape).astype(np.float32)

    # Intensity scaling (±10%)
    scale = rng.uniform(0.9, 1.1)
    out = out * scale

    return np.clip(out.astype(np.float32), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  torch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LUNA16Dataset(Dataset):
    """
    PyTorch Dataset for LUNA16 3D nodule crops.

    Each item yields:
        volume  : torch.Tensor  shape (1, D, H, W)  float32, values in [0,1]
        label   : torch.Tensor  scalar long  (0=non-nodule, 1=nodule)
        meta    : dict          {series_uid, coord_xyz, diameter_mm, label}

    The volume is a cubic crop centred on the candidate coordinate,
    resampled to 1×1×1 mm isotropic spacing via CTPipeline.
    """

    def __init__(
        self,
        luna_root:    Path,
        candidates:   List[LUNA16Candidate],
        crop_size:    Tuple[int, int, int] = (64, 64, 64),
        augment:      bool = False,
        pipeline:     Optional[CTPipeline] = None,
        cache_dir:    Optional[Path] = None,
        seed:         int = 42,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required.")
        if not HAS_SITK:
            raise RuntimeError("SimpleITK required: pip install SimpleITK")

        self.luna_root  = Path(luna_root)
        self.candidates = candidates
        self.crop_size  = crop_size
        self.augment    = augment
        self.pipeline   = pipeline or CTPipeline(target_spacing=(1.0, 1.0, 1.0))
        self.cache_dir  = Path(cache_dir) if cache_dir else None
        self._rng_seed  = seed

        # Build a series_uid → .mhd path cache
        self._mhd_cache: Dict[str, Path] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_mhd_path(self, series_uid: str) -> Path:
        if series_uid not in self._mhd_cache:
            mhd = _find_mhd_file(self.luna_root, series_uid)
            if mhd is None:
                raise FileNotFoundError(
                    f"Cannot find {series_uid}.mhd in {self.luna_root}/subset*/\n"
                    "Make sure the subset containing this scan is downloaded."
                )
            self._mhd_cache[series_uid] = mhd
        return self._mhd_cache[series_uid]

    def _cache_key(self, cand: LUNA16Candidate) -> str:
        key = f"{cand.series_uid}_{cand.coord_x:.1f}_{cand.coord_y:.1f}_{cand.coord_z:.1f}"
        return hashlib.md5(key.encode()).hexdigest()

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        p = self.cache_dir / f"{key}.npy"
        if p.exists():
            return np.load(str(p))
        return None

    def _save_to_cache(self, key: str, crop: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        np.save(str(self.cache_dir / f"{key}.npy"), crop)

    def _world_to_voxel(
        self,
        image: "sitk.Image",
        coord_xyz_mm: Tuple[float, float, float],
    ) -> Tuple[int, int, int]:
        """Convert world (mm) coordinates to voxel indices (z, y, x)."""
        voxel_xyz = image.TransformPhysicalPointToIndex(coord_xyz_mm)
        # SimpleITK uses x,y,z; our pipeline uses z,y,x
        return (voxel_xyz[2], voxel_xyz[1], voxel_xyz[0])

    def _extract_crop(self, cand: LUNA16Candidate) -> np.ndarray:
        """Load CT, resample to 1mm iso, extract cubic crop. Uses disk cache."""
        key = self._cache_key(cand)
        cached = self._load_from_cache(key)
        if cached is not None:
            return cached

        mhd_path = self._get_mhd_path(cand.series_uid)

        # Load .mhd volume
        itk_img = sitk.ReadImage(str(mhd_path))
        spacing  = tuple(reversed(itk_img.GetSpacing()))   # (x,y,z) → (z,y,x)
        volume   = sitk.GetArrayFromImage(itk_img).astype(np.float32)  # (D,H,W)

        # Resample to isotropic 1mm
        resampled, _ = self.pipeline.resample_volume(volume, spacing, order=1)

        # World → voxel in resampled space
        scale = np.array(spacing) / np.array(self.pipeline.target_spacing)
        center_zyx = (
            int(round(cand.coord_z * scale[0])) if hasattr(cand, 'coord_z') else 0,
            int(round(cand.coord_y * scale[1])),
            int(round(cand.coord_x * scale[2])),
        )
        # Proper world→voxel: use itk TransformPhysicalPointToIndex
        itk_img_rs = sitk.GetImageFromArray(resampled)
        itk_img_rs.SetSpacing([1.0, 1.0, 1.0])
        orig_origin  = itk_img.GetOrigin()
        itk_img_rs.SetOrigin(orig_origin)
        itk_img_rs.SetDirection(itk_img.GetDirection())
        vxl = itk_img_rs.TransformPhysicalPointToIndex(
            (float(cand.coord_x), float(cand.coord_y), float(cand.coord_z))
        )
        center_zyx = (int(vxl[2]), int(vxl[1]), int(vxl[0]))

        crop = self.pipeline.extract_nodule_crop(
            resampled, center_zyx, self.crop_size, pad_value=-1000.0
        )
        crop = self.pipeline.apply_window(crop)   # → [0, 1] float32

        self._save_to_cache(key, crop)
        return crop

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int):
        cand = self.candidates[idx]
        crop = self._extract_crop(cand)

        if self.augment:
            rng = np.random.default_rng(self._rng_seed + idx)
            crop = _augment_3d(crop, rng)

        volume_t = torch.from_numpy(crop.copy()).unsqueeze(0).float()  # (1,D,H,W)
        label_t  = torch.tensor(cand.label, dtype=torch.long)
        meta = {
            "series_uid":  cand.series_uid,
            "coord_x":     cand.coord_x,
            "coord_y":     cand.coord_y,
            "coord_z":     cand.coord_z,
            "diameter_mm": cand.diameter_mm,
            "label":       cand.label,
            "subset_id":   cand.subset_id,
        }
        return volume_t, label_t, meta

    def class_weights(self) -> "torch.Tensor":
        """Inverse-frequency class weights for loss weighting."""
        labels = [c.label for c in self.candidates]
        counts = np.bincount(labels, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (2.0 * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def sampler_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        cw = self.class_weights().numpy()
        return [float(cw[c.label]) for c in self.candidates]


# ─────────────────────────────────────────────────────────────────────────────
#  Collate (handles variable-length meta dicts)
# ─────────────────────────────────────────────────────────────────────────────

def luna16_collate(batch):
    volumes  = torch.stack([b[0] for b in batch])
    labels   = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return volumes, labels, metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_luna16_loaders(
    luna_root:        str | Path,
    crop_size:        int   = 64,
    batch_size:       int   = 4,
    num_workers:      int   = 2,
    max_neg_per_pos:  int   = 5,
    min_diameter_mm:  float = 0.0,
    val_frac:         float = 0.15,
    test_frac:        float = 0.15,
    seed:             int   = 42,
    cache_dir:        Optional[str] = None,
    balanced_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Args:
        luna_root:        Path to LUNA16 root containing subset0..subsetN + CSV files.
        crop_size:        Side length of cubic 3D crop in voxels (default 64).
        batch_size:       Mini-batch size.
        num_workers:      DataLoader workers.
        max_neg_per_pos:  Negative/positive ratio cap.
        min_diameter_mm:  Skip positives smaller than this diameter.
        val_frac:         Validation fraction.
        test_frac:        Test fraction.
        seed:             Random seed.
        cache_dir:        Directory to cache extracted crops as .npy files.
                          Strongly recommended — avoids re-reading 12GB .mhd files.
        balanced_sampling: Use WeightedRandomSampler for balanced training batches.
    """
    root = Path(luna_root)
    loader = LUNA16Loader(root)

    if not loader.is_available():
        raise RuntimeError(
            f"LUNA16 not found at {root}.\n\n"
            "Download options:\n"
            "  Kaggle:   kaggle datasets download -d fanbyprinciple/luna-lung-cancer\n"
            "  Official: https://zenodo.org/record/3723295\n\n"
            "Or run with --dry-run to test the pipeline on synthetic data."
        )

    candidates = loader.load_candidates(
        max_neg_per_pos=max_neg_per_pos,
        min_diameter_mm=min_diameter_mm,
        seed=seed,
    )
    train_cands, val_cands, test_cands = loader.patient_level_split(
        candidates, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    cs = (crop_size, crop_size, crop_size)
    cache = Path(cache_dir) if cache_dir else None

    train_ds = LUNA16Dataset(root, train_cands, crop_size=cs, augment=True,  cache_dir=cache, seed=seed)
    val_ds   = LUNA16Dataset(root, val_cands,   crop_size=cs, augment=False, cache_dir=cache, seed=seed)
    test_ds  = LUNA16Dataset(root, test_cands,  crop_size=cs, augment=False, cache_dir=cache, seed=seed)

    if balanced_sampling:
        sw = train_ds.sampler_weights()
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            collate_fn=luna16_collate,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            collate_fn=luna16_collate,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=luna16_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=luna16_collate,
    )

    return train_loader, val_loader, test_loader
