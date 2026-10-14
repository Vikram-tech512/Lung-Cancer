"""
backend/datasets/synthetic_volumes.py
======================================
SYNTHETIC data for pipeline smoke-testing ONLY.

Nothing in this file touches a real patient scan. It exists to answer one
question: "does the preprocessing/splitting/training code run correctly
end-to-end and behave the way it should", which can and should be checked
before ever pointing the pipeline at real LIDC-IDRI data.

Per the project's own stated principle: never let a placeholder result be
mistaken for a real one. Every function/class here is prefixed `synthetic_`
or `Synthetic*`, and every metric this module can help verify (split
correctness, mask/IoU plumbing, preprocessing shape/range invariants) is
about CODE correctness, not model accuracy — it says nothing about how
well a model will actually classify real nodules.

No torch dependency here on purpose, so these checks can run anywhere
numpy/scipy are available, including environments without a GPU or torch
installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.datasets.lidc_idri import consensus_mask_from_annotator_masks


@dataclass
class SyntheticNodule:
    patient_id: str
    volume: np.ndarray            # (D, H, W) float32, HU-like values
    consensus_mask: np.ndarray    # (D, H, W) bool — ground truth for testing
    annotator_masks: List[np.ndarray]  # simulate 1-4 LIDC-style radiologists
    centroid_zyx: Tuple[int, int, int]
    diameter_voxels: float
    binary_label: int              # 0 benign, 1 malignant


def _sphere_mask(shape: Tuple[int, int, int], center: Tuple[int, int, int],
                  radius: float, jitter_seed: Optional[int] = None,
                  jitter_scale: float = 0.0) -> np.ndarray:
    """A (possibly per-voxel jittered) solid sphere mask — stands in for one
    radiologist's contour. Real LIDC contours are irregular polygons, not
    spheres; jitter_scale > 0 gives just enough inter-annotator disagreement
    to exercise the consensus-mask logic without pretending to be a
    realistic nodule shape.
    """
    zz, yy, xx = np.indices(shape)
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 +
                   (xx - center[2]) ** 2)
    eff_radius = radius
    if jitter_scale > 0:
        rng = np.random.default_rng(jitter_seed)
        eff_radius = radius * (1.0 + rng.uniform(-jitter_scale, jitter_scale))
    return dist <= eff_radius


def make_synthetic_nodule(patient_id: str, volume_shape: Tuple[int, int, int] = (96, 96, 96),
                           malignant: bool = True, seed: int = 0,
                           n_annotators: int = 4) -> SyntheticNodule:
    """Build one synthetic CT-like volume with a lesion and simulated
    multi-radiologist contours (for consensus-mask testing), entirely
    with numpy so this needs no imaging libraries or real data.

    HU-like background: lung parenchyma ~ -800 HU with noise; lesion
    region raised toward soft-tissue HU (malignant lesions get a slightly
    higher mean HU and more irregular boundary jitter than benign, purely
    so the two classes are numerically distinguishable for pipeline
    testing — this is NOT a claim about real HU/malignancy correlation).
    """
    rng = np.random.default_rng(seed)
    D, H, W = volume_shape
    volume = -800.0 + rng.normal(0, 25.0, size=volume_shape).astype(np.float32)

    center = (D // 2 + int(rng.integers(-5, 6)),
              H // 2 + int(rng.integers(-5, 6)),
              W // 2 + int(rng.integers(-5, 6)))
    radius = float(rng.uniform(6, 12))
    lesion_hu = rng.uniform(20, 60) if malignant else rng.uniform(-40, 10)
    jitter = 0.25 if malignant else 0.1  # more irregular boundary -> malignant, for test purposes only

    consensus_core = _sphere_mask(volume_shape, center, radius)
    volume[consensus_core] += lesion_hu

    annotator_masks = [
        _sphere_mask(volume_shape, center, radius, jitter_seed=seed * 100 + i,
                     jitter_scale=jitter)
        for i in range(n_annotators)
    ]
    # LIDC-style 50% agreement consensus: voxel counted if >= half of
    # annotators marked it. Same function the real pylidc-backed loader
    # uses, so a smoke test here actually exercises production logic.
    consensus_mask = consensus_mask_from_annotator_masks(annotator_masks, agreement_frac=0.5)

    return SyntheticNodule(
        patient_id=patient_id,
        volume=volume,
        consensus_mask=consensus_mask,
        annotator_masks=annotator_masks,
        centroid_zyx=center,
        diameter_voxels=radius * 2,
        binary_label=1 if malignant else 0,
    )


def make_synthetic_cohort(n_patients: int = 12, seed: int = 42) -> List[SyntheticNodule]:
    """A small synthetic 'cohort' — enough patients to exercise a
    patient-level train/val/test split without any real data.
    """
    rng = np.random.default_rng(seed)
    cohort = []
    for i in range(n_patients):
        malignant = bool(rng.integers(0, 2))
        nodule = make_synthetic_nodule(
            patient_id=f"SYNTH-{i:03d}", malignant=malignant,
            seed=seed * 1000 + i,
        )
        cohort.append(nodule)
    return cohort
