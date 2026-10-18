"""
backend/training/smoke_test_pipeline.py
========================================
Runs the torch-independent parts of the training pipeline against SYNTHETIC
data and checks concrete invariants. This does not validate model accuracy
(there is no real model or real data involved) — it validates that the
preprocessing, splitting, and localization-metric CODE behaves correctly.

Run: python -m backend.training.smoke_test_pipeline
"""
from __future__ import annotations

import numpy as np

from backend.datasets.lidc_idri import patient_level_split, consensus_mask_from_annotator_masks
from backend.datasets.synthetic_volumes import make_synthetic_cohort
from backend.preprocessing.ct_pipeline import CTPipeline
from backend.explainability.localization_metrics import (
    heatmap_to_mask, dice_iou, pointing_game_hit, heatmap_centroid, evaluate_localization,
)


def check(label: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise AssertionError(label)


def main():
    print("=== 1. Patient-level split: no leakage, reproducible ===")
    patient_ids = [f"P{i:03d}" for i in range(40)]
    train, val, test = patient_level_split(patient_ids, val_frac=0.15, test_frac=0.15, seed=7)
    check("train/val/test partition covers all patients exactly once",
          sorted(train + val + test) == sorted(patient_ids))
    check("no patient appears in more than one split",
          len(set(train) & set(val)) == 0 and
          len(set(train) & set(test)) == 0 and
          len(set(val) & set(test)) == 0)
    train2, val2, test2 = patient_level_split(patient_ids, val_frac=0.15, test_frac=0.15, seed=7)
    check("same seed reproduces the same split", train == train2 and val == val2 and test == test2)
    train3, _, _ = patient_level_split(patient_ids, val_frac=0.15, test_frac=0.15, seed=99)
    check("different seed gives a different split", train3 != train)

    print("\n=== 2. CTPipeline: HU windowing, resampling, ROI cropping ===")
    pipeline = CTPipeline(target_spacing=(1.0, 1.0, 1.0), default_window="lung")
    raw = np.full((40, 40, 40), -1000.0, dtype=np.float32)
    raw[15:25, 15:25, 15:25] = 40.0  # a "lesion" cube of soft-tissue HU
    windowed = pipeline.apply_window(raw)
    check("windowed output is in [0, 1]", windowed.min() >= 0.0 and windowed.max() <= 1.0)
    check("air (-1000 HU) maps to ~0", abs(float(windowed[0, 0, 0])) < 1e-3)

    resampled, new_spacing = pipeline.resample_volume(raw, current_spacing=(2.0, 1.0, 1.0))
    check("resampling to isotropic spacing changes the D axis size",
          resampled.shape[0] != raw.shape[0])
    check("resample_volume reports the requested target spacing",
          new_spacing == pipeline.target_spacing)

    crop = pipeline.extract_nodule_crop(raw, center_zyx=(38, 38, 38), crop_size=(16, 16, 16),
                                         pad_value=-1000.0)
    check("out-of-bounds crop is padded to the requested size", crop.shape == (16, 16, 16))
    check("padded region uses the pad value", float(crop[-1, -1, -1]) == -1000.0)

    print("\n=== 3. Synthetic cohort + consensus masks ===")
    cohort = make_synthetic_cohort(n_patients=6, seed=1)
    check("cohort has requested number of patients", len(cohort) == 6)
    for nodule in cohort:
        recomputed = consensus_mask_from_annotator_masks(nodule.annotator_masks, agreement_frac=0.5)
        check(f"{nodule.patient_id}: stored consensus mask matches shared consensus function",
              np.array_equal(recomputed, nodule.consensus_mask))
        check(f"{nodule.patient_id}: consensus mask is non-empty",
              nodule.consensus_mask.sum() > 0)

    print("\n=== 4. Localization metrics against known ground truth ===")
    nodule = cohort[0]
    # A heatmap that's a noisy copy of the true mask should score well.
    rng = np.random.default_rng(0)
    heatmap = nodule.consensus_mask.astype(np.float32) * 0.9 + rng.normal(0, 0.05, nodule.consensus_mask.shape)
    heatmap = np.clip(heatmap, 0, 1)

    pred_mask = heatmap_to_mask(heatmap, method="fraction_of_max", max_fraction=0.5)
    dice, iou = dice_iou(pred_mask, nodule.consensus_mask)
    check("a heatmap that closely tracks ground truth scores high IoU", iou > 0.5)
    check("dice >= iou always (mathematical identity for the same masks)", dice >= iou)
    check("pointing game hits when the heatmap peak is inside the lesion",
          pointing_game_hit(heatmap, nodule.consensus_mask))

    # A heatmap centered somewhere else entirely should score near zero.
    bad_heatmap = np.zeros_like(heatmap)
    corner = tuple(min(5, s - 1) for s in bad_heatmap.shape)
    bad_heatmap[corner] = 1.0
    check("a heatmap pointing at the wrong voxel fails the pointing game",
          not pointing_game_hit(bad_heatmap, nodule.consensus_mask))

    report = evaluate_localization(heatmap, gt_mask=nodule.consensus_mask,
                                    spacing_mm=(1.0, 1.0, 1.0))
    check("evaluate_localization returns populated overlap metrics when GT is given",
          report.dice is not None and report.iou is not None)
    report_no_gt = evaluate_localization(heatmap, gt_mask=None)
    check("evaluate_localization returns None overlap metrics without GT (no fabricated numbers)",
          report_no_gt.dice is None and report_no_gt.iou is None and report_no_gt.pointing_game_hit is None)
    check("centroid is still reported without GT (usable at real inference time)",
          report_no_gt.centroid_voxel is not None)

    print("\nAll smoke tests passed. This validates pipeline CODE correctness on")
    print("synthetic data only — it says nothing about real-world model accuracy.")


if __name__ == "__main__":
    main()
