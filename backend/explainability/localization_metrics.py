"""
backend/explainability/localization_metrics.py
================================================
Quantitative validation for explainability heatmaps.

The other modules in this package (gradcam3d.py) PRODUCE heatmaps. This
module MEASURES whether a heatmap actually points at the lesion, so that
"the heatmap shows the exact location" is a number you can report, not an
impression from looking at one colorful image.

Three complementary metrics, standard in the XAI-localization literature:
  - IoU / Dice   : overlap between a thresholded heatmap region and the
                   ground-truth nodule mask. Strictest metric.
  - Pointing Game: does the heatmap's single highest-activation voxel fall
                   inside the ground-truth mask? Lenient, robust to a
                   heatmap being "blobby" as long as its peak is correct.
  - Centroid/bbox : where the model says the lesion is, in voxel and
                   physical (mm) coordinates — this is what the UI reports
                   even when there's no ground truth to check against
                   (e.g. at real inference time on a new patient).

None of this replaces radiologist review. These are attribution-quality
metrics, not diagnostic accuracy metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
#  Thresholding a heatmap into a binary "the model says lesion is here" mask
# ─────────────────────────────────────────────────────────────

def heatmap_to_mask(heatmap: np.ndarray, method: str = "fraction_of_max",
                     percentile: float = 90.0,
                     max_fraction: float = 0.5) -> np.ndarray:
    """Binarize a [0, 1] heatmap into a lesion-region mask.

    Args:
        heatmap:      (D, H, W) or (H, W) array in [0, 1].
        method:       'fraction_of_max' (default), 'top_percentile', or 'otsu'.
        percentile:   for 'top_percentile' only: keep voxels >= this
                      percentile of the heatmap's OWN value distribution
                      across the whole volume. This is only appropriate
                      when the lesion occupies a large-ish share of the
                      volume — a real nodule is typically a small fraction
                      of a scan, so a fixed volume-percentile threshold
                      under-/over-segments depending on lesion size. Kept
                      for cases (e.g. a tightly-cropped ROI) where it fits.
        max_fraction: for 'fraction_of_max' only: keep voxels whose value
                      is >= max_fraction * heatmap.max(). This scales with
                      the lesion's own signal regardless of how large a
                      share of the volume it occupies, which is the
                      standard approach in the Grad-CAM localization
                      literature (commonly 0.15-0.5 of peak activation).

    Returns:
        Boolean mask, same shape as heatmap.
    """
    if heatmap.size == 0:
        return np.zeros_like(heatmap, dtype=bool)

    if method == "fraction_of_max":
        peak = heatmap.max()
        if peak <= 0:
            return np.zeros_like(heatmap, dtype=bool)
        return heatmap >= (max_fraction * peak)

    if method == "top_percentile":
        thresh = np.percentile(heatmap, percentile)
        return heatmap >= thresh

    if method == "otsu":
        # Simple Otsu on the heatmap's own histogram — no extra deps.
        hist, bin_edges = np.histogram(heatmap, bins=256, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return np.zeros_like(heatmap, dtype=bool)
        sum_all = np.sum(hist * np.arange(256))
        sum_bg, weight_bg, max_var, thresh_idx = 0.0, 0.0, 0.0, 0
        for i in range(256):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += i * hist[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_all - sum_bg) / weight_fg
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > max_var:
                max_var = var_between
                thresh_idx = i
        thresh = bin_edges[thresh_idx]
        return heatmap >= thresh

    raise ValueError(f"Unknown method: {method}")


# ─────────────────────────────────────────────────────────────
#  Overlap metrics against a ground-truth mask
# ─────────────────────────────────────────────────────────────

def dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """Dice coefficient and IoU between two boolean masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum, gt_sum = pred.sum(), gt.sum()

    iou = float(intersection) / float(union) if union > 0 else 0.0
    dice = (2.0 * intersection / (pred_sum + gt_sum)
            if (pred_sum + gt_sum) > 0 else 0.0)
    return dice, iou


def pointing_game_hit(heatmap: np.ndarray, gt_mask: np.ndarray) -> bool:
    """True if the heatmap's single highest-activation voxel lies inside
    the ground-truth mask. Standard "pointing game" metric — lenient,
    since it only asks whether the PEAK is correct, not the whole shape.
    """
    if heatmap.size == 0 or gt_mask.sum() == 0:
        return False
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(gt_mask[peak_idx])


# ─────────────────────────────────────────────────────────────
#  Where does the model say the lesion is? (works with or without GT)
# ─────────────────────────────────────────────────────────────

def heatmap_centroid(heatmap: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> Tuple[float, ...]:
    """Intensity-weighted centroid of the heatmap (or of `mask` region if
    given), in voxel coordinates, e.g. (z, y, x) for a 3D heatmap.
    """
    weights = heatmap if mask is None else np.where(mask, heatmap, 0.0)
    total = weights.sum()
    if total <= 0:
        return tuple(s / 2.0 for s in heatmap.shape)  # fall back to center
    idx_grids = np.indices(heatmap.shape)
    centroid = tuple(float((g * weights).sum() / total) for g in idx_grids)
    return centroid


def heatmap_bbox(mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Axis-aligned bounding box of a boolean mask as ((min, max), ...)
    per axis. Returns None if the mask is empty.
    """
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple((int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def voxel_to_mm(coord: Tuple[float, ...],
                 spacing_mm: Tuple[float, ...]) -> Tuple[float, ...]:
    """Convert a voxel-space coordinate to physical mm using per-axis
    spacing (as stored in the CT's DICOM PixelSpacing / SliceThickness).
    """
    return tuple(c * s for c, s in zip(coord, spacing_mm))


# ─────────────────────────────────────────────────────────────
#  Combined report
# ─────────────────────────────────────────────────────────────

@dataclass
class LocalizationReport:
    centroid_voxel: Tuple[float, ...]
    centroid_mm: Optional[Tuple[float, ...]]
    bbox_voxel: Optional[Tuple[Tuple[int, int], ...]]
    area_or_volume_voxels: int
    dice: Optional[float]
    iou: Optional[float]
    pointing_game_hit: Optional[bool]
    threshold_method: str
    threshold_percentile: float

    def to_dict(self) -> dict:
        return {
            "centroid_voxel": self.centroid_voxel,
            "centroid_mm": self.centroid_mm,
            "bbox_voxel": self.bbox_voxel,
            "area_or_volume_voxels": self.area_or_volume_voxels,
            "dice": self.dice,
            "iou": self.iou,
            "pointing_game_hit": self.pointing_game_hit,
            "threshold_method": self.threshold_method,
            "threshold_percentile": self.threshold_percentile,
            "disclaimer": "MODEL ATTRIBUTION METRIC — NOT CLINICAL EVIDENCE",
        }


def evaluate_localization(heatmap: np.ndarray,
                           gt_mask: Optional[np.ndarray] = None,
                           spacing_mm: Optional[Tuple[float, ...]] = None,
                           method: str = "fraction_of_max",
                           percentile: float = 90.0) -> LocalizationReport:
    """One-stop report: where the heatmap says the lesion is, and (if a
    ground-truth mask is supplied, e.g. during offline evaluation on an
    annotated dataset) how well that matches reality.

    At real inference time on a new patient there is no ground truth, so
    gt_mask is optional — dice/iou/pointing_game_hit come back as None and
    only the location fields (centroid/bbox/area) are populated. That's
    exactly the split the reference UI's morphometry panel needs: fields
    that can be shown for every case vs. a quality metric only computable
    against annotated benchmark data.
    """
    pred_mask = heatmap_to_mask(heatmap, method=method, percentile=percentile)
    centroid_voxel = heatmap_centroid(heatmap, mask=pred_mask)
    centroid_mm = voxel_to_mm(centroid_voxel, spacing_mm) if spacing_mm else None
    bbox = heatmap_bbox(pred_mask)
    area = int(pred_mask.sum())

    dice = iou = hit = None
    if gt_mask is not None:
        dice, iou = dice_iou(pred_mask, gt_mask)
        hit = pointing_game_hit(heatmap, gt_mask)

    return LocalizationReport(
        centroid_voxel=centroid_voxel,
        centroid_mm=centroid_mm,
        bbox_voxel=bbox,
        area_or_volume_voxels=area,
        dice=dice,
        iou=iou,
        pointing_game_hit=hit,
        threshold_method=method,
        threshold_percentile=percentile,
    )
