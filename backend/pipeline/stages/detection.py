"""
backend/pipeline/stages/detection.py
=======================================
Stage 3 — Nodule Candidate Detection via Heuristic Multi-Scale LoG/DoG.

Status:
    Rule-based heuristic candidate generator. There is no trained deep
    learning detector checkpoint in the repository. Candidate proposals
    are extracted using:
      1. Segmentation lung mask restriction (excluding extrapulmonary tissue)
      2. Multi-scale Difference-of-Gaussians (DoG) blob detection
      3. 3D spatial covariance analysis for tubular blood vessel rejection
      4. Anatomical size filtering (3.0 mm to 30.0 mm equivalent diameter)

Reads:
    ctx.get(PREPROCESSING) → {"volume": np.ndarray, "spacing": tuple}
    ctx.get(SEGMENTATION)  → {"lung_mask": np.ndarray (optional)}

Writes:
    ctx.set(DETECTION, {
        "candidates": [
            {"bbox_zyx": [z1,y1,x1,z2,y2,x2], "score": 0.85, "diameter_mm": 8.4, "source": "heuristic_log_dog"},
            ...
        ],
        "nodule_count": int,
        "spacing_mm": list,
        "detector_type": "heuristic_multi_scale_dog",
    })
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi

from backend.monitoring import get_logger, MetricsRegistry
from backend.pipeline.orchestrator import PipelineContext, PipelineStage

log = get_logger(__name__)


class DetectionStage:
    """
    Heuristic multi-scale blob detector for pulmonary nodule candidate generation.

    Employs 3D Difference-of-Gaussians (DoG) filtering restricted strictly
    to the segmented lung parenchyma, followed by 3D spatial covariance
    eigenvalue filtering to reject tubular vascular structures and pleural sheets.
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints_3d"),
        device: str = "cpu",
        score_threshold: float = 0.005,
        max_candidates: int = 800,
        min_diameter_mm: float = 2.5,
        max_diameter_mm: float = 35.0,
        max_elongation: float = 4.5,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.score_threshold = score_threshold
        self.max_candidates = max_candidates
        self.min_diameter_mm = min_diameter_mm
        self.max_diameter_mm = max_diameter_mm
        self.max_elongation = max_elongation

    # ------------------------------------------------------------------
    # Stage entrypoint
    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        prep = ctx.get(PipelineStage.PREPROCESSING)
        if prep is None:
            ctx.warnings.append("DetectionStage: preprocessing output missing, skipping")
            return {"candidates": [], "nodule_count": 0}

        volume: Optional[np.ndarray] = prep.get("volume")
        raw_volume: Optional[np.ndarray] = prep.get("raw_volume")
        spacing: tuple = prep.get("spacing", (1.0, 1.0, 1.0))

        if volume is None:
            return {"candidates": [], "nodule_count": 0}

        seg = ctx.get(PipelineStage.SEGMENTATION) or {}
        lung_mask: Optional[np.ndarray] = seg.get("lung_mask")

        with MetricsRegistry.time_inference("HeuristicDoGDetector"):
            candidates = self.detect_candidates(
                volume=volume,
                raw_volume=raw_volume,
                spacing=spacing,
                lung_mask=lung_mask,
            )

        log.info(
            "detection_done",
            case_id=ctx.case_id,
            detector="heuristic_multi_scale_dog",
            nodule_count=len(candidates),
        )

        return {
            "candidates": candidates,
            "nodule_count": len(candidates),
            "spacing_mm": list(spacing),
            "detector_type": "heuristic_multi_scale_dog",
        }

    # ------------------------------------------------------------------
    # Multi-scale LoG/DoG Candidate Detection
    # ------------------------------------------------------------------
    def detect_candidates(
        self,
        volume: np.ndarray,
        raw_volume: Optional[np.ndarray] = None,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        lung_mask: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs multi-scale blob detection on 3D thoracic CT volume.
        Spacing is in (dz, dy, dx) mm order for numpy array.
        """
        dz, dy, dx = float(spacing[0]), float(spacing[1]), float(spacing[2])

        # Normalize volume to [0, 1] and get raw HU
        if raw_volume is not None:
            raw_hu = raw_volume
            norm_vol = np.clip((raw_hu.astype(np.float32) + 1000.0) / 1400.0, 0.0, 1.0)
        elif volume.max() > 10.0 or volume.min() < -100.0:
            raw_hu = volume
            norm_vol = np.clip((raw_hu.astype(np.float32) + 1000.0) / 1400.0, 0.0, 1.0)
        else:
            norm_vol = volume.astype(np.float32)
            raw_hu = norm_vol * 1400.0 - 1000.0

        # Anisotropic-aware downsampling: preserve axial resolution when dz >= 1.5mm
        step_z = 1 if dz >= 1.5 else 2
        step_y = 2
        step_x = 2

        down_raw = raw_hu[::step_z, ::step_y, ::step_x]
        down_norm = norm_vol[::step_z, ::step_y, ::step_x]
        dz2, dy2, dx2 = dz * step_z, dy * step_y, dx * step_x

        # Step 1: Obtain or compute lung mask on downsampled volume
        if lung_mask is not None and np.any(lung_mask):
            down_mask = lung_mask[::step_z, ::step_y, ::step_x]
        else:
            down_mask = self._extract_lung_mask_downsampled(down_raw)

        # Step 2: Multi-scale Difference of Gaussians (DoG) with bounded filter sizes
        s1 = np.maximum(np.array([1.2 / dz2, 1.2 / dy2, 1.2 / dx2]), 0.5)
        s2 = np.maximum(np.array([2.5 / dz2, 2.5 / dy2, 2.5 / dx2]), 0.75)
        s3 = np.maximum(np.array([4.2 / dz2, 4.2 / dy2, 4.2 / dx2]), 1.0)

        dog1 = ndi.gaussian_filter(down_norm, sigma=s1) - ndi.gaussian_filter(down_norm, sigma=s2)
        dog2 = ndi.gaussian_filter(down_norm, sigma=s2) - ndi.gaussian_filter(down_norm, sigma=s3)
        dog = np.maximum(dog1, dog2)

        # Step 3: Find 3D local maxima inside lung mask with physical HU > -650
        footprint = np.ones((3, 3, 3), dtype=bool)
        is_max = (dog == ndi.maximum_filter(dog, footprint=footprint))
        cand_mask = is_max & down_mask & (down_raw >= -650) & (down_raw <= 300) & (dog > self.score_threshold)

        pts = np.argwhere(cand_mask)
        if len(pts) == 0:
            return []

        scores = dog[cand_mask]
        candidates: List[Dict[str, Any]] = []

        # Step 4: Size & Vessel-likeness filtering via 3D spatial covariance
        for idx, (pz, py, px) in enumerate(pts):
            dog_score = float(scores[idx])
            hu_val = float(down_raw[pz, py, px])

            # Local 5x5x5 patch for covariance analysis
            z0, z1 = max(0, pz - 2), min(down_raw.shape[0], pz + 3)
            y0, y1 = max(0, py - 2), min(down_raw.shape[1], py + 3)
            x0, x1 = max(0, px - 2), min(down_raw.shape[2], px + 3)
            patch = down_raw[z0:z1, y0:y1, x0:x1]

            thresh = max(-650.0, hu_val - 180.0)
            blob = np.argwhere(patch >= thresh)

            if len(blob) >= 4:
                p_coords = blob * np.array([dz2, dy2, dx2])
                centered = p_coords - p_coords.mean(axis=0)
                cov = (centered.T @ centered) / len(blob)
                eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
                e1, e2, e3 = max(float(eigs[0]), 1e-4), max(float(eigs[1]), 1e-4), max(float(eigs[2]), 1e-4)
                elongation = float(np.sqrt(e1 / e2))
                flatness = float(np.sqrt(e2 / e3))

                # Vessel rejection: elongated tubular structures & pleural sheets
                if elongation > 5.5 or flatness > 7.0:
                    continue

                vol_mm3 = len(blob) * (dz2 * dy2 * dx2)
                diam_mm = float(2.0 * ((3.0 * vol_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0))
                if diam_mm < 2.0 or diam_mm > 40.0:
                    continue
            else:
                elongation = 1.0
                diam_mm = 3.0

            # Composite ranking score: DoG peak weighted by sphericity and nodular HU
            sphericity_weight = 1.0 / (1.0 + 0.3 * max(0.0, elongation - 1.5) ** 2)
            hu_weight = 1.25 if (-400.0 <= hu_val <= 100.0) else 0.85
            composite_score = float(dog_score * sphericity_weight * hu_weight)

            # Map to original coordinate space
            cz = float(pz * step_z)
            cy = float(py * step_y)
            cx = float(px * step_x)

            half_box = max(int(np.ceil(diam_mm / (2.0 * min(dz, dy, dx)))), 4)
            z1_box = max(0.0, cz - half_box)
            z2_box = min(float(volume.shape[0]), cz + half_box)
            y1_box = max(0.0, cy - half_box)
            y2_box = min(float(volume.shape[1]), cy + half_box)
            x1_box = max(0.0, cx - half_box)
            x2_box = min(float(volume.shape[2]), cx + half_box)

            candidates.append({
                "bbox_zyx": [z1_box, y1_box, x1_box, z2_box, y2_box, x2_box],
                "centroid_zyx": [cz, cy, cx],
                "score": round(composite_score, 4),
                "dog_score": round(dog_score, 4),
                "diameter_mm": round(diam_mm, 2),
                "elongation": round(elongation, 2),
                "source": "heuristic_log_dog",
            })

        # Sort by composite score descending
        candidates.sort(key=lambda c: -c["score"])
        return candidates[: self.max_candidates]

    # ------------------------------------------------------------------
    # Morphological Lung Mask Extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_lung_mask_downsampled(down_raw: np.ndarray) -> np.ndarray:
        """
        Extracts 3D lung parenchyma mask on downsampled volume (fast).
        """
        lung_mask = np.zeros_like(down_raw, dtype=bool)

        for z in range(down_raw.shape[0]):
            sl = down_raw[z]
            air = sl < -400.0
            lbl, n = ndi.label(air)
            if n == 0:
                continue
            corners = {lbl[0, 0], lbl[0, -1], lbl[-1, 0], lbl[-1, -1]}
            for c in range(1, n + 1):
                if c not in corners and np.sum(lbl == c) > 50:
                    lung_mask[z] |= (lbl == c)
            if np.any(lung_mask[z]):
                lung_mask[z] = ndi.binary_fill_holes(lung_mask[z])

        return ndi.binary_dilation(lung_mask, iterations=2)
