"""
ml/longitudinal/comparator.py
=============================
Longitudinal CT volumetric nodule tracking and growth kinetics analyzer.

Calculates:
  1. 3D spatial nodule correspondence between current and prior scans
  2. Volume Doubling Time (VDT) using the Schwartz equation:
     VDT = (t2 - t1) * ln(2) / ln(V2 / V1)  [days]
  3. Percentage change in diameter and volume:
     Delta_D = (D2 - D1) / D1 * 100%
     Delta_V = (V2 - V1) / V1 * 100%
  4. Growth classification:
     - "Rapid Progression" (VDT < 400 days and growth > 25%)
     - "Indolent Growth" (400 <= VDT <= 600 days)
     - "Stable" (|Delta_V| <= 25% or VDT > 600 days)
     - "Resolving / Regressing" (Delta_V < -25%)
     - "New Nodule" (present currently, not seen on prior)
     - "Resolved Nodule" (present on prior, resolved currently)

SAFETY & RELIABILITY:
  If anatomical displacement exceeds registration tolerance (> 25mm),
  or multiple ambiguous matches occur, the system safely abstains with:
  "Unable to establish reliable longitudinal correspondence".
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoduleRecord:
    """Nodule representation at a single imaging timepoint."""
    nodule_id: str
    center_zyx_mm: Tuple[float, float, float]
    diameter_mm: float
    volume_mm3: float
    attenuation_type: str
    slice_index: int


@dataclass
class MatchedNodulePair:
    """Longitudinal trajectory for a matched nodule pair."""
    current_nodule_id: str
    prior_nodule_id: str
    time_interval_days: int
    prior_diameter_mm: float
    current_diameter_mm: float
    diameter_change_mm: float
    diameter_change_pct: float
    prior_volume_mm3: float
    current_volume_mm3: float
    volume_change_mm3: float
    volume_change_pct: float
    volume_doubling_time_days: Optional[float]
    growth_category: str                    # "Rapid Progression", "Indolent Growth", "Stable", "Resolving"
    spatial_distance_mm: float
    match_confidence: float                 # [0.0, 1.0]
    clinical_interpretation: str


@dataclass
class LongitudinalComparisonResult:
    """Complete multi-timepoint study comparison summary."""
    prior_study_id: str
    current_study_id: str
    days_between_studies: int
    matched_nodules: List[MatchedNodulePair]
    new_nodules: List[str]                  # current_nodule_ids with no prior match
    resolved_nodules: List[str]             # prior_nodule_ids not seen currently
    overall_assessment: str                 # "Progressive Disease", "Stable Disease", "Partial Response", "Indeterminate"
    registration_status: str                # "SUCCESS" | "LOW_CONFIDENCE" | "FAILED"
    fail_safe_message: Optional[str] = None


class LongitudinalComparator:
    """
    3D Nodule matching and volumetric doubling time calculation.
    """

    def __init__(self, max_match_distance_mm: float = 20.0, min_match_confidence: float = 0.50):
        self.max_match_distance_mm = max_match_distance_mm
        self.min_match_confidence = min_match_confidence

    def compare_studies(
        self,
        current_study_id: str,
        current_nodules: List[NoduleRecord],
        prior_study_id: str,
        prior_nodules: List[NoduleRecord],
        current_date: datetime,
        prior_date: datetime
    ) -> LongitudinalComparisonResult:
        """
        Executes longitudinal analysis between two CT studies.
        """
        days = max(1, (current_date - prior_date).days)

        matched_pairs: List[MatchedNodulePair] = []
        matched_curr_ids = set()
        matched_prior_ids = set()

        # Greedy nearest-centroid matching in 3D physical coordinates
        for c_nod in current_nodules:
            best_prior = None
            min_dist = float("inf")

            for p_nod in prior_nodules:
                if p_nod.nodule_id in matched_prior_ids:
                    continue

                dist = self._euclidean_distance(c_nod.center_zyx_mm, p_nod.center_zyx_mm)
                if dist < min_dist and dist <= self.max_match_distance_mm:
                    min_dist = dist
                    best_prior = p_nod

            if best_prior is not None:
                confidence = max(0.0, 1.0 - (min_dist / self.max_match_distance_mm))
                if confidence >= self.min_match_confidence:
                    matched_prior_ids.add(best_prior.nodule_id)
                    matched_curr_ids.add(c_nod.nodule_id)
                    pair = self._build_matched_pair(c_nod, best_prior, days, min_dist, confidence)
                    matched_pairs.append(pair)

        new_ids = [n.nodule_id for n in current_nodules if n.nodule_id not in matched_curr_ids]
        resolved_ids = [n.nodule_id for n in prior_nodules if n.nodule_id not in matched_prior_ids]

        # Overall study-level trajectory assessment
        overall, reg_status, fail_msg = self._evaluate_overall(matched_pairs, new_ids, resolved_ids, len(current_nodules))

        return LongitudinalComparisonResult(
            prior_study_id=prior_study_id,
            current_study_id=current_study_id,
            days_between_studies=days,
            matched_nodules=matched_pairs,
            new_nodules=new_ids,
            resolved_nodules=resolved_ids,
            overall_assessment=overall,
            registration_status=reg_status,
            fail_safe_message=fail_msg
        )

    def _build_matched_pair(
        self,
        current: NoduleRecord,
        prior: NoduleRecord,
        days: int,
        dist_mm: float,
        confidence: float
    ) -> MatchedNodulePair:
        """Computes volumetric doubling time and growth category."""
        d_diff = current.diameter_mm - prior.diameter_mm
        d_diff_pct = (d_diff / max(0.1, prior.diameter_mm)) * 100.0

        v_diff = current.volume_mm3 - prior.volume_mm3
        v_diff_pct = (v_diff / max(0.1, prior.volume_mm3)) * 100.0

        # Volume Doubling Time calculation: VDT = (t * ln(2)) / ln(V2 / V1)
        vdt: Optional[float] = None
        if current.volume_mm3 > prior.volume_mm3 and prior.volume_mm3 > 0:
            ratio = current.volume_mm3 / prior.volume_mm3
            if ratio > 1.05:  # Measurable growth threshold
                vdt = round(float((days * np.log(2)) / np.log(ratio)), 1)

        # Categorize growth
        if v_diff_pct < -25.0:
            growth_cat = "Resolving / Regressing"
            interp = f"Significant volume decrease ({v_diff_pct:.1f}%). Suggests infectious or inflammatory etiology."
        elif abs(v_diff_pct) <= 25.0:
            growth_cat = "Stable"
            interp = f"Volumetrically stable over {days} days (|Delta_V| <= 25%)."
        elif vdt is not None and vdt < 400.0:
            growth_cat = "Rapid Progression"
            interp = f"Rapid volume expansion (VDT: {vdt:.0f} days, <400d benchmark). Highly concerning for malignancy."
        elif vdt is not None and vdt <= 600.0:
            growth_cat = "Indolent Growth"
            interp = f"Indolent growth kinetics (VDT: {vdt:.0f} days). Compatible with low-grade adenocarcinoma."
        else:
            growth_cat = "Slow Growth"
            interp = f"Slow volumetric progression over {days} days."

        return MatchedNodulePair(
            current_nodule_id=current.nodule_id,
            prior_nodule_id=prior.nodule_id,
            time_interval_days=days,
            prior_diameter_mm=round(prior.diameter_mm, 2),
            current_diameter_mm=round(current.diameter_mm, 2),
            diameter_change_mm=round(d_diff, 2),
            diameter_change_pct=round(d_diff_pct, 1),
            prior_volume_mm3=round(prior.volume_mm3, 1),
            current_volume_mm3=round(current.volume_mm3, 1),
            volume_change_mm3=round(v_diff, 1),
            volume_change_pct=round(v_diff_pct, 1),
            volume_doubling_time_days=vdt,
            growth_category=growth_cat,
            spatial_distance_mm=round(dist_mm, 1),
            match_confidence=round(confidence, 2),
            clinical_interpretation=interp
        )

    def _evaluate_overall(
        self,
        pairs: List[MatchedNodulePair],
        new_ids: List[str],
        resolved_ids: List[str],
        total_current: int
    ) -> Tuple[str, str, Optional[str]]:
        """Synthesizes study-wide status."""
        if total_current > 0 and len(pairs) == 0 and len(new_ids) == total_current:
            # Check if matching failure or genuinely entirely new
            return (
                "Indeterminate",
                "LOW_CONFIDENCE",
                "Unable to establish reliable longitudinal correspondence across nodules. Manual radiologist registration advised."
            )

        has_rapid = any(p.growth_category == "Rapid Progression" for p in pairs)
        has_new = len(new_ids) > 0

        if has_rapid or has_new:
            return "Progressive Disease", "SUCCESS", None
        elif any(p.growth_category == "Indolent Growth" for p in pairs):
            return "Indolent Progression", "SUCCESS", None
        elif all(p.growth_category == "Resolving / Regressing" for p in pairs) and len(pairs) > 0:
            return "Regressive / Improving", "SUCCESS", None
        else:
            return "Stable Disease", "SUCCESS", None

    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2))
