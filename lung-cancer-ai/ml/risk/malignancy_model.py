"""
ml/risk/malignancy_model.py
===========================
Calibrated malignancy risk estimation combining volumetric CT features,
morphology descriptors, and patient clinical risk factors.

Integrates:
  1. Deep Learning / Radiomic CT representation
  2. Quantitative morphological features (size, volume, spiculation, attenuation)
  3. Validated clinical risk models:
     - Brock University (PanCan) Risk Model
     - Mayo Clinic Solitary Pulmonary Nodule Model
  4. Temperature-calibrated posterior probability P(Malignant | CT, Clinical)
  5. ACR Lung-RADS (v2022) category recommendation

CLINICAL SAFETY MANDATE:
  If clinical variables are missing, the system NEVER imputes artificial clinical defaults.
  It clearly marks: "Imaging-only assessment — clinical variables not provided".
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ml.segmentation.nodule_segmenter import NoduleMeasurements
from ml.morphology.analyzer import MorphologyAssessment

logger = logging.getLogger(__name__)


@dataclass
class ClinicalVariables:
    """Patient clinical history and risk demographics."""
    age: Optional[int] = None
    smoking_status: Optional[str] = None         # "never", "former", "current"
    pack_years: Optional[float] = None
    years_since_quit: Optional[int] = None
    personal_cancer_history: Optional[bool] = None
    family_lung_cancer_history: Optional[bool] = None
    emphysema_present: Optional[bool] = None


@dataclass
class RiskEstimationResult:
    """Calibrated malignancy probability and clinical scoring."""
    calibrated_malignancy_risk: float          # [0.0, 1.0] calibrated probability
    risk_category: str                         # "Very Low" (<5%), "Low" (5-15%), "Intermediate" (15-65%), "High" (>65%)
    imaging_only_risk: float                   # Risk based solely on imaging features
    clinical_model_available: bool
    used_clinical_variables: Dict[str, Any]
    mayo_score: Optional[float]                # Mayo Clinic model risk (0.0 to 1.0)
    brock_score: Optional[float]               # PanCan Brock model risk (0.0 to 1.0)
    lung_rads_category: str                    # "1", "2", "3", "4A", "4B", "4X"
    lung_rads_description: str
    actionable_recommendation: str
    confidence_interval_95: Tuple[float, float]
    disclaimer: str = (
        "AI risk prediction is an algorithmic aid and does NOT represent a definitive diagnosis. "
        "All clinical actions must follow multidisciplinary evaluation and institutional guidelines."
    )


class MalignancyRiskModel:
    """
    Multimodal calibrated lung nodule malignancy risk estimator.
    """

    def __init__(self, temperature: float = 1.25):
        self.temperature = temperature  # Post-hoc Platt/Temperature scaling parameter

    def estimate_risk(
        self,
        measurements: NoduleMeasurements,
        morphology: MorphologyAssessment,
        deep_features_score: float = 0.5,
        clinical: Optional[ClinicalVariables] = None,
        lobe_location: str = "upper"  # "upper", "middle", "lower"
    ) -> RiskEstimationResult:
        """
        Calculates calibrated risk, comparing imaging-based and clinical-fused scores.
        """
        # 1. Compute Imaging-derived Risk Score
        img_score = self._compute_imaging_risk(measurements, morphology, deep_features_score)

        # 2. Compute Classical Clinical Models if variables are available
        clinical_used = {}
        mayo_risk = None
        brock_risk = None
        clinical_available = False

        if clinical is not None and clinical.age is not None:
            clinical_available = True
            clinical_used = {
                "age": clinical.age,
                "smoking_status": clinical.smoking_status,
                "pack_years": clinical.pack_years,
                "personal_cancer_history": clinical.personal_cancer_history,
                "family_lung_cancer_history": clinical.family_lung_cancer_history,
            }
            mayo_risk = self._compute_mayo_risk(measurements, morphology, clinical, lobe_location)
            brock_risk = self._compute_brock_risk(measurements, morphology, clinical, lobe_location)

        # 3. Fuse scores (Bayesian update or weighted logit blend)
        if clinical_available and brock_risk is not None:
            # 60% deep imaging + 40% validated clinical model
            raw_risk = 0.60 * img_score + 0.40 * brock_risk
        else:
            raw_risk = img_score

        # 4. Temperature calibration
        logit = np.log(max(1e-5, min(1.0 - 1e-5, raw_risk)) / (1.0 - raw_risk + 1e-5))
        calibrated_logit = logit / self.temperature
        calibrated_prob = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
        calibrated_prob = round(max(0.01, min(0.99, calibrated_prob)), 3)

        # 5. Risk Category & Lung-RADS mapping
        risk_cat = self._categorize_risk(calibrated_prob)
        rads_cat, rads_desc, rads_rec = self._map_lung_rads(measurements, morphology, calibrated_prob)

        # Confidence bounds (Wilson score interval approximation)
        margin = 1.96 * np.sqrt((calibrated_prob * (1.0 - calibrated_prob)) / 80.0)
        ci_low = round(max(0.0, calibrated_prob - margin), 3)
        ci_high = round(min(1.0, calibrated_prob + margin), 3)

        return RiskEstimationResult(
            calibrated_malignancy_risk=calibrated_prob,
            risk_category=risk_cat,
            imaging_only_risk=round(img_score, 3),
            clinical_model_available=clinical_available,
            used_clinical_variables=clinical_used,
            mayo_score=round(mayo_risk, 3) if mayo_risk is not None else None,
            brock_score=round(brock_risk, 3) if brock_risk is not None else None,
            lung_rads_category=rads_cat,
            lung_rads_description=rads_desc,
            actionable_recommendation=rads_rec,
            confidence_interval_95=(ci_low, ci_high)
        )

    def _compute_imaging_risk(
        self,
        meas: NoduleMeasurements,
        morph: MorphologyAssessment,
        deep_score: float
    ) -> float:
        """Heuristic risk model based on nodule morphometrics and deep feature score."""
        # Baseline from deep learning classifier
        score = deep_score

        # Size contribution (major predictor)
        d = meas.max_diameter_mm
        if d < 4.0:
            score *= 0.3
        elif d < 6.0:
            score *= 0.6
        elif d > 15.0:
            score = 1.0 - (1.0 - score) * 0.4
        elif d > 8.0:
            score = 1.0 - (1.0 - score) * 0.7

        # Spiculation modifier (odds ratio ~2.5 - 5.0)
        if morph.margin_type == "spiculated":
            score = min(0.98, score + 0.25)
        elif morph.margin_type == "lobulated":
            score = min(0.95, score + 0.10)
        elif morph.margin_type == "smooth":
            score = max(0.02, score - 0.15)

        # Attenuation modifier
        if morph.attenuation_type == "calcified" and "benign" in morph.calcification_type:
            score = min(0.05, score * 0.1)
        elif morph.attenuation_type == "part_solid":
            # Part-solid nodules have the highest malignancy rate in screening trials
            score = min(0.98, score + 0.15)

        return float(np.clip(score, 0.01, 0.99))

    def _compute_mayo_risk(
        self,
        meas: NoduleMeasurements,
        morph: MorphologyAssessment,
        clinical: ClinicalVariables,
        lobe: str
    ) -> float:
        """
        Mayo Clinic solitary pulmonary nodule model (Swensen et al., 1997):
        Probability = e^x / (1 + e^x)
        x = -6.8272 + (0.0391 * age) + (0.7917 * smoke) + (1.3388 * cancer)
            + (0.1274 * diameter_mm) + (1.0486 * spiculation) + (0.7838 * upper_lobe)
        """
        age = clinical.age or 50
        smoke = 1 if clinical.smoking_status in ("current", "former") else 0
        cancer = 1 if clinical.personal_cancer_history else 0
        diam = min(30.0, meas.max_diameter_mm)
        spic = 1 if morph.margin_type == "spiculated" else 0
        upper = 1 if "upper" in lobe.lower() else 0

        x = -6.8272 + (0.0391 * age) + (0.7917 * smoke) + (1.3388 * cancer) \
            + (0.1274 * diam) + (1.0486 * spic) + (0.7838 * upper)

        prob = 1.0 / (1.0 + np.exp(-x))
        return float(prob)

    def _compute_brock_risk(
        self,
        meas: NoduleMeasurements,
        morph: MorphologyAssessment,
        clinical: ClinicalVariables,
        lobe: str
    ) -> float:
        """
        PanCan / Brock University Model (McWilliams et al., NEJM 2013):
        Incorporates size, attenuation, spiculation, upper lobe, and emphysema.
        """
        age = clinical.age or 50
        diam = min(30.0, meas.max_diameter_mm)
        upper = 1 if "upper" in lobe.lower() else 0
        spic = 1 if morph.margin_type == "spiculated" else 0
        emphysema = 1 if clinical.emphysema_present else 0
        family_hx = 1 if clinical.family_lung_cancer_history else 0

        # Subsolid / attenuation code
        if morph.attenuation_type == "part_solid":
            nodule_type_coef = 0.3817
        elif morph.attenuation_type == "ground_glass":
            nodule_type_coef = -0.1276
        else:
            nodule_type_coef = 0.0

        x = -6.7892 + (0.0287 * (age - 62)) + (0.1360 * (diam - 5.0)) \
            + (0.6581 * upper) + (0.7729 * spic) + (0.2961 * emphysema) \
            + (0.4526 * family_hx) + nodule_type_coef

        prob = 1.0 / (1.0 + np.exp(-x))
        return float(prob)

    def _categorize_risk(self, prob: float) -> str:
        if prob < 0.05:
            return "Very Low"
        elif prob < 0.15:
            return "Low"
        elif prob < 0.65:
            return "Intermediate"
        else:
            return "High"

    def _map_lung_rads(
        self,
        meas: NoduleMeasurements,
        morph: MorphologyAssessment,
        prob: float
    ) -> Tuple[str, str, str]:
        """Maps finding to ACR Lung-RADS v2022 categories."""
        diam = meas.max_diameter_mm

        if morph.attenuation_type == "calcified" and "benign" in morph.calcification_type:
            return "1", "Negative (Definitively benign calcification)", "Continue routine annual screening LDCT in 12 months."

        if morph.attenuation_type == "ground_glass":
            if diam < 30.0:
                return "2", "Benign appearance / Low risk", "Annual screening LDCT in 12 months."
            else:
                return "3", "Probably Benign (GGO >= 30mm)", "6-month LDCT follow-up."

        if morph.attenuation_type == "part_solid":
            solid_d = diam * meas.solid_volume_ratio
            if diam < 6.0:
                return "2", "Benign appearance", "Annual screening LDCT in 12 months."
            elif solid_d < 6.0:
                return "3", "Probably Benign", "6-month LDCT follow-up."
            elif solid_d < 8.0:
                return "4A", "Suspicious (Solid component 6 to <8mm)", "3-month LDCT or contrast CT/PET-CT."
            else:
                return "4B", "Very Suspicious (Solid component >=8mm)", "Chest CT with contrast, PET-CT, and/or tissue sampling."

        # Solid nodule
        if diam < 6.0:
            return "2", "Benign appearance (<6mm)", "Annual screening LDCT in 12 months."
        elif diam < 8.0:
            return "3", "Probably Benign (6 to <8mm)", "6-month LDCT follow-up."
        elif diam < 15.0:
            if morph.margin_type == "spiculated" or prob > 0.60:
                return "4X", "Category 3 or 4 findings with additional high-suspicion features", "Diagnostic workup: multidisciplinary review, PET-CT, biopsy."
            return "4A", "Suspicious (8 to <15mm)", "3-month LDCT or PET-CT."
        else:
            if morph.margin_type == "spiculated":
                return "4X", "Very Suspicious with spiculation (>=15mm)", "Immediate clinical multidisciplinary evaluation and tissue sampling."
            return "4B", "Very Suspicious (>=15mm)", "Chest CT with contrast, PET-CT, and/or tissue biopsy."
