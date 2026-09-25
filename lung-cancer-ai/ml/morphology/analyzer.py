"""
ml/morphology/analyzer.py
=========================
Nodule morphological characterization engine.

Analyzes:
  1. Attenuation pattern: Solid, Part-Solid (Sub-solid), Pure Ground-Glass Opacity (GGO), Calcified
  2. Margin characteristics: Smooth, Lobulated, Spiculated (corona radiata), Ill-defined
  3. Internal architecture: Solid, Cavitary, Air bronchogram, Internal lucency
  4. Anatomical relationship: Pleural contact (pleural-based, juxta-pleural, parenchymal),
     Vascular relationship (juxta-vascular, solitary)
  5. Calcification pattern: Central, Diffuse, Popcorn, Stippled, Eccentric

CLINICAL & ETHICAL REQUIREMENT:
  This module explicitly characterizes morphological imaging phenotypes according to
  Fleischner Society and Lung-RADS terminology. It NEVER claims or attempts histopathological
  diagnosis (e.g. adenocarcinoma vs. squamous cell carcinoma).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import scipy.ndimage as ndi
except ImportError:
    ndi = None

from ml.segmentation.nodule_segmenter import NoduleMeasurements

logger = logging.getLogger(__name__)


@dataclass
class MorphologyAssessment:
    """Standardized radiological morphological characterization."""
    attenuation_type: str                   # "solid" | "part_solid" | "ground_glass" | "calcified"
    margin_type: str                        # "smooth" | "lobulated" | "spiculated" | "ill_defined"
    internal_structure: str                 # "homogeneous" | "cavitary" | "air_bronchogram" | "heterogeneous"
    calcification_type: str                 # "none" | "benign_diffuse" | "benign_popcorn" | "eccentric_suspicious"
    pleural_relationship: str               # "parenchymal" | "juxta_pleural" | "pleural_abutment"
    vascular_relationship: str              # "solitary" | "juxta_vascular" | "vessel_embedded"
    spiculation_index: float                # [0.0, 1.0] (higher = more irregular/spiculated)
    lobulation_index: float                 # [0.0, 1.0]
    calcification_percentage: float         # [0.0, 100.0]
    clinical_note: str
    histopathology_disclaimer: str = (
        "Morphological assessment is based strictly on radiologic density and contour features. "
        "Histopathological subtype cannot and should not be determined from imaging alone."
    )


class MorphologyAnalyzer:
    """
    Evaluates nodule morphological phenotypes using HU distribution, gradient analysis,
    and anatomical neighborhood proximity.
    """

    def analyze(
        self,
        crop_hu: np.ndarray,
        mask: np.ndarray,
        measurements: NoduleMeasurements,
        lung_mask_crop: Optional[np.ndarray] = None
    ) -> MorphologyAssessment:
        """
        Classifies nodule morphology based on 3D crop and segmentation mask.
        """
        if mask.sum() == 0:
            return self._default_empty()

        # 1. Attenuation classification (Fleischner criteria)
        attenuation, calc_type, calc_pct = self._classify_attenuation(crop_hu, mask, measurements)

        # 2. Margin analysis (spiculation, lobulation, smooth)
        margin, spiculation_idx, lobulation_idx = self._classify_margin(mask, measurements)

        # 3. Internal structure (cavity, air bronchogram)
        internal = self._classify_internal(crop_hu, mask)

        # 4. Pleural and vascular relationship
        pleural, vascular = self._classify_relationships(crop_hu, mask, lung_mask_crop)

        # 5. Synthesize clinical observation note
        note = (
            f"{attenuation.replace('_', '-').title()} nodule measuring {measurements.max_diameter_mm:.1f} mm "
            f"with {margin} margins and {internal} internal architecture. "
            f"Anatomically {pleural.replace('_', ' ')} and {vascular.replace('_', ' ')}."
        )

        return MorphologyAssessment(
            attenuation_type=attenuation,
            margin_type=margin,
            internal_structure=internal,
            calcification_type=calc_type,
            pleural_relationship=pleural,
            vascular_relationship=vascular,
            spiculation_index=round(spiculation_idx, 3),
            lobulation_index=round(lobulation_idx, 3),
            calcification_percentage=round(calc_pct, 1),
            clinical_note=note
        )

    def _classify_attenuation(
        self,
        crop_hu: np.ndarray,
        mask: np.ndarray,
        meas: NoduleMeasurements
    ) -> Tuple[str, str, float]:
        """Categorizes attenuation based on HU distribution within mask."""
        voxels_hu = crop_hu[mask > 0]
        n_total = len(voxels_hu)

        # Calcification check (> 200 HU)
        calc_voxels = (voxels_hu > 200.0).sum()
        calc_pct = (calc_voxels / n_total) * 100.0

        if calc_pct > 50.0:
            # Dense calcification - typically benign (hamartoma / granuloma)
            if calc_pct > 80.0:
                calc_type = "benign_diffuse"
            else:
                calc_type = "benign_popcorn"
            return "calcified", calc_type, calc_pct
        elif calc_pct > 10.0:
            calc_type = "eccentric_suspicious"
        else:
            calc_type = "none"

        # Ground glass opacity (-750 to -300 HU) vs Solid (> -300 HU)
        solid_voxels = (voxels_hu > -300.0).sum()
        solid_fraction = solid_voxels / n_total

        if solid_fraction < 0.15:
            attenuation = "ground_glass"
        elif solid_fraction < 0.85:
            attenuation = "part_solid"
        else:
            attenuation = "solid"

        return attenuation, calc_type, calc_pct

    def _classify_margin(
        self,
        mask: np.ndarray,
        meas: NoduleMeasurements
    ) -> Tuple[str, float, float]:
        """Evaluates margin roughness and contour regularity."""
        sphericity = meas.sphericity
        compactness = meas.compactness

        # Spiculation proxy: low sphericity + high surface-to-volume ratio deviation
        spiculation_idx = max(0.0, min(1.0, 1.0 - sphericity))

        # Lobulation proxy: elongation and intermediate moment discrepancy
        lobulation_idx = max(0.0, min(1.0, (meas.elongation - 1.0) / 2.0))

        if spiculation_idx > 0.45:
            margin = "spiculated"
        elif lobulation_idx > 0.35:
            margin = "lobulated"
        elif sphericity > 0.75:
            margin = "smooth"
        else:
            margin = "ill_defined"

        return margin, spiculation_idx, lobulation_idx

    def _classify_internal(self, crop_hu: np.ndarray, mask: np.ndarray) -> str:
        """Checks for internal cavitation (air pockets < -850 HU inside nodule) or air bronchograms."""
        voxels_hu = crop_hu[mask > 0]
        air_voxels = (voxels_hu < -850.0).sum()
        air_fraction = air_voxels / len(voxels_hu)

        if air_fraction > 0.20:
            return "cavitary"
        elif air_fraction > 0.08:
            return "air_bronchogram"
        elif float(voxels_hu.std()) > 180.0:
            return "heterogeneous"
        else:
            return "homogeneous"

    def _classify_relationships(
        self,
        crop_hu: np.ndarray,
        mask: np.ndarray,
        lung_mask_crop: Optional[np.ndarray] = None
    ) -> Tuple[str, str]:
        """Assesses pleural contact and vascular abutment."""
        if ndi is None:
            return "parenchymal", "solitary"

        # Dilate nodule mask by 2 voxels to check immediate shell
        dilated = ndi.binary_dilation(mask, structure=np.ones((3, 3, 3)))
        boundary_shell = (dilated & ~mask)

        # Check pleural contact: voxels in shell that are chest wall / outside lung
        if lung_mask_crop is not None:
            non_lung_adjacent = (boundary_shell & (lung_mask_crop == 0)).sum()
            if non_lung_adjacent > 15:
                pleural = "pleural_abutment"
            elif non_lung_adjacent > 3:
                pleural = "juxta_pleural"
            else:
                pleural = "parenchymal"
        else:
            pleural = "parenchymal"

        # Check vascular contact: adjacent voxels with blood vessel density (30 to 150 HU)
        shell_hu = crop_hu[boundary_shell]
        vessel_like_voxels = ((shell_hu >= 30.0) & (shell_hu <= 150.0)).sum()
        if vessel_like_voxels > 25:
            vascular = "vessel_embedded"
        elif vessel_like_voxels > 8:
            vascular = "juxta_vascular"
        else:
            vascular = "solitary"

        return pleural, vascular

    def _default_empty(self) -> MorphologyAssessment:
        return MorphologyAssessment(
            attenuation_type="solid",
            margin_type="ill_defined",
            internal_structure="homogeneous",
            calcification_type="none",
            pleural_relationship="parenchymal",
            vascular_relationship="solitary",
            spiculation_index=0.0,
            lobulation_index=0.0,
            calcification_percentage=0.0,
            clinical_note="Insufficient nodule volume for morphological analysis."
        )
