"""
backend/models/radiomics/extractor.py
=======================================
PyRadiomics feature extraction from nodule ROI + segmentation mask.

Feature groups (IBSI-compliant):
  - firstorder:  mean, std, skewness, kurtosis, energy, entropy, ...
  - shape:       volume, diameter, sphericity, surface area, ...
  - glcm:        contrast, correlation, energy, homogeneity, ...
  - glrlm:       run-length features
  - glszm:       size-zone features
  - gldm:        grey-level dependence features

Input:
  image_nifti:  path to NIfTI CT volume (resampled, HU)
  mask_nifti:   path to NIfTI binary nodule mask

Output:
  Dict[str, float] — 107 standardised radiomic features
  (exact count depends on PyRadiomics version and settings)

After extraction, features are normalised and passed to XGBoost / RF.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import SimpleITK as sitk
    from radiomics import featureextractor, setVerbosity
    setVerbosity(60)   # suppress radiomics warnings
    HAS_RADIOMICS = True
except ImportError:
    HAS_RADIOMICS = False

logger = logging.getLogger(__name__)

# PyRadiomics configuration — all six feature groups enabled
_PARAMS = {
    "imageType":    {"Original": {}},
    "featureClass": {
        "firstorder":  [],
        "shape":       [],
        "glcm":        [],
        "glrlm":       [],
        "glszm":       [],
        "gldm":        [],
    },
    "setting": {
        "binWidth":        25,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "verbose": False,
    },
}

# Keys to EXCLUDE from the feature dict (non-numeric metadata)
_METADATA_KEYS = {
    "diagnostics_Versions_PyRadiomics",
    "diagnostics_Versions_Numpy",
    "diagnostics_Versions_SimpleITK",
    "diagnostics_Versions_PyWavelet",
    "diagnostics_Versions_Python",
}


def extract_radiomic_features(
    image_path: str | Path,
    mask_path:  str | Path,
    nodule_label: int = 1,
) -> Dict[str, float]:
    """Extract radiomic features from a nodule volume + mask.

    Args:
        image_path:   Path to NIfTI CT image (.nii / .nii.gz)
        mask_path:    Path to NIfTI binary nodule mask (label=1)
        nodule_label: Integer label for the nodule in the mask (default 1)

    Returns:
        Dict of feature_name → float value (non-numeric keys excluded)
        Empty dict if PyRadiomics is not installed or extraction fails.
    """
    if not HAS_RADIOMICS:
        logger.warning("PyRadiomics not installed. "
                       "pip install pyradiomics")
        return {}

    try:
        extractor = featureextractor.RadiomicsFeatureExtractor(_PARAMS)
        result    = extractor.execute(str(image_path), str(mask_path),
                                      label=nodule_label)
    except Exception as e:
        logger.error(f"PyRadiomics extraction failed: {e}")
        return {}

    features = {}
    for k, v in result.items():
        if k in _METADATA_KEYS or k.startswith("diagnostics_"):
            continue
        try:
            features[k] = float(v)
        except (TypeError, ValueError):
            pass   # skip non-numeric (e.g. version strings)

    logger.info(f"Extracted {len(features)} radiomic features.")
    return features


def extract_shape_metrics(mask_nifti_path: str | Path,
                           spacing_mm: tuple = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """Extract geometric shape metrics directly from a binary mask.

    Returns:
        volume_mm3, max_diameter_mm, mean_diameter_mm,
        surface_area_mm2, sphericity, bounding_box
    """
    import nibabel as nib

    img = nib.load(str(mask_nifti_path))
    mask = img.get_fdata().astype(bool)
    dz, dy, dx = spacing_mm

    voxel_vol = dz * dy * dx
    volume_mm3 = float(mask.sum() * voxel_vol)

    # Find bounding box
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return {"volume_mm3": 0.0, "max_diameter_mm": 0.0,
                "mean_diameter_mm": 0.0, "surface_area_mm2": 0.0,
                "sphericity": 0.0}

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    dims_mm = np.array([
        (z_max - z_min + 1) * dz,
        (y_max - y_min + 1) * dy,
        (x_max - x_min + 1) * dx,
    ])
    max_diameter_mm  = float(dims_mm.max())
    mean_diameter_mm = float(dims_mm.mean())

    # Sphericity: 6π^(1/3) * V^(2/3) / SA
    # Approximate SA via voxel surface count
    padded = np.pad(mask, 1)
    surface_voxels = 0
    for axis, delta in [
        ((1, 0, 0), dz), ((0, 1, 0), dy), ((0, 0, 1), dx)
    ]:
        roll_p = np.roll(padded,  1, axis=np.argmax(axis))
        roll_n = np.roll(padded, -1, axis=np.argmax(axis))
        surface_voxels += int(((padded & ~roll_p) | (padded & ~roll_n)).sum())

    # Approximate surface area
    surface_area_mm2 = float(surface_voxels * min(dy*dx, dz*dx, dz*dy))

    if surface_area_mm2 > 0 and volume_mm3 > 0:
        import math
        sphericity = (math.pi ** (1/3)) * (6 * volume_mm3) ** (2/3) / surface_area_mm2
    else:
        sphericity = 0.0

    return {
        "volume_mm3":        volume_mm3,
        "max_diameter_mm":   max_diameter_mm,
        "mean_diameter_mm":  mean_diameter_mm,
        "surface_area_mm2":  surface_area_mm2,
        "sphericity":        float(np.clip(sphericity, 0, 1)),
        "bbox_z_min": int(z_min), "bbox_z_max": int(z_max),
        "bbox_y_min": int(y_min), "bbox_y_max": int(y_max),
        "bbox_x_min": int(x_min), "bbox_x_max": int(x_max),
    }


def features_to_vector(feature_dict: Dict[str, float],
                        feature_names: Optional[list] = None) -> np.ndarray:
    """Convert feature dict to sorted numpy array.

    Args:
        feature_dict:  Output of extract_radiomic_features()
        feature_names: Optional list to enforce ordering.

    Returns:
        (N,) float32 feature vector
    """
    if feature_names is not None:
        return np.array([feature_dict.get(k, 0.0)
                         for k in feature_names], dtype=np.float32)
    return np.array(list(feature_dict.values()), dtype=np.float32)
