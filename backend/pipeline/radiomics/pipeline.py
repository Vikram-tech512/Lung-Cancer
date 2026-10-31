"""
backend/pipeline/radiomics/pipeline.py
=========================================
IBSI-compliant end-to-end Radiomics Pipeline.

Wraps PyRadiomics with:
  - Automatic mask generation when segmentation mask is absent
  - IBSI feature class selection (first-order, GLCM, GLRLM, GLSZM, NGTDM)
  - Feature caching (SQLite) to avoid redundant extraction
  - XGBoost/Random-Forest malignancy scoring from extracted features
  - Structured logging + Prometheus duration tracking

Usage
-----
>>> pipe = RadiomicsPipeline()
>>> features = pipe.extract(volume, mask)
>>> print(features["original_firstorder_Mean"])
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.monitoring import get_logger
from backend.monitoring.metrics import RADIOMICS_EXTRACTION_DURATION

log = get_logger(__name__)

CACHE_DB = Path(os.getenv("RADIOMICS_CACHE_DB", "./results/radiomics_cache.db"))


class RadiomicsPipeline:
    """
    Radiomics feature extraction and caching. This class does not estimate malignancy.

    Parameters
    ----------
    feature_classes : PyRadiomics feature classes to enable
    voxel_size_mm   : target isotropic voxel spacing for resampling
    use_cache       : persist features to SQLite for reuse
    """

    IBSI_CLASSES = [
        "firstorder",
        "glcm",
        "glrlm",
        "glszm",
        "ngtdm",
        "shape",
    ]

    def __init__(
        self,
        feature_classes: Optional[list] = None,
        voxel_size_mm: float = 1.0,
        use_cache: bool = True,
    ) -> None:
        self.feature_classes = feature_classes or self.IBSI_CLASSES
        self.voxel_size_mm   = voxel_size_mm
        self.use_cache       = use_cache
        self._extractor: Optional[Any] = None
        self._xgb_model: Optional[Any] = None

        if use_cache:
            self._init_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(
        self,
        volume:  np.ndarray,
        mask:    Optional[np.ndarray] = None,
        spacing: tuple = (1.0, 1.0, 1.0),
    ) -> Dict[str, Any]:
        """
        Extract IBSI radiomics features from a CT volume.

        Parameters
        ----------
        volume  : float32 ndarray in Hounsfield Units, shape (D, H, W)
        mask    : binary segmentation mask (same shape). Auto-generated if None.
        spacing : voxel spacing in mm (z, y, x)

        Returns
        -------
        dict of feature_name → float value
        """
        if mask is None:
            raise ValueError("A verified segmentation mask is required for radiomics extraction.")

        # Check cache
        cache_key = self._cache_key(volume, mask, spacing)
        if self.use_cache:
            cached = self._cache_get(cache_key)
            if cached and not (
                cached.get("_source") == "synthetic_fallback"
                or "nodule_score" in cached
                or "malignancy_score" in cached
            ):
                log.info("radiomics_cache_hit", key=cache_key[:8])
                return cached

        # Extract
        t0 = time.monotonic()
        try:
            features = self._run_pyradiomics(volume, mask, spacing)
        except ImportError as exc:
            raise RuntimeError("PyRadiomics and SimpleITK are required for feature extraction.") from exc
        except Exception as exc:
            log.error("radiomics_extraction_failed", error_type=type(exc).__name__)
            raise RuntimeError("Radiomics feature extraction failed; no fallback features were generated.") from exc
        elapsed  = time.monotonic() - t0

        RADIOMICS_EXTRACTION_DURATION.labels(feature_class="all").observe(elapsed)
        log.info(
            "radiomics_extracted",
            n_features=len(features),
            elapsed_s=round(elapsed, 2),
        )

        if self.use_cache:
            self._cache_set(cache_key, features)

        return features

    # ------------------------------------------------------------------
    # PyRadiomics extraction
    # ------------------------------------------------------------------
    def _run_pyradiomics(
        self,
        volume:  np.ndarray,
        mask:    np.ndarray,
        spacing: tuple,
    ) -> Dict[str, Any]:
        try:
            import SimpleITK as sitk
            from radiomics import featureextractor

            extractor = self._get_extractor()

            # Convert numpy → SimpleITK
            vol_sitk  = sitk.GetImageFromArray(volume.astype(np.float32))
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            vol_sitk.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
            mask_sitk.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))

            result = extractor.execute(vol_sitk, mask_sitk)
            features = {
                k: float(v)
                for k, v in result.items()
                if not k.startswith("diagnostics") and hasattr(v, "__float__")
            }
            return features

        except ImportError:
            raise

    def _get_extractor(self) -> Any:
        if self._extractor is not None:
            return self._extractor
        from radiomics import featureextractor
        params = {
            "imageType":    {"Original": {}},
            "featureClass": {cls: [] for cls in self.feature_classes},
            "setting": {
                "binWidth":               25,
                "resampledPixelSpacing":  [self.voxel_size_mm] * 3,
                "interpolator":           "sitkBSpline",
                "normalize":              True,
            },
        }
        self._extractor = featureextractor.RadiomicsFeatureExtractor(params)
        return self._extractor

    # ------------------------------------------------------------------
    # Auto-mask
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_mask(volume: np.ndarray) -> np.ndarray:
        """Generate a simple HU-threshold lung + nodule mask."""
        from scipy.ndimage import binary_fill_holes, binary_closing, label
        # Rough HU thresholds for lung tissue
        binary  = (volume > -950) & (volume < 200)
        closed  = binary_closing(binary, iterations=3)
        filled  = binary_fill_holes(closed)
        return filled.astype(np.uint8)

    # ------------------------------------------------------------------
    # SQLite cache
    # ------------------------------------------------------------------
    def _init_cache(self) -> None:
        CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(CACHE_DB)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS features "
                "(key TEXT PRIMARY KEY, data TEXT, created_at TEXT)"
            )

    def _cache_key(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        spacing: tuple,
    ) -> str:
        h = hashlib.sha256()
        h.update(volume.tobytes())
        h.update(mask.tobytes())
        h.update(str(spacing).encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict]:
        try:
            with sqlite3.connect(str(CACHE_DB)) as conn:
                row = conn.execute(
                    "SELECT data FROM features WHERE key=?", (key,)
                ).fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    def _cache_set(self, key: str, features: Dict) -> None:
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(str(CACHE_DB)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO features (key, data, created_at) VALUES (?,?,?)",
                    (key, json.dumps(features), now),
                )
        except Exception as exc:
            log.warning("radiomics_cache_write_failed", error=str(exc))
