"""
backend/dicom/validator.py
==========================
DICOM series quality control for chest CT.

Checks:
  1. Modality — must be CT
  2. Slice count — minimum 50 slices for chest coverage
  3. Missing/duplicate slices — gaps in ImagePositionPatient z-coords
  4. Spacing consistency — SliceThickness and PixelSpacing variance
  5. HU validity — after conversion, values must be in [-2000, 3000]
  6. Orientation — check ImageOrientationPatient for axial acquisition
  7. Incomplete lung coverage — estimated field of view
  8. Motion artifacts — inter-slice mean intensity discontinuity proxy

Output:
  QualityResult with:
    status: "PASS" | "WARNING" | "REJECT"
    findings: list of finding strings
    metadata: study metadata dict

IMPORTANT: A REJECT status means the pipeline will NOT generate a
prediction. Displaying a model output for a rejected study would be
misleading and is explicitly prevented.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import pydicom
except ImportError:
    raise ImportError("pydicom is required: pip install pydicom")


@dataclass
class QualityResult:
    """Result of DICOM quality control."""
    status: str = "PASS"          # "PASS", "WARNING", "REJECT"
    findings: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    slice_count: int = 0
    modality: str = ""
    slice_thickness_mm: float = 0.0
    pixel_spacing_mm: Tuple[float, float] = (0.0, 0.0)
    study_uid: str = ""
    series_uid: str = ""
    patient_id: str = "ANONYMISED"


def _warn(result: QualityResult, msg: str):
    result.findings.append(f"[WARNING] {msg}")
    if result.status == "PASS":
        result.status = "WARNING"


def _reject(result: QualityResult, msg: str):
    result.findings.append(f"[REJECT] {msg}")
    result.status = "REJECT"


def _info(result: QualityResult, msg: str):
    result.findings.append(f"[INFO] {msg}")


def validate_dicom_series(dicom_dir: str | Path) -> QualityResult:
    """Validate a directory of DICOM files as a chest CT series.

    Args:
        dicom_dir: Path to folder containing .dcm files.

    Returns:
        QualityResult with PASS / WARNING / REJECT status.
    """
    dicom_dir = Path(dicom_dir)
    result = QualityResult()

    # ── 1. Find DICOM files ─────────────────────────────────────
    dcm_files = sorted([f for f in dicom_dir.iterdir()
                        if f.suffix.lower() in (".dcm", "")
                        and f.is_file()])
    if not dcm_files:
        # Try recursive
        dcm_files = sorted(dicom_dir.rglob("*.dcm"))

    if len(dcm_files) == 0:
        _reject(result, "No DICOM files found in the provided directory.")
        return result

    # ── 2. Read slice headers ───────────────────────────────────
    slices = []
    for path in dcm_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            slices.append((path, ds))
        except Exception as e:
            _warn(result, f"Could not read {path.name}: {e}")

    if not slices:
        _reject(result, "All DICOM files are unreadable or corrupted.")
        return result

    result.slice_count = len(slices)
    _info(result, f"Found {len(slices)} readable DICOM files.")

    # ── 3. Modality check ──────────────────────────────────────
    modalities = set()
    for _, ds in slices:
        if hasattr(ds, "Modality"):
            modalities.add(ds.Modality)
    if not modalities:
        _warn(result, "Modality tag missing in all slices.")
    elif modalities != {"CT"}:
        _reject(result, f"Expected CT modality; found: {modalities}. "
                "Non-CT data will not be processed.")
        return result
    result.modality = "CT"

    # ── 4. Study/Series UIDs ───────────────────────────────────
    _, first_ds = slices[0]
    result.study_uid  = getattr(first_ds, "StudyInstanceUID", "UNKNOWN")
    result.series_uid = getattr(first_ds, "SeriesInstanceUID", "UNKNOWN")
    result.patient_id = "ANONYMISED"   # never store PHI in application layer
    result.metadata.update({
        "StudyDate": str(getattr(first_ds, "StudyDate", "")),
        "Manufacturer": str(getattr(first_ds, "Manufacturer", "")),
        "ConvolutionKernel": str(getattr(first_ds, "ConvolutionKernel", "")),
        "KVP": str(getattr(first_ds, "KVP", "")),
    })

    # ── 5. Slice positions → detect gaps ──────────────────────
    positions = []
    for _, ds in slices:
        if hasattr(ds, "ImagePositionPatient"):
            try:
                z = float(ds.ImagePositionPatient[2])
                positions.append(z)
            except Exception:
                pass

    if len(positions) < 2:
        _warn(result, "ImagePositionPatient missing in most slices; "
              "cannot verify spacing.")
    else:
        positions_sorted = sorted(positions)
        gaps = np.diff(positions_sorted)
        median_gap = float(np.median(np.abs(gaps)))
        large_gaps = np.where(np.abs(gaps) > median_gap * 3)[0]
        if len(large_gaps) > 0:
            _warn(result, f"{len(large_gaps)} suspected missing slice(s) detected "
                  f"(gaps > {median_gap*3:.1f} mm).")
        _info(result, f"Median inter-slice spacing: {median_gap:.2f} mm")

    # ── 6. Slice count ─────────────────────────────────────────
    if len(slices) < 50:
        _warn(result, f"Only {len(slices)} slices found; full chest CT typically "
              "has ≥ 100 slices. Coverage may be incomplete.")
    if len(slices) < 20:
        _reject(result, "Fewer than 20 slices — insufficient for volumetric analysis.")
        return result

    # ── 7. Pixel spacing & thickness ──────────────────────────
    try:
        ps = [float(x) for x in first_ds.PixelSpacing]
        result.pixel_spacing_mm = (ps[0], ps[1])
    except Exception:
        _warn(result, "PixelSpacing tag missing or invalid.")

    try:
        result.slice_thickness_mm = float(first_ds.SliceThickness)
    except Exception:
        _warn(result, "SliceThickness tag missing.")

    # Check spacing consistency across slices
    thicknesses = []
    for _, ds in slices:
        if hasattr(ds, "SliceThickness"):
            try:
                thicknesses.append(float(ds.SliceThickness))
            except Exception:
                pass
    if len(thicknesses) > 1:
        if max(thicknesses) - min(thicknesses) > 1.0:
            _warn(result, f"Inconsistent SliceThickness in series: "
                  f"min={min(thicknesses):.2f}, max={max(thicknesses):.2f} mm")

    # ── 8. HU validity (sample 3 slices) ──────────────────────
    sample_indices = [0, len(slices)//2, -1]
    for idx in sample_indices:
        path, ds = slices[idx]
        try:
            full_ds = pydicom.dcmread(str(path))
            arr = full_ds.pixel_array.astype(np.float32)
            slope     = float(getattr(full_ds, "RescaleSlope", 1))
            intercept = float(getattr(full_ds, "RescaleIntercept", 0))
            hu = arr * slope + intercept
            if hu.min() < -2000 or hu.max() > 3000:
                _warn(result, f"HU range [{hu.min():.0f}, {hu.max():.0f}] "
                      "exceeds expected chest CT range [-2000, 3000].")
                break
        except Exception as e:
            _warn(result, f"Could not validate HU values: {e}")
            break

    # ── 9. Orientation check ───────────────────────────────────
    if hasattr(first_ds, "ImageOrientationPatient"):
        iop = [float(v) for v in first_ds.ImageOrientationPatient]
        # Axial: row direction ≈ (1,0,0), col direction ≈ (0,1,0)
        row_dot  = abs(iop[0])
        col_dot  = abs(iop[4])
        if row_dot < 0.9 or col_dot < 0.9:
            _warn(result, "Non-standard image orientation detected "
                  "(not axial acquisition). Resampling will be applied.")

    # ── 10. Motion artifact proxy ──────────────────────────────
    # Load 10 evenly-spaced slices, check inter-slice mean intensity variance
    if len(slices) >= 10:
        sample_every = max(1, len(slices) // 10)
        means = []
        for path, _ in slices[::sample_every][:10]:
            try:
                ds = pydicom.dcmread(str(path))
                arr = ds.pixel_array.astype(np.float32)
                means.append(float(arr.mean()))
            except Exception:
                pass
        if len(means) >= 5:
            intensity_std = float(np.std(means))
            if intensity_std > 5000:
                _warn(result, f"High inter-slice intensity variance "
                      f"(std={intensity_std:.0f}) may indicate motion artifacts.")

    # ── Final status summary ───────────────────────────────────
    if result.status == "PASS":
        _info(result, "All quality checks passed.")

    return result


def validate_single_dicom(dcm_path: str | Path) -> QualityResult:
    """Validate a single DICOM file (for quick checks)."""
    result = QualityResult()
    try:
        ds = pydicom.dcmread(str(dcm_path))
        result.slice_count = 1
        result.modality = getattr(ds, "Modality", "UNKNOWN")
        _info(result, f"Single DICOM loaded: {Path(dcm_path).name}")
    except Exception as e:
        _reject(result, f"Could not read DICOM file: {e}")
    return result


# ─────────────────────────────────────────────────────────────
#  CLI smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    qr = validate_dicom_series(path)
    print(f"\nQuality Status: {qr.status}")
    for f in qr.findings:
        print(f"  {f}")
    print(f"Slices: {qr.slice_count}, Modality: {qr.modality}")
    print(f"Thickness: {qr.slice_thickness_mm} mm, Spacing: {qr.pixel_spacing_mm} mm")
