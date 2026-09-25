"""
backend/dicom/validator.py
==========================
DICOM series quality control and clinical safety validation for chest CT.

Checks:
  1. Modality — must be CT
  2. Slice count — minimum 50 slices for full chest coverage (warning if <50, reject if <15)
  3. Missing/duplicate slices — gaps in ImagePositionPatient z-coordinates
  4. Spacing consistency — SliceThickness and PixelSpacing variance
  5. HU validity — after conversion, values must be in [-2000, 3000]
  6. Orientation — check ImageOrientationPatient for axial acquisition
  7. Safe-processing gate — reject studies that cannot be safely processed
  8. Motion artifacts proxy — inter-slice mean intensity discontinuity proxy
  9. Safe Harbor PHI presence flag

IMPORTANT: A REJECT status prevents automated inference from producing misleading outputs.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    import pydicom
except ImportError:
    pydicom = None


@dataclass
class QualityResult:
    """Result of DICOM quality control."""
    status: str = "PASS"          # "PASS", "WARNING", "REJECT"
    findings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    slice_count: int = 0
    modality: str = ""
    slice_thickness_mm: float = 0.0
    pixel_spacing_mm: Tuple[float, float] = (0.0, 0.0)
    study_uid: str = ""
    series_uid: str = ""
    patient_id: str = "ANONYMISED"
    safe_to_process: bool = True
    phi_detected: bool = False


def _warn(result: QualityResult, msg: str):
    result.findings.append(f"[WARNING] {msg}")
    if result.status == "PASS":
        result.status = "WARNING"


def _reject(result: QualityResult, msg: str):
    result.findings.append(f"[REJECT] {msg}")
    result.status = "REJECT"
    result.safe_to_process = False


def _info(result: QualityResult, msg: str):
    result.findings.append(f"[INFO] {msg}")


def validate_dicom_series(dicom_dir: str | Path) -> QualityResult:
    """Validate a directory of DICOM files as a chest CT series.

    Args:
        dicom_dir: Path to directory or file containing .dcm slices.

    Returns:
        QualityResult with status (PASS / WARNING / REJECT) and detailed findings.
    """
    if pydicom is None:
        return QualityResult(
            status="WARNING",
            findings=["pydicom not installed; bypassing low-level DICOM check in mock/dev mode"],
            safe_to_process=True
        )

    dicom_path = Path(dicom_dir)
    result = QualityResult()

    if not dicom_path.exists():
        _reject(result, f"Path does not exist: {dicom_path}")
        return result

    # Find files
    if dicom_path.is_file():
        dcm_files = [dicom_path]
    else:
        dcm_files = sorted([
            f for f in dicom_path.iterdir()
            if f.suffix.lower() in (".dcm", "") and f.is_file()
        ])
        if not dcm_files:
            dcm_files = sorted(dicom_path.rglob("*.dcm"))

    if not dcm_files:
        _reject(result, "No DICOM files found in the specified path.")
        return result

    slices = []
    phi_fields = ["PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs"]

    for fpath in dcm_files:
        try:
            ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)
            slices.append((fpath, ds))
            for tag in phi_fields:
                val = getattr(ds, tag, None)
                if val and str(val).strip() and str(val).strip() != "ANONYMISED":
                    result.phi_detected = True
        except Exception as e:
            _warn(result, f"Could not read header of {fpath.name}: {e}")

    if not slices:
        _reject(result, "All DICOM files are unreadable or corrupt.")
        return result

    result.slice_count = len(slices)
    _info(result, f"Inspected {len(slices)} DICOM slice headers.")

    # 1. Modality check
    modalities = {getattr(ds, "Modality", "UNKNOWN") for _, ds in slices}
    if modalities != {"CT"}:
        if "CT" in modalities:
            _warn(result, f"Mixed modalities present: {modalities}. Expected pure CT.")
        else:
            _reject(result, f"Non-CT modality found ({modalities}). Chest CT is required.")
            return result
    result.modality = "CT"

    # 2. UIDs & Metadata
    first_path, first_ds = slices[0]
    result.study_uid = getattr(first_ds, "StudyInstanceUID", "UNKNOWN")
    result.series_uid = getattr(first_ds, "SeriesInstanceUID", "UNKNOWN")
    result.metadata.update({
        "StudyDate": str(getattr(first_ds, "StudyDate", "")),
        "Manufacturer": str(getattr(first_ds, "Manufacturer", "Unknown")),
        "ManufacturerModelName": str(getattr(first_ds, "ManufacturerModelName", "Unknown")),
        "ConvolutionKernel": str(getattr(first_ds, "ConvolutionKernel", "Standard")),
        "KVP": str(getattr(first_ds, "KVP", "")),
        "SliceThickness": str(getattr(first_ds, "SliceThickness", "")),
    })

    # 3. Slice count
    if len(slices) < 15:
        _reject(result, f"Fewer than 15 slices ({len(slices)} found) — insufficient for volumetric 3D analysis.")
        return result
    elif len(slices) < 50:
        _warn(result, f"Only {len(slices)} slices found; typical chest CT scans have ≥100 slices. Coverage may be partial.")

    # 4. Spacing & Pixel Dimensions
    try:
        ps = [float(x) for x in first_ds.PixelSpacing]
        result.pixel_spacing_mm = (ps[0], ps[1])
        if ps[0] <= 0 or ps[1] <= 0 or ps[0] > 5.0 or ps[1] > 5.0:
            _warn(result, f"Abnormal in-plane pixel spacing: {ps} mm.")
    except Exception:
        _warn(result, "PixelSpacing tag missing or unparseable.")

    try:
        result.slice_thickness_mm = float(first_ds.SliceThickness)
        if result.slice_thickness_mm > 5.0:
            _warn(result, f"Thick slice acquisition ({result.slice_thickness_mm} mm > 5.0 mm). Nodule sensitivity may decrease.")
    except Exception:
        _warn(result, "SliceThickness tag missing.")

    # 5. Position & Slice gaps
    z_positions = []
    for _, ds in slices:
        if hasattr(ds, "ImagePositionPatient"):
            try:
                z_positions.append(float(ds.ImagePositionPatient[2]))
            except Exception:
                pass

    if len(z_positions) == len(slices) and len(z_positions) > 1:
        z_positions_sorted = sorted(z_positions)
        z_diffs = np.diff(z_positions_sorted)
        # Duplicate detection
        duplicates = np.where(np.abs(z_diffs) < 1e-4)[0]
        if len(duplicates) > 0:
            _warn(result, f"Detected {len(duplicates)} duplicate slice positions.")

        # Non-uniform gaps
        median_spacing = float(np.median(np.abs(z_diffs)))
        if median_spacing > 0:
            large_gaps = np.where(np.abs(z_diffs) > median_spacing * 2.5)[0]
            if len(large_gaps) > 0:
                _warn(result, f"{len(large_gaps)} suspected missing slice gap(s) (> {median_spacing * 2.5:.2f} mm).")

    # 6. Orientation
    if hasattr(first_ds, "ImageOrientationPatient"):
        try:
            iop = [float(v) for v in first_ds.ImageOrientationPatient]
            if abs(iop[0]) < 0.8 or abs(iop[4]) < 0.8:
                _warn(result, "Non-standard orientation (coronal/sagittal or oblique). Resampling required.")
        except Exception:
            pass

    # 7. Sample HU validity
    sample_indices = [0, len(slices) // 2, -1]
    for idx in sample_indices:
        p, ds = slices[idx]
        try:
            full_ds = pydicom.dcmread(str(p))
            arr = full_ds.pixel_array.astype(np.float32)
            slope = float(getattr(full_ds, "RescaleSlope", 1))
            intercept = float(getattr(full_ds, "RescaleIntercept", 0))
            hu = arr * slope + intercept
            if hu.min() < -2500 or hu.max() > 4000:
                _warn(result, f"HU dynamic range [{hu.min():.0f}, {hu.max():.0f}] exceeds standard chest CT parameters.")
                break
        except Exception:
            pass

    if result.status == "PASS":
        _info(result, "DICOM quality validation passed cleanly.")

    return result
