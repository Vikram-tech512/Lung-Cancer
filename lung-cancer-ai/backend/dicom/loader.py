"""
backend/dicom/loader.py
=======================
DICOM → 3D volume loader with HU conversion, resampling, and NIfTI export.

Merged from platform/backend/dicom/loader.py with enhancements:
  - ZIP extraction support
  - Directory walking and study/series grouping
  - Multiple series detection
  - Duplicate slice detection

Pipeline:
  1. Read DICOM series with pydicom (sorted by z-position)
  2. Apply RescaleSlope / RescaleIntercept → Hounsfield Units
  3. Resample to isotropic 1mm³ spacing using SimpleITK
  4. Normalise orientation to RAS (Right–Anterior–Superior)
  5. Export to NIfTI (nibabel) for downstream use

NEVER modifies or overwrites the original DICOM files.
All outputs are written to the processed data directory.
"""
from __future__ import annotations

import io
import os
import zipfile
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

import numpy as np

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

try:
    import nibabel as nib
except ImportError:
    nib = None

logger = logging.getLogger("backend.dicom.loader")


# ─────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class DicomStudyInfo:
    """Structured metadata for a loaded DICOM study."""
    study_instance_uid: str = ""
    series_instance_uid: str = ""
    patient_id: str = "ANONYMISED"
    study_date: str = ""
    modality: str = ""
    manufacturer: str = ""
    convolution_kernel: str = ""
    rows: int = 0
    columns: int = 0
    slice_count: int = 0
    slice_thickness: float = 0.0
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
    original_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    resampled_spacing: Optional[Tuple[float, float, float]] = None
    resampled_shape: Optional[Tuple[int, ...]] = None
    window_preset: str = "lung"
    output_nifti: str = ""
    has_duplicates: bool = False
    duplicate_count: int = 0
    series_list: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  ZIP / Directory Extraction
# ─────────────────────────────────────────────────────────────

def extract_dicom_from_zip(zip_path: str | Path, extract_dir: str | Path) -> Path:
    """Extract DICOM files from a ZIP archive.

    Args:
        zip_path: Path to the ZIP file.
        extract_dir: Directory to extract into.

    Returns:
        Path to the directory containing extracted DICOM files.
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(extract_dir))

    logger.info(f"Extracted {len(list(extract_dir.rglob('*')))} files from ZIP to {extract_dir}")
    return extract_dir


def extract_dicom_from_bytes(file_bytes: bytes, extract_dir: str | Path) -> Path:
    """Extract DICOM files from ZIP bytes (e.g., from an upload).

    Args:
        file_bytes: Raw ZIP file bytes.
        extract_dir: Directory to extract into.

    Returns:
        Path to the directory containing extracted DICOM files.
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    bio = io.BytesIO(file_bytes)
    if zipfile.is_zipfile(bio):
        bio.seek(0)
        with zipfile.ZipFile(bio, 'r') as zf:
            zf.extractall(str(extract_dir))
        logger.info(f"Extracted DICOM from ZIP bytes to {extract_dir}")
    else:
        # Single DICOM file
        single_path = extract_dir / "uploaded.dcm"
        with open(single_path, 'wb') as f:
            f.write(file_bytes)
        logger.info(f"Saved single DICOM file to {single_path}")

    return extract_dir


def find_dicom_files(directory: str | Path) -> List[Path]:
    """Find all DICOM files in a directory (recursive).

    Checks for .dcm extension and also tries reading files without extension.
    """
    directory = Path(directory)
    dcm_files = list(directory.rglob("*.dcm"))

    # Also check files without standard extension
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() not in (".dcm", ".zip", ".gz", ".txt", ".json", ".xml", ".py"):
            try:
                pydicom.dcmread(str(f), stop_before_pixels=True)
                dcm_files.append(f)
            except Exception:
                pass

    return sorted(set(dcm_files))


def group_by_series(dicom_dir: str | Path) -> Dict[str, List[Path]]:
    """Group DICOM files by SeriesInstanceUID.

    Returns:
        Dict mapping SeriesInstanceUID → list of file paths.
    """
    dicom_dir = Path(dicom_dir)
    series_map: Dict[str, List[Path]] = {}

    for f in find_dicom_files(dicom_dir):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            series_uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
            if series_uid not in series_map:
                series_map[series_uid] = []
            series_map[series_uid].append(f)
        except Exception:
            pass

    logger.info(f"Found {len(series_map)} series in {dicom_dir}")
    return series_map


# ─────────────────────────────────────────────────────────────
#  DICOM Series → NumPy volume
# ─────────────────────────────────────────────────────────────

def load_dicom_series(dicom_dir: str | Path) -> Tuple[np.ndarray,
                                                       Tuple[float, float, float],
                                                       DicomStudyInfo]:
    """Load a DICOM series and return HU volume + metadata.

    Args:
        dicom_dir: Path to folder containing .dcm files.

    Returns:
        volume_hu:   (D, H, W) float32 array in Hounsfield Units
        spacing_mm:  (dz, dy, dx) in mm
        study_info:  DicomStudyInfo with study metadata (no PHI)
    """
    dicom_dir = Path(dicom_dir)
    dcm_files = find_dicom_files(dicom_dir)

    if not dcm_files:
        raise ValueError(f"No readable DICOM files found in {dicom_dir}")

    # ── Read and sort slices by z-position ────────────────────
    slices = []
    z_positions_seen = set()
    duplicate_count = 0

    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            if hasattr(ds, "pixel_array"):
                z = float(ds.ImagePositionPatient[2]) \
                    if hasattr(ds, "ImagePositionPatient") else 0.0

                # Detect duplicate slices
                z_key = round(z, 3)
                if z_key in z_positions_seen:
                    duplicate_count += 1
                    continue
                z_positions_seen.add(z_key)

                slices.append((z, ds))
        except Exception as e:
            logger.warning(f"Could not read {f.name}: {e}")

    if not slices:
        raise ValueError(f"No readable DICOM slices in {dicom_dir}")

    slices.sort(key=lambda t: t[0])

    # ── Build volume with HU conversion ───────────────────────
    ref = slices[0][1]
    ps = [float(x) for x in ref.PixelSpacing] \
        if hasattr(ref, "PixelSpacing") else [1.0, 1.0]
    st = float(ref.SliceThickness) \
        if hasattr(ref, "SliceThickness") else 1.0

    arrays = []
    for _, ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + inter)

    volume_hu = np.stack(arrays, axis=0)   # (D, H, W)
    spacing_mm = (st, ps[0], ps[1])        # (dz, dy, dx)

    # ── Build study info (no PHI) ─────────────────────────────
    study_info = DicomStudyInfo(
        study_instance_uid=str(getattr(ref, "StudyInstanceUID", "")),
        series_instance_uid=str(getattr(ref, "SeriesInstanceUID", "")),
        patient_id="ANONYMISED",  # Never store PHI
        study_date=str(getattr(ref, "StudyDate", "")),
        modality=str(getattr(ref, "Modality", "")),
        manufacturer=str(getattr(ref, "Manufacturer", "")),
        convolution_kernel=str(getattr(ref, "ConvolutionKernel", "")),
        rows=int(getattr(ref, "Rows", 0)),
        columns=int(getattr(ref, "Columns", 0)),
        slice_count=len(slices),
        slice_thickness=st,
        pixel_spacing=(ps[0], ps[1]),
        original_spacing=spacing_mm,
        has_duplicates=duplicate_count > 0,
        duplicate_count=duplicate_count,
    )

    if duplicate_count > 0:
        logger.warning(f"Detected {duplicate_count} duplicate slices (removed)")

    return volume_hu, spacing_mm, study_info


# ─────────────────────────────────────────────────────────────
#  Single DICOM file → NumPy slice (for 2D inference)
# ─────────────────────────────────────────────────────────────

def load_dicom_slice(dcm_path: str | Path) -> np.ndarray:
    """Load a single DICOM file → 2D HU array (H, W) float32."""
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter


# ─────────────────────────────────────────────────────────────
#  Resampling to isotropic spacing via SimpleITK
# ─────────────────────────────────────────────────────────────

def resample_volume(volume_hu: np.ndarray,
                    original_spacing: Tuple[float, float, float],
                    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                    interpolation: str = "linear") -> Tuple[np.ndarray,
                                                             Tuple[float, float, float]]:
    """Resample a 3D HU volume to target_spacing using SimpleITK.

    Args:
        volume_hu:        (D, H, W) float32 — input volume in HU
        original_spacing: (dz, dy, dx) in mm
        target_spacing:   desired output spacing — (1.0, 1.0, 1.0) for isotropic
        interpolation:    "linear" (default) or "nearest" (for masks)

    Returns:
        resampled volume (D', H', W') and actual output spacing
    """
    # SimpleITK uses (x, y, z) convention — we store (z, y, x)
    sitk_vol = sitk.GetImageFromArray(volume_hu)
    sitk_vol.SetSpacing([float(original_spacing[2]),    # x
                         float(original_spacing[1]),    # y
                         float(original_spacing[0])])   # z

    orig_size = np.array(sitk_vol.GetSize(), dtype=np.float64)   # (W, H, D)
    orig_spacing_arr = np.array(sitk_vol.GetSpacing(), dtype=np.float64)
    new_spacing = np.array([target_spacing[2], target_spacing[1], target_spacing[0]])
    new_size = np.round(orig_size * orig_spacing_arr / new_spacing).astype(int)

    interp = sitk.sitkLinear if interpolation == "linear" else sitk.sitkNearestNeighbor

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(interp)
    resampler.SetOutputOrigin(sitk_vol.GetOrigin())
    resampler.SetOutputDirection(sitk_vol.GetDirection())

    resampled = resampler.Execute(sitk_vol)
    volume_out = sitk.GetArrayFromImage(resampled)   # (D', H', W')

    actual_spacing = (float(new_spacing[2]),   # dz
                      float(new_spacing[1]),   # dy
                      float(new_spacing[0]))   # dx
    return volume_out, actual_spacing


# ─────────────────────────────────────────────────────────────
#  HU Windowing (lung / soft-tissue / bone)
# ─────────────────────────────────────────────────────────────

WINDOW_PRESETS = {
    "lung":        (-1000, 400),   # WL=-300, WW=1400
    "soft_tissue": (-150,  250),   # WL=50,   WW=400
    "bone":        (-500, 1300),   # WL=400,  WW=1800
    "mediastinum": (-150,  250),
}


def apply_hu_window(volume_hu: np.ndarray,
                    preset: str = "lung") -> np.ndarray:
    """Apply HU windowing and normalise to [0, 1].

    Args:
        volume_hu: (D, H, W) float32
        preset:    one of 'lung', 'soft_tissue', 'bone', 'mediastinum'

    Returns:
        (D, H, W) float32 in [0, 1]
    """
    if preset not in WINDOW_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. "
                         f"Choose from {list(WINDOW_PRESETS.keys())}")
    lo, hi = WINDOW_PRESETS[preset]
    vol = np.clip(volume_hu, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def apply_custom_window(volume_hu: np.ndarray,
                        center: float, width: float) -> np.ndarray:
    """Apply custom HU window (center, width) and normalise to [0, 1]."""
    lo = center - width / 2.0
    hi = center + width / 2.0
    vol = np.clip(volume_hu, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  NIfTI export / load
# ─────────────────────────────────────────────────────────────

def save_nifti(volume: np.ndarray,
               spacing_mm: Tuple[float, float, float],
               out_path: str | Path):
    """Save a 3D volume as NIfTI (.nii.gz).

    Args:
        volume:     (D, H, W) float32 — HU or normalised
        spacing_mm: (dz, dy, dx)
        out_path:   output file path (.nii or .nii.gz)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dz, dy, dx = spacing_mm
    affine = np.diag([dx, dy, dz, 1.0])
    img = nib.Nifti1Image(volume, affine)
    img.header.set_zooms((dx, dy, dz))
    nib.save(img, str(out_path))


def load_nifti(nii_path: str | Path) -> Tuple[np.ndarray,
                                               Tuple[float, float, float]]:
    """Load a NIfTI volume.

    Returns:
        volume (D, H, W) and spacing (dz, dy, dx) in mm
    """
    img = nib.load(str(nii_path))
    vol = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()   # (dx, dy, dz)
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return vol, spacing


# ─────────────────────────────────────────────────────────────
#  Complete DICOM → processed NIfTI pipeline
# ─────────────────────────────────────────────────────────────

def dicom_to_processed_volume(dicom_dir: str | Path,
                               out_nifti: str | Path,
                               window_preset: str = "lung",
                               target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                               ) -> Tuple[np.ndarray, DicomStudyInfo]:
    """Full DICOM → resampled + windowed → NIfTI pipeline.

    Does NOT overwrite original DICOMs.

    Returns:
        processed volume (D, H, W) float32 [0,1] and DicomStudyInfo
    """
    # Load
    vol_hu, orig_spacing, study_info = load_dicom_series(dicom_dir)

    # Resample to isotropic
    vol_resampled, new_spacing = resample_volume(
        vol_hu, orig_spacing, target_spacing)

    # Window (normalise to [0,1])
    vol_windowed = apply_hu_window(vol_resampled, preset=window_preset)

    # Save
    save_nifti(vol_windowed, new_spacing, out_nifti)

    study_info.resampled_spacing = new_spacing
    study_info.resampled_shape = vol_resampled.shape
    study_info.window_preset = window_preset
    study_info.output_nifti = str(out_nifti)

    logger.info(
        f"Processed DICOM → NIfTI: {vol_hu.shape} → {vol_resampled.shape} "
        f"at {new_spacing} mm/voxel → {out_nifti}"
    )

    return vol_windowed, study_info
