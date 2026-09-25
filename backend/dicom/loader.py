"""
backend/dicom/loader.py
=======================
DICOM → 3D volume loader with HU conversion, resampling, and NIfTI export.

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

import os
from pathlib import Path
from typing import Optional, Tuple

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


# ─────────────────────────────────────────────────────────────
#  DICOM Series → NumPy volume
# ─────────────────────────────────────────────────────────────

def load_dicom_series(dicom_dir: str | Path) -> Tuple[np.ndarray,
                                                       Tuple[float, float, float],
                                                       dict]:
    """Load a DICOM series and return HU volume + metadata.

    Args:
        dicom_dir: Path to folder containing .dcm files.

    Returns:
        volume_hu:   (D, H, W) float32 array in Hounsfield Units
        spacing_mm:  (dz, dy, dx) in mm
        metadata:    dict with study/series info (no PHI)
    """
    dicom_dir = Path(dicom_dir)

    # ── Read and sort slices by z-position ────────────────────
    dcm_files = sorted(dicom_dir.rglob("*.dcm"))
    if not dcm_files:
        dcm_files = [f for f in dicom_dir.iterdir() if f.is_file()]

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            if hasattr(ds, "pixel_array"):
                z = float(ds.ImagePositionPatient[2]) \
                    if hasattr(ds, "ImagePositionPatient") else 0.0
                slices.append((z, ds))
        except Exception:
            pass

    if not slices:
        raise ValueError(f"No readable DICOM slices in {dicom_dir}")

    slices.sort(key=lambda t: t[0])

    # ── Build volume with HU conversion ───────────────────────
    ref = slices[0][1]
    ps  = [float(x) for x in ref.PixelSpacing] \
          if hasattr(ref, "PixelSpacing") else [1.0, 1.0]
    st  = float(ref.SliceThickness) \
          if hasattr(ref, "SliceThickness") else 1.0

    arrays = []
    for _, ds in slices:
        arr   = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + inter)

    volume_hu = np.stack(arrays, axis=0)   # (D, H, W)
    spacing_mm = (st, ps[0], ps[1])        # (dz, dy, dx)

    metadata = {
        "StudyDate":        str(getattr(ref, "StudyDate", "")),
        "Manufacturer":     str(getattr(ref, "Manufacturer", "")),
        "Rows":             int(getattr(ref, "Rows", 0)),
        "Columns":          int(getattr(ref, "Columns", 0)),
        "SliceCount":       len(slices),
        "SliceThickness":   st,
        "PixelSpacingRow":  ps[0],
        "PixelSpacingCol":  ps[1],
        "ConvolutionKernel": str(getattr(ref, "ConvolutionKernel", "")),
    }
    return volume_hu, spacing_mm, metadata


# ─────────────────────────────────────────────────────────────
#  Single DICOM file → NumPy slice (for 2D inference)
# ─────────────────────────────────────────────────────────────

def load_dicom_slice(dcm_path: str | Path) -> np.ndarray:
    """Load a single DICOM file → 2D HU array (H, W) float32."""
    ds    = pydicom.dcmread(str(dcm_path))
    arr   = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + inter


# ─────────────────────────────────────────────────────────────
#  Resampling to isotropic 1mm³ spacing via SimpleITK
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

    orig_size    = np.array(sitk_vol.GetSize(), dtype=np.float64)   # (W, H, D)
    orig_spacing = np.array(sitk_vol.GetSpacing(), dtype=np.float64)
    new_spacing  = np.array([target_spacing[2], target_spacing[1], target_spacing[0]])
    new_size     = np.round(orig_size * orig_spacing / new_spacing).astype(int)

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


# ─────────────────────────────────────────────────────────────
#  NIfTI export
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
    # NIfTI affine: diagonal with voxel sizes
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
                               ) -> Tuple[np.ndarray, dict]:
    """Full DICOM → resampled + windowed → NIfTI pipeline.

    Does NOT overwrite original DICOMs.

    Returns:
        processed volume (D, H, W) float32 [0,1] and metadata dict
    """
    # Load
    vol_hu, orig_spacing, metadata = load_dicom_series(dicom_dir)

    # Resample to isotropic
    vol_resampled, new_spacing = resample_volume(
        vol_hu, orig_spacing, target_spacing)

    # Window (normalise to [0,1])
    vol_windowed = apply_hu_window(vol_resampled, preset=window_preset)

    # Save
    save_nifti(vol_windowed, new_spacing, out_nifti)

    metadata.update({
        "original_spacing_mm": orig_spacing,
        "resampled_spacing_mm": new_spacing,
        "resampled_shape": vol_resampled.shape,
        "window_preset": window_preset,
        "output_nifti": str(out_nifti),
    })
    return vol_windowed, metadata


class DICOMLoader:
    """Class wrapper for DICOM volumetric loading and preprocessing."""
    @staticmethod
    def load_series(dicom_dir: str | Path):
        return load_dicom_series(dicom_dir)

    @staticmethod
    def resample(volume_hu: np.ndarray, orig_spacing: Tuple[float, float, float], target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        return resample_volume(volume_hu, orig_spacing, target_spacing)

    @staticmethod
    def to_processed_volume(dicom_dir: str | Path, out_nifti: str | Path, window_preset: str = "lung"):
        return dicom_to_processed_volume(dicom_dir, out_nifti, window_preset=window_preset)

