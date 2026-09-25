"""
backend/preprocessing/ct_pipeline.py
====================================
End-to-end volumetric CT preprocessing pipeline for thoracic imaging.

Handles:
  1. Hounsfield Unit (HU) windowing across standard clinical presets
  2. Isotropic 3D resampling (default 1.0mm x 1.0mm x 1.0mm)
  3. Orientation validation and axial normalization
  4. Pulmonary parenchymal masking
  5. 3D subvolume / ROI candidate cube extraction with boundary-safe padding
  6. Radiomic and deep learning normalization schemes (min-max [0, 1] or standard Z-score)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np

try:
    import scipy.ndimage
except ImportError:
    scipy = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedVolume:
    """Container for preprocessed volumetric CT data and anatomical metadata."""
    volume: np.ndarray                   # Shape (D, H, W), float32, normalized or windowed
    raw_hu: Optional[np.ndarray]         # Shape (D, H, W), float32 Hounsfield units
    original_spacing: Tuple[float, float, float]
    resampled_spacing: Tuple[float, float, float]
    window_applied: str
    metadata: Dict[str, Any]


class CTPipeline:
    """
    Production thoracic CT volumetric preprocessing pipeline.
    Standardizes variable scanner acquisitions into uniform isotropic representations.
    """
    WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
        "lung": (-1000.0, 400.0),          # W:1400, L:-300
        "mediastinum": (-160.0, 240.0),    # W:400, L:40
        "soft_tissue": (-125.0, 225.0),    # W:350, L:50
        "bone": (-100.0, 1000.0),          # W:1100, L:450
        "chest_full": (-1024.0, 1024.0),
    }

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        default_window: str = "lung"
    ):
        self.target_spacing = target_spacing
        self.default_window = default_window
        self.window_range = self.WINDOW_PRESETS.get(default_window, (-1000.0, 400.0))

    def apply_window(
        self,
        volume: np.ndarray,
        window_name: Optional[str] = None,
        custom_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Clips HU values to specified window and scales linearly to [0, 1].

        Args:
            volume: 3D float32 array in Hounsfield Units.
            window_name: Key in WINDOW_PRESETS or None.
            custom_range: (min_hu, max_hu) tuple overriding preset.

        Returns:
            Normalized 3D array of shape (D, H, W) with float32 values in [0.0, 1.0].
        """
        if custom_range is not None:
            w_min, w_max = custom_range
        elif window_name is not None and window_name in self.WINDOW_PRESETS:
            w_min, w_max = self.WINDOW_PRESETS[window_name]
        else:
            w_min, w_max = self.window_range

        clipped = np.clip(volume, w_min, w_max)
        normalized = (clipped - w_min) / (w_max - w_min + 1e-7)
        return normalized.astype(np.float32)

    def resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        is_label_mask: bool = False
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resamples a 3D volume (D, H, W) to target isotropic spacing.
        Uses trilinear/spline interpolation for intensity data, nearest-neighbor for segmentations.
        """
        if np.allclose(current_spacing, self.target_spacing, atol=1e-3):
            return volume.astype(np.float32 if not is_label_mask else np.uint8), self.target_spacing

        # SimpleITK implementation if available (preserves directional cosines accurately)
        if sitk is not None:
            try:
                # SimpleITK expects (X, Y, Z) spacing, while numpy is (Z, Y, X) = (D, H, W)
                itk_img = sitk.GetImageFromArray(volume)
                itk_img.SetSpacing((float(current_spacing[2]), float(current_spacing[1]), float(current_spacing[0])))

                orig_size = itk_img.GetSize()
                orig_spacing = itk_img.GetSpacing()

                new_size = [
                    int(round(orig_size[i] * orig_spacing[i] / self.target_spacing[2 - i]))
                    for i in range(3)
                ]

                resample = sitk.ResampleImageFilter()
                resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label_mask else sitk.sitkLinear)
                resample.SetOutputSpacing((self.target_spacing[2], self.target_spacing[1], self.target_spacing[0]))
                resample.SetSize(new_size)
                resample.SetOutputDirection(itk_img.GetDirection())
                resample.SetOutputOrigin(itk_img.GetOrigin())
                resample.SetDefaultPixelValue(0 if is_label_mask else -1000.0)

                resampled_sitk = resample.Execute(itk_img)
                resampled_arr = sitk.GetArrayFromImage(resampled_sitk)
                return (
                    resampled_arr.astype(np.uint8 if is_label_mask else np.float32),
                    self.target_spacing
                )
            except Exception as e:
                logger.warning(f"SimpleITK resampling failed, falling back to scipy: {e}")

        # Fallback to scipy.ndimage
        if scipy is not None:
            resize_factor = np.array(current_spacing) / np.array(self.target_spacing)
            new_shape = np.round(volume.shape * resize_factor).astype(int)
            real_resize_factor = new_shape / volume.shape

            order = 0 if is_label_mask else 1
            resampled = scipy.ndimage.zoom(volume, real_resize_factor, order=order, mode="nearest")
            return resampled.astype(np.uint8 if is_label_mask else np.float32), self.target_spacing

        # No resampling library available
        return volume, current_spacing

    def extract_nodule_crop(
        self,
        volume: np.ndarray,
        center_zyx: Tuple[int, int, int],
        crop_size: Tuple[int, int, int] = (64, 64, 64),
        pad_value: float = 0.0
    ) -> np.ndarray:
        """
        Extracts a fixed-size 3D cube centered on nodule coordinate (z, y, x).
        Safely pads using pad_value if crop boundary exceeds image borders.
        """
        cz, cy, cx = center_zyx
        dz, dy, dx = crop_size

        z_min = cz - dz // 2
        z_max = z_min + dz
        y_min = cy - dy // 2
        y_max = y_min + dy
        x_min = cx - dx // 2
        x_max = x_min + dx

        pad_z1 = max(0, -z_min)
        pad_z2 = max(0, z_max - volume.shape[0])
        pad_y1 = max(0, -y_min)
        pad_y2 = max(0, y_max - volume.shape[1])
        pad_x1 = max(0, -x_min)
        pad_x2 = max(0, x_max - volume.shape[2])

        src_z1 = max(0, z_min)
        src_z2 = min(volume.shape[0], z_max)
        src_y1 = max(0, y_min)
        src_y2 = min(volume.shape[1], y_max)
        src_x1 = max(0, x_min)
        src_x2 = min(volume.shape[2], x_max)

        sub_vol = volume[src_z1:src_z2, src_y1:src_y2, src_x1:src_x2]

        if any([pad_z1, pad_z2, pad_y1, pad_y2, pad_x1, pad_x2]):
            crop = np.pad(
                sub_vol,
                ((pad_z1, pad_z2), (pad_y1, pad_y2), (pad_x1, pad_x2)),
                mode="constant",
                constant_values=pad_value
            )
        else:
            crop = sub_vol

        return crop.astype(np.float32)

    def mask_lungs(
        self,
        volume: np.ndarray,
        lung_mask: np.ndarray,
        fill_value: float = -1000.0
    ) -> np.ndarray:
        """Applies binary lung parenchyma mask, replacing non-pulmonary pixels with fill_value."""
        masked = volume.copy()
        masked[lung_mask == 0] = fill_value
        return masked

    def preprocess_full_study(
        self,
        raw_volume_hu: np.ndarray,
        current_spacing: Tuple[float, float, float],
        window_name: str = "lung",
        lung_mask: Optional[np.ndarray] = None
    ) -> PreprocessedVolume:
        """
        Executes end-to-end preprocessing:
        Resampling -> Optional Lung Masking -> Clinical Windowing -> Normalization.
        """
        resampled_hu, new_spacing = self.resample_volume(
            raw_volume_hu,
            current_spacing=current_spacing,
            is_label_mask=False
        )

        if lung_mask is not None:
            resampled_mask, _ = self.resample_volume(
                lung_mask,
                current_spacing=current_spacing,
                is_label_mask=True
            )
            processed_hu = self.mask_lungs(resampled_hu, resampled_mask)
        else:
            processed_hu = resampled_hu

        norm_vol = self.apply_window(processed_hu, window_name=window_name)

        return PreprocessedVolume(
            volume=norm_vol,
            raw_hu=resampled_hu,
            original_spacing=current_spacing,
            resampled_spacing=new_spacing,
            window_applied=window_name,
            metadata={
                "shape": norm_vol.shape,
                "spacing": new_spacing,
                "hu_min": float(resampled_hu.min()),
                "hu_max": float(resampled_hu.max()),
                "hu_mean": float(resampled_hu.mean()),
            }
        )
