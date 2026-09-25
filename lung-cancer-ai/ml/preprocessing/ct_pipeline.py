"""
backend/preprocessing/ct_pipeline.py
====================================
End-to-end volumetric CT preprocessing pipeline for thoracic imaging.
Handles HU windowing, isotropic resampling, lung segmentation masking,
and extracting normalized 3D candidate nodule bounding cubes.
"""

from __future__ import annotations
import numpy as np
import scipy.ndimage
from typing import Tuple, Optional, Dict, Any

class CTPipeline:
    """
    Standardizes raw 3D CT volumes for downstream detection, segmentation, and classification.
    """
    WINDOW_PRESETS = {
        "lung": (-1000.0, 400.0),
        "mediastinum": (-160.0, 240.0),
        "soft_tissue": (-125.0, 225.0),
        "bone": (-100.0, 1000.0),
    }

    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0), default_window: str = "lung"):
        self.target_spacing = target_spacing
        self.window_range = self.WINDOW_PRESETS.get(default_window, (-1000.0, 400.0))

    def apply_window(self, volume: np.ndarray, window_min: Optional[float] = None, window_max: Optional[float] = None) -> np.ndarray:
        """Clips HU values to window and scales linearly to [0, 1]."""
        w_min = window_min if window_min is not None else self.window_range[0]
        w_max = window_max if window_max is not None else self.window_range[1]
        clipped = np.clip(volume, w_min, w_max)
        normalized = (clipped - w_min) / (w_max - w_min + 1e-7)
        return normalized.astype(np.float32)

    def resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        order: int = 1
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resamples a 3D volume (D, H, W) to isotropic target spacing using spline interpolation.
        order=1 (linear) for CT volumes, order=0 (nearest neighbor) for masks.
        """
        if current_spacing == self.target_spacing:
            return volume, current_spacing

        resize_factor = np.array(current_spacing) / np.array(self.target_spacing)
        new_shape = np.round(volume.shape * resize_factor).astype(int)
        real_resize_factor = new_shape / volume.shape

        resampled = scipy.ndimage.zoom(volume, real_resize_factor, order=order, mode="nearest")
        return resampled, self.target_spacing

    def extract_nodule_crop(
        self,
        volume: np.ndarray,
        center_zyx: Tuple[int, int, int],
        crop_size: Tuple[int, int, int] = (64, 64, 64),
        pad_value: float = 0.0
    ) -> np.ndarray:
        """
        Extracts a fixed-size 3D cube centered on candidate nodule coordinates (z, y, x).
        Pads safely if crop extends outside volume boundaries.
        Returns: array of shape `crop_size`.
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

    def mask_lungs(self, volume: np.ndarray, lung_mask: np.ndarray, fill_value: float = -1000.0) -> np.ndarray:
        """Applies binary lung mask, filling non-pulmonary parenchyma with air HU (-1000)."""
        masked = volume.copy()
        masked[lung_mask == 0] = fill_value
        return masked
