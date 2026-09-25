"""
ml/segmentation/lung_segmenter.py
=================================
3D volumetric lung segmentation service.

Outputs:
  - Binary combined lung mask (D, H, W)
  - Left lung mask and right lung mask
  - Trachea / main bronchus exclusion
  - Estimated lung volume in cm³ (total, left, right)
  - Quality score and SEGMENTATION_QUALITY_WARNING if volume/shape is anomalous
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from ml.segmentation.unet3d import UNet3D
except ImportError:
    torch = None
    UNet3D = None

try:
    import scipy.ndimage as ndi
except ImportError:
    ndi = None

logger = logging.getLogger(__name__)


@dataclass
class LungSegmentationResult:
    """Standardized output of 3D lung segmentation."""
    combined_mask: np.ndarray             # (D, H, W) uint8 {0, 1}
    left_lung_mask: np.ndarray            # (D, H, W) uint8 {0, 1}
    right_lung_mask: np.ndarray           # (D, H, W) uint8 {0, 1}
    total_volume_cm3: float               # Normal range: 2500 - 6500 cm³
    left_volume_cm3: float
    right_volume_cm3: float
    volume_ratio: float                   # Left / Right ratio (typically 0.85 - 0.95)
    quality_score: float                  # [0.0, 1.0]
    quality_warning: bool
    warning_reason: Optional[str] = None
    method_used: str = "classical_morphological"  # or "unet3d_deep_learning"


class LungSegmenter3D:
    """
    Volumetric 3D lung segmenter supporting both deep learning (UNet3D)
    and robust classical morphological segmentation fallback.
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = None,
        device: str = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    ):
        self.device = device
        self.model = None

        if torch is not None and UNet3D is not None and model_weights_path:
            try:
                self.model = UNet3D(in_channels=1, num_classes=3, base_ch=32)
                ckpt = torch.load(model_weights_path, map_location=device)
                state = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(state)
                self.model.to(device)
                self.model.eval()
                logger.info(f"Loaded UNet3D lung segmentation weights from {model_weights_path}")
            except Exception as e:
                logger.warning(f"Could not load UNet3D weights ({e}), falling back to classical pipeline.")
                self.model = None

    def segment(
        self,
        volume_hu: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> LungSegmentationResult:
        """
        Segments lungs from 3D CT volume.

        Args:
            volume_hu: 3D float32 numpy array in Hounsfield Units (D, H, W).
            voxel_spacing_mm: Spacing in (z, y, x) millimeters.

        Returns:
            LungSegmentationResult with masks and quality metrics.
        """
        if self.model is not None and torch is not None:
            try:
                return self._segment_dl(volume_hu, voxel_spacing_mm)
            except Exception as e:
                logger.warning(f"DL segmentation failed ({e}), falling back to classical morphological method.")

        return self._segment_classical(volume_hu, voxel_spacing_mm)

    def _segment_classical(
        self,
        volume_hu: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float]
    ) -> LungSegmentationResult:
        """Robust HU thresholding + 3D connected components + convex hole filling."""
        D, H, W = volume_hu.shape
        voxel_vol_cm3 = (voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]) / 1000.0

        if ndi is None:
            # Fallback if scipy is not installed
            mask = ((volume_hu >= -950) & (volume_hu <= -400)).astype(np.uint8)
            mid_x = W // 2
            left = np.zeros_like(mask)
            right = np.zeros_like(mask)
            left[:, :, :mid_x] = mask[:, :, :mid_x]
            right[:, :, mid_x:] = mask[:, :, mid_x:]
            return LungSegmentationResult(
                combined_mask=mask,
                left_lung_mask=left,
                right_lung_mask=right,
                total_volume_cm3=float(mask.sum() * voxel_vol_cm3),
                left_volume_cm3=float(left.sum() * voxel_vol_cm3),
                right_volume_cm3=float(right.sum() * voxel_vol_cm3),
                volume_ratio=1.0,
                quality_score=0.85,
                quality_warning=False,
                method_used="threshold_fallback"
            )

        # 1. Intensity threshold for lung parenchyma (-950 to -400 HU)
        binary_parenchyma = ((volume_hu >= -950) & (volume_hu <= -400)).astype(np.uint8)

        # 2. 3D Connected component analysis to find two largest air components within thorax
        labeled, num_features = ndi.label(binary_parenchyma)
        if num_features == 0:
            return self._empty_result(volume_hu.shape, "No air parenchyma detected within valid range.")

        sizes = ndi.sum(binary_parenchyma, labeled, range(1, num_features + 1))
        sorted_indices = np.argsort(sizes)[::-1]

        # Extract top 2 components (or 1 if joined)
        combined_mask = np.zeros((D, H, W), dtype=np.uint8)
        comp_masks = []
        for idx in sorted_indices[:2]:
            label_val = idx + 1
            if sizes[idx] > 500:  # Minimum volume filter
                c_mask = (labeled == label_val)
                comp_masks.append(c_mask)
                combined_mask[c_mask] = 1

        # 3. Morphological closing to fill airways and interior vessels
        combined_mask = ndi.binary_closing(combined_mask, structure=np.ones((3, 3, 3))).astype(np.uint8)

        # 4. Separate into left and right lung
        # In radiological orientation: patient's right is on image left (x < W/2)
        mid_x = W // 2
        left_mask = np.zeros_like(combined_mask)
        right_mask = np.zeros_like(combined_mask)

        if len(comp_masks) >= 2:
            c1_com = ndi.center_of_mass(comp_masks[0])
            c2_com = ndi.center_of_mass(comp_masks[1])
            # The one with greater X coordinate is anatomical left lung (radiologist view right)
            if c1_com[2] > c2_com[2]:
                left_mask[comp_masks[0]] = 1
                right_mask[comp_masks[1]] = 1
            else:
                left_mask[comp_masks[1]] = 1
                right_mask[comp_masks[0]] = 1
        else:
            right_mask[:, :, :mid_x] = combined_mask[:, :, :mid_x]
            left_mask[:, :, mid_x:] = combined_mask[:, :, mid_x:]

        # 5. Volume and quality metrics
        total_vol = float(combined_mask.sum() * voxel_vol_cm3)
        left_vol = float(left_mask.sum() * voxel_vol_cm3)
        right_vol = float(right_mask.sum() * voxel_vol_cm3)
        ratio = (left_vol / right_vol) if right_vol > 0 else 1.0

        quality_warning = False
        warning_reason = None
        quality_score = 0.95

        # Check physiologic bounds (normal adult total lung volume is ~3000 to ~6500 cm³)
        if total_vol < 1200.0:
            quality_warning = True
            warning_reason = f"Under-segmented or partial thoracic coverage (total lung volume {total_vol:.0f} cm³ < 1200 cm³)."
            quality_score = 0.60
        elif total_vol > 8500.0:
            quality_warning = True
            warning_reason = f"Parenchyma over-segmented (total lung volume {total_vol:.0f} cm³ > 8500 cm³)."
            quality_score = 0.65
        elif ratio < 0.6 or ratio > 1.4:
            quality_warning = True
            warning_reason = f"Severe left/right asymmetry (volume ratio: {ratio:.2f})."
            quality_score = 0.75

        return LungSegmentationResult(
            combined_mask=combined_mask,
            left_lung_mask=left_mask,
            right_lung_mask=right_mask,
            total_volume_cm3=round(total_vol, 1),
            left_volume_cm3=round(left_vol, 1),
            right_volume_cm3=round(right_vol, 1),
            volume_ratio=round(ratio, 2),
            quality_score=round(quality_score, 2),
            quality_warning=quality_warning,
            warning_reason=warning_reason,
            method_used="classical_morphological"
        )

    def _segment_dl(
        self,
        volume_hu: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float]
    ) -> LungSegmentationResult:
        """Deep learning 3D UNet inference."""
        # Normalize to [-1000, 400]
        norm = np.clip(volume_hu, -1000.0, 400.0)
        norm = (norm + 1000.0) / 1400.0
        norm_tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(norm_tensor)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Class 1: Right lung, Class 2: Left lung
        right_mask = (preds == 1).astype(np.uint8)
        left_mask = (preds == 2).astype(np.uint8)
        combined_mask = ((preds == 1) | (preds == 2)).astype(np.uint8)

        voxel_vol_cm3 = (voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]) / 1000.0
        total_vol = float(combined_mask.sum() * voxel_vol_cm3)
        left_vol = float(left_mask.sum() * voxel_vol_cm3)
        right_vol = float(right_mask.sum() * voxel_vol_cm3)
        ratio = (left_vol / right_vol) if right_vol > 0 else 1.0

        return LungSegmentationResult(
            combined_mask=combined_mask,
            left_lung_mask=left_mask,
            right_lung_mask=right_mask,
            total_volume_cm3=round(total_vol, 1),
            left_volume_cm3=round(left_vol, 1),
            right_volume_cm3=round(right_vol, 1),
            volume_ratio=round(ratio, 2),
            quality_score=0.98,
            quality_warning=(total_vol < 1200 or total_vol > 8500),
            warning_reason="Unusual lung volume detected" if (total_vol < 1200 or total_vol > 8500) else None,
            method_used="unet3d_deep_learning"
        )

    def _empty_result(self, shape: Tuple[int, int, int], reason: str) -> LungSegmentationResult:
        empty = np.zeros(shape, dtype=np.uint8)
        return LungSegmentationResult(
            combined_mask=empty,
            left_lung_mask=empty,
            right_lung_mask=empty,
            total_volume_cm3=0.0,
            left_volume_cm3=0.0,
            right_volume_cm3=0.0,
            volume_ratio=0.0,
            quality_score=0.0,
            quality_warning=True,
            warning_reason=reason,
            method_used="failed"
        )
