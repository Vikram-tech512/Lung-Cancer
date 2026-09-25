"""
backend/robustness/ct_perturbations.py
======================================
Chest CT-specific volumetric perturbations for robustness stress testing.
Extends the baseline 18 2D perturbations with physical scanner acquisition variations:
  - HU shift (+50 HU, -50 HU)
  - Spacing degradation / slice thickness (1mm -> 2.5mm / 5mm)
  - Quantum noise (low-dose CT simulation)
  - Ring artifacts (detector element miscalibration)
  - Motion streak artifacts (respiratory artifact simulation)
"""

from __future__ import annotations
import numpy as np
import scipy.ndimage
from typing import Dict, Any, Callable, List

class CTPerturbationSuite:
    """
    Registry of 3D CT perturbations to evaluate deep learning model robustness against
    clinical scan condition shifts.
    """

    @staticmethod
    def hu_shift(volume: np.ndarray, delta_hu: float = 50.0) -> np.ndarray:
        """Simulates global calibration offset or contrast enhancement."""
        return volume + delta_hu

    @staticmethod
    def low_dose_quantum_noise(volume: np.ndarray, sigma_hu: float = 25.0) -> np.ndarray:
        """Simulates ultra-low dose thoracic CT photon starvation noise."""
        noise = np.random.normal(0, sigma_hu, size=volume.shape).astype(np.float32)
        return volume + noise

    @staticmethod
    def slice_thickness_degradation(volume: np.ndarray, factor: float = 2.5) -> np.ndarray:
        """Simulates thicker slice thickness reconstruction (e.g. 1mm -> 2.5mm or 5mm)."""
        # Downsample along z-axis, then resample back
        d, h, w = volume.shape
        down_d = max(2, int(d / factor))
        downsampled = scipy.ndimage.zoom(volume, (down_d / d, 1.0, 1.0), order=1)
        restored = scipy.ndimage.zoom(downsampled, (d / down_d, 1.0, 1.0), order=1)
        return restored[:d, :h, :w]

    @staticmethod
    def ring_artifact(volume: np.ndarray, radius_px: int = 24, intensity_hu: float = 80.0) -> np.ndarray:
        """Simulates CT detector ring artifact in axial planes."""
        d, h, w = volume.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        ring_mask = np.abs(dist - radius_px) < 1.2

        perturbed = volume.copy()
        perturbed[:, ring_mask] += intensity_hu
        return perturbed

    @staticmethod
    def respiratory_motion_streak(volume: np.ndarray, streak_angle_deg: float = 15.0, severity: float = 40.0) -> np.ndarray:
        """Simulates patient respiratory motion blur during helical scan acquisition."""
        rad = np.radians(streak_angle_deg)
        kernel_len = 9
        kernel = np.zeros((kernel_len, kernel_len))
        for i in range(kernel_len):
            offset = int(np.round((i - kernel_len // 2) * np.tan(rad)))
            col = np.clip(kernel_len // 2 + offset, 0, kernel_len - 1)
            kernel[i, col] = 1.0
        kernel /= (kernel.sum() + 1e-7)

        perturbed = np.empty_like(volume)
        for z in range(volume.shape[0]):
            perturbed[z] = scipy.ndimage.convolve(volume[z], kernel, mode="reflect")
        return perturbed

    @classmethod
    def get_all_perturbations(cls) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        return {
            "hu_shift_+50": lambda v: cls.hu_shift(v, 50.0),
            "hu_shift_-50": lambda v: cls.hu_shift(v, -50.0),
            "low_dose_noise": lambda v: cls.low_dose_quantum_noise(v, 25.0),
            "slice_thickness_2.5mm": lambda v: cls.slice_thickness_degradation(v, 2.5),
            "ring_artifact": lambda v: cls.ring_artifact(v, 24, 75.0),
            "respiratory_motion": lambda v: cls.respiratory_motion_streak(v, 15.0, 40.0),
        }

    @classmethod
    def run_suite(cls, volume: np.ndarray, predict_fn: Callable[[np.ndarray], float]) -> List[Dict[str, Any]]:
        """
        Runs full perturbation suite and tracks prediction delta, robustness degradation,
        and stability index.
        """
        baseline_prob = predict_fn(volume)
        results = [
            {
                "perturbation": "baseline",
                "probability": float(baseline_prob),
                "delta": 0.0,
                "relative_shift": 0.0,
                "status": "PASS"
            }
        ]

        for name, perturb_func in cls.get_all_perturbations().items():
            perturbed_vol = perturb_func(volume)
            perturbed_prob = predict_fn(perturbed_vol)
            delta = float(perturbed_prob - baseline_prob)
            rel_shift = float(abs(delta) / (baseline_prob + 1e-5))

            status = "PASS" if abs(delta) < 0.15 else ("WARNING" if abs(delta) < 0.30 else "VULNERABLE")

            results.append({
                "perturbation": name,
                "probability": float(perturbed_prob),
                "delta": round(delta, 4),
                "relative_shift": round(rel_shift, 4),
                "status": status
            })

        return results
