# src/robustness/perturbations.py
"""Image perturbation functions for robustness analysis.
Implements all perturbation types described in the project synopsis:
  - Gaussian noise
  - Salt-and-pepper noise
  - Gaussian blur
  - Image rotation
  - Brightness variation
  - Contrast variation
  - JPEG compression artifacts
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import torch
from torchvision import transforms


def add_gaussian_noise(img: Image.Image, mean: float = 0.0, std: float = 25.0) -> Image.Image:
    """Add Gaussian noise to a PIL Image."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(mean, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def add_salt_pepper_noise(img: Image.Image, prob: float = 0.05) -> Image.Image:
    """Add salt-and-pepper noise to a PIL Image."""
    arr = np.array(img, dtype=np.uint8).copy()
    # Salt
    salt_mask = np.random.random(arr.shape[:2]) < prob / 2
    arr[salt_mask] = 255
    # Pepper
    pepper_mask = np.random.random(arr.shape[:2]) < prob / 2
    arr[pepper_mask] = 0
    return Image.fromarray(arr)


def apply_gaussian_blur(img: Image.Image, radius: float = 3.0) -> Image.Image:
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_rotation(img: Image.Image, angle: float = 15.0) -> Image.Image:
    """Rotate image by angle degrees (black fill)."""
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))


def adjust_brightness(img: Image.Image, factor: float = 1.5) -> Image.Image:
    """Adjust brightness. factor > 1 = brighter, < 1 = darker."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img: Image.Image, factor: float = 0.5) -> Image.Image:
    """Adjust contrast. factor > 1 = more contrast, < 1 = less."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_jpeg_compression(img: Image.Image, quality: int = 10) -> Image.Image:
    """Simulate JPEG compression artifacts."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ── Registry ──────────────────────────────────
PERTURBATION_REGISTRY = {
    "gaussian_noise_low":    lambda img: add_gaussian_noise(img, std=15.0),
    "gaussian_noise_med":    lambda img: add_gaussian_noise(img, std=30.0),
    "gaussian_noise_high":   lambda img: add_gaussian_noise(img, std=50.0),
    "salt_pepper_low":       lambda img: add_salt_pepper_noise(img, prob=0.02),
    "salt_pepper_med":       lambda img: add_salt_pepper_noise(img, prob=0.05),
    "salt_pepper_high":      lambda img: add_salt_pepper_noise(img, prob=0.10),
    "blur_low":              lambda img: apply_gaussian_blur(img, radius=1.5),
    "blur_med":              lambda img: apply_gaussian_blur(img, radius=3.0),
    "blur_high":             lambda img: apply_gaussian_blur(img, radius=5.0),
    "rotate_5":              lambda img: apply_rotation(img, 5.0),
    "rotate_15":             lambda img: apply_rotation(img, 15.0),
    "rotate_30":             lambda img: apply_rotation(img, 30.0),
    "brightness_low":        lambda img: adjust_brightness(img, 0.5),
    "brightness_high":       lambda img: adjust_brightness(img, 1.5),
    "contrast_low":          lambda img: adjust_contrast(img, 0.5),
    "contrast_high":         lambda img: adjust_contrast(img, 1.5),
    "jpeg_q10":              lambda img: apply_jpeg_compression(img, 10),
    "jpeg_q30":              lambda img: apply_jpeg_compression(img, 30),
}


def apply_perturbation(img: Image.Image, name: str) -> Image.Image:
    """Apply a named perturbation from the registry."""
    if name not in PERTURBATION_REGISTRY:
        raise ValueError(f"Unknown perturbation: {name}. Available: {list(PERTURBATION_REGISTRY.keys())}")
    return PERTURBATION_REGISTRY[name](img)


def get_all_perturbation_names():
    """Return all available perturbation names."""
    return list(PERTURBATION_REGISTRY.keys())
