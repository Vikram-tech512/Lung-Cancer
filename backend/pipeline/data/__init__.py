"""
backend/pipeline/data/__init__.py
====================================
Data pipeline sub-package.

Provides
--------
* AugmentationPipeline  — 3D CT augmentation (flips, rotations, noise, contrast)
* CTSplitter            — patient-stratified train / val / test splitting
* DataLoaderFactory     — builds PyTorch DataLoader from any dataset class
* SyntheticVolumeGen    — fast synthetic CT generator for smoke-testing
"""

from .augmentation import AugmentationPipeline
from .splitter import CTSplitter
from .loader_factory import DataLoaderFactory

__all__ = [
    "AugmentationPipeline",
    "CTSplitter",
    "DataLoaderFactory",
]
