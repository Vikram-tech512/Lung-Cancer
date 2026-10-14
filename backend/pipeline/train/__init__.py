"""
backend/pipeline/train/__init__.py
======================================
Training pipeline sub-package.

Provides
--------
* Trainer           — unified training loop for all 3D model architectures
* LRSchedulerFactory— factory for cosine, StepLR, OneCycleLR, ReduceLROnPlateau
* EarlyStopping     — patience-based early stopping with metric tracking
* CheckpointManager — versioned checkpoint save/restore with model registry sync
"""

from .trainer import Trainer
from .scheduler import LRSchedulerFactory
from .early_stopping import EarlyStopping
from .checkpoint_manager import CheckpointManager

__all__ = [
    "Trainer",
    "LRSchedulerFactory",
    "EarlyStopping",
    "CheckpointManager",
]
