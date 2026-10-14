"""
backend/pipeline/__init__.py
=============================
Unified ML Pipeline orchestration layer.
Exposes all pipeline stages as importable modules:
  - data      : Dataset loaders, augmentation, splitting
  - train     : Training loops, schedulers, optimisers
  - evaluate  : Metrics, curves, statistical tests
  - infer     : Inference engine with batched CT scanning
  - radiomics : IBSI PyRadiomics feature extraction
  - xai       : 3D Grad-CAM, IG, Attention Rollout
  - uncertainty: MC Dropout, ensemble variance
  - calibration: Platt & Temperature scaling
  - registry  : MLflow model lifecycle management
"""

from .orchestrator import PipelineOrchestrator

__all__ = ["PipelineOrchestrator"]
