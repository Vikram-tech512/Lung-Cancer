"""
backend/pipeline/radiomics/__init__.py
========================================
Radiomics pipeline sub-package.

Provides
--------
* RadiomicsPipeline  — end-to-end IBSI-compliant feature extraction + selection
* FeatureSelector    — variance, correlation & LASSO-based feature selection
* RadiomicsCache     — SQLite-backed cache for expensive feature extraction
"""

from .pipeline import RadiomicsPipeline
from .feature_selector import FeatureSelector

__all__ = ["RadiomicsPipeline", "FeatureSelector"]
