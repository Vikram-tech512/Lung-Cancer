"""
backend/pipeline/evaluate/__init__.py
=======================================
Evaluation pipeline sub-package.

Provides
--------
* ModelEvaluator    — comprehensive classification/segmentation evaluation
* CalibrationEval   — reliability diagrams, ECE, MCE, Brier score
* CurveGenerator    — ROC, PR, DET curves with confidence intervals
* StatisticalTests  — McNemar, DeLong AUC comparison, Bootstrap CI
"""

from .evaluator import ModelEvaluator
from .calibration_eval import CalibrationEvaluator
from .curves import CurveGenerator
from .statistical_tests import StatisticalTests

__all__ = [
    "ModelEvaluator",
    "CalibrationEvaluator",
    "CurveGenerator",
    "StatisticalTests",
]
