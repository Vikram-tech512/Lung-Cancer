"""
backend/uncertainty/mc_dropout.py
==================================
Monte Carlo Dropout uncertainty estimation for 3D classifiers.

Method: Gal & Ghahramani, 2016 — "Dropout as a Bayesian Approximation"

Algorithm:
  1. Call model.enable_mc_dropout()  (keeps only Dropout layers in train mode)
  2. Run T=30 stochastic forward passes with the SAME input
  3. Collect T probability distributions (softmax outputs)
  4. Compute: mean, std, entropy → uncertainty category

UI Display (NEVER "diagnostic certainty"):
  MALIGNANT-CLASS PROBABILITY: XX.X%
  UNCERTAINTY: Low / Moderate / High
  (based on std of malignant-class probability across T passes)

Uncertainty thresholds (validated on LIDC-IDRI val set):
  Low:      std < 0.05   — model is consistent across passes
  Moderate: std 0.05–0.15
  High:     std > 0.15   — predictions vary substantially; interpret with caution
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class UncertaintyResult:
    """Result of MC Dropout uncertainty estimation."""
    mean_probs: np.ndarray          # (num_classes,) mean probability over T passes
    std_probs: np.ndarray           # (num_classes,) std over T passes
    entropy: float                  # predictive entropy
    malignant_prob: float           # mean P(malignant) — for UI display
    malignant_std: float            # std P(malignant) — primary uncertainty metric
    uncertainty_category: str       # "Low" | "Moderate" | "High"
    n_passes: int                   # T
    all_probs: np.ndarray           # (T, num_classes) — all raw pass results

    # Disclaimer — always attach to API response
    disclaimer: str = (
        "This AI output is for research/decision-support purposes and does "
        "NOT constitute a definitive medical diagnosis."
    )


# Uncertainty thresholds (std of malignant-class probability)
_UNCERTAINTY_THRESHOLDS = {
    "Low":      (0.0,  0.05),
    "Moderate": (0.05, 0.15),
    "High":     (0.15, 1.0),
}


def _categorise_uncertainty(std: float) -> str:
    for cat, (lo, hi) in _UNCERTAINTY_THRESHOLDS.items():
        if lo <= std < hi:
            return cat
    return "High"


def mc_dropout_predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_passes: int = 30,
    malignant_class_idx: int = 1,
) -> UncertaintyResult:
    """Run MC Dropout uncertainty estimation.

    Args:
        model:               PyTorch model with enable_mc_dropout() method.
        input_tensor:        (1, 1, D, H, W) CT nodule volume.
        n_passes:            Number of stochastic forward passes (default 30).
        malignant_class_idx: Which output index is "malignant" (default 1).

    Returns:
        UncertaintyResult with mean, std, entropy, and category.
    """
    if not hasattr(model, "enable_mc_dropout"):
        raise AttributeError(
            "Model must implement enable_mc_dropout(). "
            "Add it to ResNet3D50, DenseNet3D121, or SwinTransformer3D."
        )

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Enable MC Dropout: model.eval() + Dropout layers in train()
    model.enable_mc_dropout()

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(input_tensor)         # (1, num_classes)
            probs  = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy()[0])   # (num_classes,)

    all_probs_arr = np.stack(all_probs, axis=0)         # (T, num_classes)

    mean_probs = all_probs_arr.mean(axis=0)             # (num_classes,)
    std_probs  = all_probs_arr.std(axis=0)              # (num_classes,)

    # Predictive entropy: H = -sum(p * log(p))
    p_clipped = np.clip(mean_probs, 1e-9, 1.0)
    entropy   = float(-np.sum(p_clipped * np.log(p_clipped)))

    malignant_prob = float(mean_probs[malignant_class_idx])
    malignant_std  = float(std_probs[malignant_class_idx])
    uncertainty_cat = _categorise_uncertainty(malignant_std)

    return UncertaintyResult(
        mean_probs=mean_probs,
        std_probs=std_probs,
        entropy=entropy,
        malignant_prob=malignant_prob,
        malignant_std=malignant_std,
        uncertainty_category=uncertainty_cat,
        n_passes=n_passes,
        all_probs=all_probs_arr,
    )


def deep_ensemble_predict(
    models: List[nn.Module],
    input_tensor: torch.Tensor,
    malignant_class_idx: int = 1,
) -> UncertaintyResult:
    """Deep Ensemble uncertainty: average predictions from multiple models.

    Args:
        models:       List of independently trained models.
        input_tensor: (1, 1, D, H, W)
        malignant_class_idx: Index of malignant class.

    Returns:
        UncertaintyResult using ensemble disagreement as uncertainty.
    """
    device = next(models[0].parameters()).device
    input_tensor = input_tensor.to(device)

    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=-1)
            all_probs.append(probs.cpu().numpy()[0])

    all_probs_arr = np.stack(all_probs, axis=0)
    mean_probs    = all_probs_arr.mean(axis=0)
    std_probs     = all_probs_arr.std(axis=0)

    p_clipped = np.clip(mean_probs, 1e-9, 1.0)
    entropy   = float(-np.sum(p_clipped * np.log(p_clipped)))

    malignant_prob = float(mean_probs[malignant_class_idx])
    malignant_std  = float(std_probs[malignant_class_idx])

    return UncertaintyResult(
        mean_probs=mean_probs,
        std_probs=std_probs,
        entropy=entropy,
        malignant_prob=malignant_prob,
        malignant_std=malignant_std,
        uncertainty_category=_categorise_uncertainty(malignant_std),
        n_passes=len(models),
        all_probs=all_probs_arr,
    )


class MCDropoutEstimator:
    """Class wrapper for Monte Carlo Dropout uncertainty estimation."""
    @staticmethod
    def estimate(model: nn.Module, input_tensor: torch.Tensor, n_passes: int = 30, malignant_class_idx: int = 1) -> UncertaintyResult:
        return mc_dropout_predict(model, input_tensor, n_passes=n_passes, malignant_class_idx=malignant_class_idx)

    @staticmethod
    def deep_ensemble(models: List[nn.Module], input_tensor: torch.Tensor, malignant_class_idx: int = 1) -> UncertaintyResult:
        return deep_ensemble_predict(models, input_tensor, malignant_class_idx=malignant_class_idx)

