"""
backend/calibration/temperature_scaling.py
==========================================
Temperature Scaling for post-hoc probability calibration.

Extends existing calibration.py (ECE/MCE/reliability diagram) from
lung-cancer-xai-phase4/src/calibration/calibration.py with:

  1. TemperatureScaler — single scalar T optimised on validation logits
  2. Brier score computation
  3. Raw vs calibrated probability comparison (shown side-by-side in UI)

Reference: Guo et al., 2017 — "On Calibration of Modern Neural Networks"

IMPORTANT: Raw probability and calibrated probability are ALWAYS
shown separately. Never replace raw output silently.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class CalibrationResult:
    """Calibration metrics for a model on a given split."""
    model_name: str
    ece: float                # Expected Calibration Error
    mce: float                # Maximum Calibration Error
    brier_score: float        # Brier score (mean squared error of probabilities)
    temperature: float        # Optimal temperature T (1.0 = uncalibrated)
    n_bins: int = 10
    dataset: str = ""
    split: str = "validation"


# ─────────────────────────────────────────────────────────────
#  Temperature Scaler
# ─────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling.

    Usage::
        scaler = TemperatureScaler()
        # Fit on validation set (logits + labels)
        scaler.fit(val_logits, val_labels)
        # Apply to test logits
        calibrated_probs = scaler.calibrate(test_logits)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1, max=10.0)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling → calibrated probabilities."""
        with torch.no_grad():
            return torch.softmax(self.forward(logits), dim=-1)

    def fit(self, val_logits: torch.Tensor,
            val_labels: torch.Tensor,
            max_iter: int = 200,
            lr: float = 0.01) -> float:
        """Optimise temperature T on validation NLL.

        Args:
            val_logits: (N, num_classes) raw logits (NO softmax applied).
            val_labels: (N,) integer class labels.
            max_iter:   Optimiser iterations.
            lr:         Learning rate.

        Returns:
            Optimal temperature T.
        """
        self.train()
        optimiser = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def _eval():
            optimiser.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimiser.step(_eval)
        t = float(self.temperature.item())
        self.eval()
        return t

    def save(self, path: str | Path):
        torch.save({"temperature": self.temperature.item()}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        ckpt = torch.load(str(path), map_location="cpu")
        scaler = cls()
        scaler.temperature = nn.Parameter(
            torch.tensor([ckpt["temperature"]]))
        return scaler


# ─────────────────────────────────────────────────────────────
#  Brier Score
# ─────────────────────────────────────────────────────────────

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multiclass Brier score.

    Args:
        probs:  (N, C) probability array.
        labels: (N,) integer class labels.

    Returns:
        float — mean squared error of probability estimates.
    """
    N, C = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ─────────────────────────────────────────────────────────────
#  ECE / MCE (reused from calibration.py, self-contained)
# ─────────────────────────────────────────────────────────────

def expected_calibration_error(confidences: np.ndarray,
                                 correct: np.ndarray,
                                 n_bins: int = 10) -> dict:
    """ECE/MCE (Guo et al., 2017).

    Reuses logic from lung-cancer-xai-phase4/src/calibration/calibration.py.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids   = np.digitize(confidences, bin_edges[1:-1])
    ece = mce = 0.0
    n   = len(confidences)
    bin_stats = []
    for b in range(n_bins):
        mask  = bin_ids == b
        count = mask.sum()
        if count == 0:
            bin_stats.append({"bin": b, "count": 0})
            continue
        avg_conf = float(confidences[mask].mean())
        acc      = float(correct[mask].mean())
        gap      = abs(avg_conf - acc)
        ece     += (count / n) * gap
        mce      = max(mce, gap)
        bin_stats.append({"bin": b, "count": int(count),
                          "avg_confidence": avg_conf, "accuracy": acc})
    return {"ece": float(ece), "mce": float(mce), "bins": bin_stats}


# ─────────────────────────────────────────────────────────────
#  Full calibration evaluation for a model
# ─────────────────────────────────────────────────────────────

def evaluate_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model_name: str,
    scaler: Optional[TemperatureScaler] = None,
    dataset: str = "",
    split: str = "test",
) -> CalibrationResult:
    """Compute full calibration metrics for raw and (optionally) calibrated probs.

    Returns CalibrationResult with ECE, MCE, Brier score.
    Always shows raw and calibrated results separately — never silently replaces.
    """
    with torch.no_grad():
        if scaler is not None:
            probs_tensor = scaler.calibrate(logits)
            T = float(scaler.temperature.item())
        else:
            probs_tensor = torch.softmax(logits, dim=-1)
            T = 1.0

    probs  = probs_tensor.cpu().numpy()         # (N, C)
    labels_np = labels.cpu().numpy()            # (N,)

    confidences = probs.max(axis=1)             # (N,) max class prob
    preds       = probs.argmax(axis=1)          # (N,)
    correct     = (preds == labels_np).astype(float)

    cal = expected_calibration_error(confidences, correct)
    bs  = brier_score(probs, labels_np)

    return CalibrationResult(
        model_name=model_name,
        ece=cal["ece"],
        mce=cal["mce"],
        brier_score=bs,
        temperature=T,
        dataset=dataset,
        split=split,
    )
