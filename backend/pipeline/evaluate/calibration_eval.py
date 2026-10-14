"""
backend/pipeline/evaluate/calibration_eval.py
================================================
Calibration Evaluator — reliability diagrams, ECE, MCE, Brier score.

Calibration measures how well predicted probabilities match empirical
frequencies. A model with perfect calibration has ECE = 0.

Metrics
-------
* ECE (Expected Calibration Error)
* MCE (Maximum Calibration Error)
* Brier Score
* Log Loss
* Reliability diagram data (for Grafana / Matplotlib rendering)

Usage
-----
>>> cal_eval = CalibrationEvaluator(n_bins=15)
>>> report = cal_eval.evaluate(y_true, y_prob)
>>> print(f"ECE={report['ece']:.4f}  Brier={report['brier']:.4f}")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from backend.monitoring import get_logger

log = get_logger(__name__)


class CalibrationEvaluator:
    """
    Computes calibration metrics from raw probability scores.

    Parameters
    ----------
    n_bins    : number of confidence bins for reliability diagram
    """

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        y_true:   "list | np.ndarray",
        y_prob:   "list | np.ndarray",
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """
        Compute all calibration metrics.

        Parameters
        ----------
        y_true  : binary ground-truth labels (0 / 1)
        y_prob  : predicted malignancy probabilities ∈ [0, 1]

        Returns
        -------
        dict with ece, mce, brier, log_loss, reliability_diagram data
        """
        y_true = np.asarray(y_true, dtype=np.float32)
        y_prob = np.clip(np.asarray(y_prob, dtype=np.float32), 1e-7, 1 - 1e-7)

        ece, mce, diagram = self._reliability_diagram(y_true, y_prob)
        brier = self._brier_score(y_true, y_prob)
        log_loss = self._log_loss(y_true, y_prob)

        report = {
            "model_name":       model_name,
            "n_samples":        int(len(y_true)),
            "ece":              round(float(ece), 5),
            "mce":              round(float(mce), 5),
            "brier_score":      round(float(brier), 5),
            "log_loss":         round(float(log_loss), 5),
            "reliability_diagram": diagram,
            "calibration_grade": self._grade(ece),
        }
        log.info(
            "calibration_evaluated",
            model=model_name,
            ece=report["ece"],
            brier=report["brier_score"],
            grade=report["calibration_grade"],
        )
        return report

    # ------------------------------------------------------------------
    # Reliability diagram
    # ------------------------------------------------------------------
    def _reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple:
        """
        Compute bin-wise accuracy vs mean confidence.
        Returns (ece, mce, diagram_data).
        """
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        diagram = []
        ece = mce = 0.0
        total_n = len(y_true)

        for b in range(self.n_bins):
            mask = bin_indices == b
            if not mask.any():
                continue
            n_b        = mask.sum()
            accuracy_b = float(y_true[mask].mean())
            confidence_b = float(y_prob[mask].mean())
            gap        = abs(accuracy_b - confidence_b)

            ece += (n_b / total_n) * gap
            mce = max(mce, gap)

            diagram.append({
                "bin_lower":    round(float(bins[b]),    3),
                "bin_upper":    round(float(bins[b + 1]), 3),
                "bin_midpoint": round(float(bins[b] + (bins[b + 1] - bins[b]) / 2), 3),
                "accuracy":     round(accuracy_b, 4),
                "confidence":   round(confidence_b, 4),
                "gap":          round(gap, 4),
                "n_samples":    int(n_b),
            })

        return float(ece), float(mce), diagram

    # ------------------------------------------------------------------
    # Brier score & log loss
    # ------------------------------------------------------------------
    @staticmethod
    def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def _log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(-np.mean(
            y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
        ))

    # ------------------------------------------------------------------
    # Qualitative grade
    # ------------------------------------------------------------------
    @staticmethod
    def _grade(ece: float) -> str:
        if ece < 0.02:   return "Excellent"
        if ece < 0.05:   return "Good"
        if ece < 0.10:   return "Fair"
        if ece < 0.15:   return "Poor"
        return "Very Poor"
