"""
backend/pipeline/evaluate/curves.py
=======================================
ROC, Precision-Recall, and DET curve generation with bootstrap CIs.

Usage
-----
>>> gen = CurveGenerator(n_bootstrap=1000, ci=0.95)
>>> roc  = gen.roc_curve(y_true, y_prob)
>>> pr   = gen.pr_curve(y_true, y_prob)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend.monitoring import get_logger

log = get_logger(__name__)


class CurveGenerator:
    """
    Generates ROC, PR, and DET curves with optional bootstrap confidence intervals.

    Parameters
    ----------
    n_bootstrap : number of bootstrap iterations for CI
    ci          : confidence level (e.g. 0.95 for 95% CI)
    """

    def __init__(self, n_bootstrap: int = 1000, ci: float = 0.95) -> None:
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self._alpha = 1.0 - ci

    # ------------------------------------------------------------------
    # ROC Curve
    # ------------------------------------------------------------------
    def roc_curve(
        self,
        y_true: "list | np.ndarray",
        y_prob: "list | np.ndarray",
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        from sklearn.metrics import roc_curve as sk_roc, roc_auc_score

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        fpr, tpr, thresholds = sk_roc(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))

        # Optimal threshold — Youden-J maximisation
        youden = tpr - fpr
        opt_idx = int(np.argmax(youden))

        # Bootstrap AUC CI
        auc_ci_lower, auc_ci_upper = self._bootstrap_auc(y_true, y_prob)

        result = {
            "model_name":    model_name,
            "auc":           round(auc, 4),
            "auc_ci_lower":  round(auc_ci_lower, 4),
            "auc_ci_upper":  round(auc_ci_upper, 4),
            "ci_level":      self.ci,
            "optimal_threshold": round(float(thresholds[opt_idx]), 4),
            "optimal_fpr":   round(float(fpr[opt_idx]), 4),
            "optimal_tpr":   round(float(tpr[opt_idx]), 4),
            "curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
        }
        log.info("roc_curve_generated", model=model_name, auc=result["auc"])
        return result

    # ------------------------------------------------------------------
    # Precision-Recall Curve
    # ------------------------------------------------------------------
    def pr_curve(
        self,
        y_true: "list | np.ndarray",
        y_prob: "list | np.ndarray",
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        from sklearn.metrics import (
            precision_recall_curve,
            average_precision_score,
        )

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap = float(average_precision_score(y_true, y_prob))

        # Optimal F1 threshold
        f1_scores = (
            2 * precision[:-1] * recall[:-1]
            / np.maximum(precision[:-1] + recall[:-1], 1e-8)
        )
        opt_idx = int(np.argmax(f1_scores))

        return {
            "model_name":        model_name,
            "average_precision": round(ap, 4),
            "optimal_threshold": round(float(thresholds[opt_idx]), 4),
            "optimal_f1":        round(float(f1_scores[opt_idx]), 4),
            "optimal_precision": round(float(precision[opt_idx]), 4),
            "optimal_recall":    round(float(recall[opt_idx]), 4),
            "curve": {
                "precision":  precision.tolist(),
                "recall":     recall.tolist(),
                "thresholds": thresholds.tolist(),
            },
        }

    # ------------------------------------------------------------------
    # Multi-model comparison
    # ------------------------------------------------------------------
    def compare_roc(
        self,
        model_results: Dict[str, Dict],  # {"ModelA": {"y_true":..., "y_prob":...}}
    ) -> Dict[str, Any]:
        """Generate multiple ROC curves for side-by-side comparison."""
        curves = {}
        for name, data in model_results.items():
            curves[name] = self.roc_curve(
                data["y_true"], data["y_prob"], model_name=name
            )
        # Sort by AUC descending
        ranked = sorted(curves.items(), key=lambda x: -x[1]["auc"])
        return {
            "models": dict(ranked),
            "best_model": ranked[0][0] if ranked else None,
        }

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------
    def _bootstrap_auc(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> tuple[float, float]:
        from sklearn.metrics import roc_auc_score

        rng    = np.random.default_rng(42)
        n      = len(y_true)
        aucs   = []

        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            y_t = y_true[idx]
            y_p = y_prob[idx]
            if len(np.unique(y_t)) < 2:
                continue
            aucs.append(float(roc_auc_score(y_t, y_p)))

        if not aucs:
            return 0.0, 1.0
        lower = float(np.percentile(aucs, 100 * self._alpha / 2))
        upper = float(np.percentile(aucs, 100 * (1 - self._alpha / 2)))
        return lower, upper
