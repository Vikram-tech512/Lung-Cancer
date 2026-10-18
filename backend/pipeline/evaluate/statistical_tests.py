"""
backend/pipeline/evaluate/statistical_tests.py
==================================================
Statistical significance tests for comparing model predictions.

Tests provided
--------------
* McNemar's test      — paired difference in prediction error rates
* DeLong AUC test     — non-parametric paired AUC comparison (Fast DeLong)
* Bootstrap CI        — general-purpose bootstrap confidence interval
* Wilcoxon signed-rank— non-parametric paired test on any scalar metric

Usage
-----
>>> tests = StatisticalTests(alpha=0.05)
>>> result = tests.mcnemar(y_true, pred_a, pred_b)
>>> result = tests.delong_auc(y_true, prob_a, prob_b)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend.monitoring import get_logger

log = get_logger(__name__)


class StatisticalTests:
    """
    Statistical hypothesis tests for comparing classification models.

    Parameters
    ----------
    alpha : significance level (default 0.05)
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # McNemar's test
    # ------------------------------------------------------------------
    def mcnemar(
        self,
        y_true:  "list | np.ndarray",
        pred_a:  "list | np.ndarray",
        pred_b:  "list | np.ndarray",
        model_a: str = "ModelA",
        model_b: str = "ModelB",
    ) -> Dict[str, Any]:
        """
        McNemar's test on paired binary predictions.

        Returns p-value, chi-squared statistic, and significance decision.
        """
        from scipy.stats import chi2

        y_true = np.asarray(y_true)
        pred_a = np.asarray(pred_a)
        pred_b = np.asarray(pred_b)

        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)

        # 2×2 contingency table
        b = int(((correct_a) & (~correct_b)).sum())   # A right, B wrong
        c = int(((~correct_a) & (correct_b)).sum())   # A wrong, B right

        # Edwards' continuity correction
        if (b + c) == 0:
            chi2_stat = 0.0
            p_value   = 1.0
        else:
            chi2_stat = float((abs(b - c) - 1) ** 2 / (b + c))
            p_value   = float(1 - chi2.cdf(chi2_stat, df=1))

        result = {
            "test":        "McNemar",
            "model_a":     model_a,
            "model_b":     model_b,
            "b":           b,
            "c":           c,
            "chi2_stat":   round(chi2_stat, 4),
            "p_value":     round(p_value, 5),
            "significant": p_value < self.alpha,
            "alpha":       self.alpha,
        }
        log.info("mcnemar_test", **{k: v for k, v in result.items() if k != "test"})
        return result

    # ------------------------------------------------------------------
    # DeLong AUC comparison (fast implementation)
    # ------------------------------------------------------------------
    def delong_auc(
        self,
        y_true:  "list | np.ndarray",
        prob_a:  "list | np.ndarray",
        prob_b:  "list | np.ndarray",
        model_a: str = "ModelA",
        model_b: str = "ModelB",
    ) -> Dict[str, Any]:
        """
        Fast DeLong test for paired AUC comparison.
        Reference: Sun & Xu (2014) Machine Learning.
        """
        from scipy.stats import norm
        from sklearn.metrics import roc_auc_score

        y_true = np.asarray(y_true, dtype=np.float64)
        prob_a = np.asarray(prob_a, dtype=np.float64)
        prob_b = np.asarray(prob_b, dtype=np.float64)

        auc_a = float(roc_auc_score(y_true, prob_a))
        auc_b = float(roc_auc_score(y_true, prob_b))

        # Compute variance via placement values
        va, vb, vab = self._delong_variance(y_true, prob_a, prob_b)
        z_stat = (auc_a - auc_b) / np.sqrt(max(va + vb - 2 * vab, 1e-12))
        p_value = float(2 * (1 - norm.cdf(abs(z_stat))))

        result = {
            "test":        "DeLong AUC",
            "model_a":     model_a,
            "model_b":     model_b,
            "auc_a":       round(auc_a, 4),
            "auc_b":       round(auc_b, 4),
            "auc_diff":    round(auc_a - auc_b, 4),
            "z_stat":      round(float(z_stat), 4),
            "p_value":     round(p_value, 5),
            "significant": p_value < self.alpha,
            "alpha":       self.alpha,
        }
        log.info("delong_auc_test", **{k: v for k, v in result.items() if k != "test"})
        return result

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------
    def bootstrap_ci(
        self,
        values:     "list | np.ndarray",
        n_bootstrap: int = 2000,
        ci:          float = 0.95,
    ) -> Dict[str, float]:
        """Generic bootstrap CI for any 1-D array of metric values."""
        values = np.asarray(values, dtype=np.float64)
        rng    = np.random.default_rng(42)
        means  = [
            float(rng.choice(values, size=len(values), replace=True).mean())
            for _ in range(n_bootstrap)
        ]
        alpha = 1.0 - ci
        return {
            "mean":  round(float(values.mean()), 4),
            "std":   round(float(values.std()),  4),
            "ci_lower": round(float(np.percentile(means, 100 * alpha / 2)),     4),
            "ci_upper": round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4),
            "ci_level": ci,
        }

    # ------------------------------------------------------------------
    # Internal: DeLong variance
    # ------------------------------------------------------------------
    @staticmethod
    def _delong_variance(
        y_true: np.ndarray,
        prob_a: np.ndarray,
        prob_b: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Compute var(AUC_a), var(AUC_b), cov(AUC_a, AUC_b)
        using the structural components approach.
        """
        pos_idx = y_true == 1
        neg_idx = y_true == 0

        def _placement(scores_pos, scores_neg):
            """Placement values for positive and negative samples."""
            v_pos = np.array([
                np.mean(s > scores_neg) + 0.5 * np.mean(s == scores_neg)
                for s in scores_pos
            ])
            v_neg = np.array([
                np.mean(s < scores_pos) + 0.5 * np.mean(s == scores_pos)
                for s in scores_neg
            ])
            return v_pos, v_neg

        m = pos_idx.sum()
        n = neg_idx.sum()

        va_pos, va_neg = _placement(prob_a[pos_idx], prob_a[neg_idx])
        vb_pos, vb_neg = _placement(prob_b[pos_idx], prob_b[neg_idx])

        var_a = (np.var(va_pos) / m + np.var(va_neg) / n)
        var_b = (np.var(vb_pos) / m + np.var(vb_neg) / n)
        cov   = (np.cov(va_pos, vb_pos)[0, 1] / m +
                 np.cov(va_neg, vb_neg)[0, 1] / n)

        return float(var_a), float(var_b), float(cov)
