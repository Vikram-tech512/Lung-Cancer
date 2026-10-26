"""
backend/pipeline/evaluate/curves.py
=======================================
ROC, Precision-Recall, and DET curve generation with bootstrap CIs.

Usage
-----
>>> gen = CurveGenerator(n_bootstrap=1000, ci=0.95, seed=42)
>>> roc  = gen.roc_curve(y_true, y_prob, patient_ids=patient_ids)
>>> pr   = gen.pr_curve(y_true, y_prob)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from backend.monitoring import get_logger

log = get_logger(__name__)


class CurveGenerator:
    """
    Generates ROC, PR, and DET curves with optional bootstrap confidence intervals.

    Parameters
    ----------
    n_bootstrap : number of patient-cluster bootstrap iterations for CI
    ci          : confidence level (e.g. 0.95 for 95% CI)
    seed        : reproducible random seed for cluster resampling
    """

    def __init__(
        self, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.ci = ci
        self._alpha = 1.0 - ci
        self.seed = seed

    # ------------------------------------------------------------------
    # ROC Curve
    # ------------------------------------------------------------------
    def roc_curve(
        self,
        y_true: "list | np.ndarray",
        y_prob: "list | np.ndarray",
        model_name: str = "Model",
        patient_ids: "list | np.ndarray | None" = None,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        from sklearn.metrics import roc_curve as sk_roc, roc_auc_score

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        fpr, tpr, thresholds = sk_roc(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))

        # Optimal threshold — Youden-J maximisation
        youden = tpr - fpr
        opt_idx = int(np.argmax(youden))

        # Candidate-level resampling understates uncertainty when patients
        # contribute multiple candidates. Require cluster identifiers.
        if self.n_bootstrap:
            if patient_ids is None:
                raise ValueError("patient_ids are required for patient-cluster bootstrap CIs")
            auc_ci = self._bootstrap_auc(y_true, y_prob, patient_ids, seed=seed)
        else:
            auc_ci = None

        result = {
            "model_name":    model_name,
            "auc":           round(auc, 4),
            "auc_ci_lower":  round(auc_ci[0], 4) if auc_ci else None,
            "auc_ci_upper":  round(auc_ci[1], 4) if auc_ci else None,
            "ci_level":      self.ci if auc_ci else None,
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
                data["y_true"], data["y_prob"], model_name=name,
                patient_ids=data.get("patient_ids"),
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
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        patient_ids: "list | np.ndarray",
        seed: int | None = None,
    ) -> tuple[float, float] | None:
        from sklearn.metrics import roc_auc_score

        patient_ids = np.asarray(patient_ids)
        if len(y_true) != len(y_prob) or len(y_true) != len(patient_ids):
            raise ValueError("y_true, y_prob, and patient_ids must have equal lengths")
        patients, inverse = np.unique(patient_ids, return_inverse=True)
        if not len(patients):
            return None
        patient_rows = [np.flatnonzero(inverse == i) for i in range(len(patients))]
        rng = np.random.default_rng(self.seed if seed is None else seed)
        aucs = []

        for _ in range(self.n_bootstrap):
            sampled_patients = rng.integers(0, len(patients), size=len(patients))
            idx = np.concatenate([patient_rows[i] for i in sampled_patients])
            y_t = y_true[idx]
            y_p = y_prob[idx]
            if len(np.unique(y_t)) < 2:
                continue
            aucs.append(float(roc_auc_score(y_t, y_p)))

        if not aucs:
            return None
        lower = float(np.percentile(aucs, 100 * self._alpha / 2))
        upper = float(np.percentile(aucs, 100 * (1 - self._alpha / 2)))
        return lower, upper
