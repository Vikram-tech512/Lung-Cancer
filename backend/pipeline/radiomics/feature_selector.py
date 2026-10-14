"""
backend/pipeline/radiomics/feature_selector.py
=================================================
Radiomics feature selection pipeline.

Methods
-------
1. Variance threshold   — removes near-zero-variance features
2. Correlation filter   — removes highly correlated pairs (Pearson r > threshold)
3. LASSO / ElasticNet   — L1-regularised linear model selection
4. Recursive Feature Elimination (RFE) — wrapper method with cross-validation
5. Mutual Information    — filter by MI with label

Usage
-----
>>> selector = FeatureSelector(method="lasso", n_features=30)
>>> X_selected, feature_names = selector.fit_transform(X_train, y_train, feature_names)
>>> X_test_selected = selector.transform(X_test)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.monitoring import get_logger

log = get_logger(__name__)


class FeatureSelector:
    """
    Multi-strategy radiomics feature selector.

    Parameters
    ----------
    method       : "variance" | "correlation" | "lasso" | "rfe" | "mutual_info"
    n_features   : target number of features to select (used by lasso, rfe, mutual_info)
    corr_threshold : Pearson |r| threshold for correlation filtering
    var_threshold  : minimum variance to keep a feature
    alpha         : regularisation strength for LASSO
    """

    def __init__(
        self,
        method: str = "lasso",
        n_features: int = 30,
        corr_threshold: float = 0.95,
        var_threshold: float = 0.01,
        alpha: float = 0.01,
    ) -> None:
        self.method         = method
        self.n_features     = n_features
        self.corr_threshold = corr_threshold
        self.var_threshold  = var_threshold
        self.alpha          = alpha
        self._selected_idx: Optional[np.ndarray] = None
        self._selected_names: Optional[List[str]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the selector and return (X_selected, selected_feature_names).

        Parameters
        ----------
        X            : (n_samples, n_features)
        y            : (n_samples,) binary labels
        feature_names: list of feature names (length = n_features)
        """
        names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_sel, sel_names = self._fit(X, y, names)
        self._fitted = True
        return X_sel, sel_names

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted selection to a new feature matrix."""
        if not self._fitted or self._selected_idx is None:
            raise RuntimeError("Call fit_transform() before transform()")
        return X[:, self._selected_idx]

    # ------------------------------------------------------------------
    # Strategy dispatcher
    # ------------------------------------------------------------------
    def _fit(
        self, X: np.ndarray, y: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        # Step 1: always apply variance threshold first
        X, names, keep = self._variance_filter(X, names)
        log.info("feature_selection_variance", remaining=len(names))

        # Step 2: correlation filter (always applied before wrapper methods)
        if self.method in ("lasso", "rfe", "mutual_info", "correlation"):
            X, names = self._correlation_filter(X, names)
            log.info("feature_selection_correlation", remaining=len(names))

        # Step 3: method-specific selection
        if self.method == "lasso":
            X, names = self._lasso_select(X, y, names)
        elif self.method == "rfe":
            X, names = self._rfe_select(X, y, names)
        elif self.method == "mutual_info":
            X, names = self._mutual_info_select(X, y, names)

        # Record selected column indices relative to original X
        self._selected_names = names
        self._selected_idx   = keep
        log.info(
            "feature_selection_complete",
            method=self.method,
            selected=len(names),
        )
        return X, names

    # ------------------------------------------------------------------
    # Individual methods
    # ------------------------------------------------------------------
    def _variance_filter(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        variances = X.var(axis=0)
        keep = np.where(variances > self.var_threshold)[0]
        return X[:, keep], [names[i] for i in keep], keep

    def _correlation_filter(
        self, X: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        corr = np.corrcoef(X.T)
        to_drop = set()
        n = corr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > self.corr_threshold:
                    to_drop.add(j)   # drop the later feature in a correlated pair
        keep = [i for i in range(n) if i not in to_drop]
        return X[:, keep], [names[i] for i in keep]

    def _lasso_select(
        self, X: np.ndarray, y: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
        lasso.fit(X_scaled, y)
        keep = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        if len(keep) == 0:
            keep = np.argsort(np.abs(lasso.coef_))[-self.n_features:]
        keep = keep[:self.n_features]
        self._selected_idx = keep
        return X[:, keep], [names[i] for i in keep]

    def _rfe_select(
        self, X: np.ndarray, y: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        from sklearn.feature_selection import RFECV
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfe = RFECV(estimator, step=5, cv=3, scoring="roc_auc", n_jobs=-1)
        rfe.fit(X, y)
        keep = np.where(rfe.support_)[0][:self.n_features]
        self._selected_idx = keep
        return X[:, keep], [names[i] for i in keep]

    def _mutual_info_select(
        self, X: np.ndarray, y: np.ndarray, names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(X, y, random_state=42)
        keep = np.argsort(mi_scores)[-self.n_features:][::-1]
        self._selected_idx = keep
        return X[:, keep], [names[i] for i in keep]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        return {
            "method":   self.method,
            "selected": len(self._selected_names or []),
            "features": self._selected_names or [],
        }
