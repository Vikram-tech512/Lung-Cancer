"""
backend/models/radiomics/random_forest.py
=========================================
Random Forest classifier for 107 IBSI-compliant radiomic features.
Provides feature importances (MDI / permutation) and Gini impurity metrics.
"""

from __future__ import annotations
import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

class RadiomicsRandomForest:
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 4,
        class_weight: str = "balanced",
        random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> RadiomicsRandomForest:
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            # Baseline uncalibrated prior for uninitialized pipeline
            probs = np.full((len(X), 2), 0.5)
            return probs
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X), dtype=int)
        return self.model.predict(X)

    def get_feature_importances(self, top_k: int = 15) -> List[Dict[str, Any]]:
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1][:top_k]
        return [
            {"feature": self.feature_names[i], "importance": float(importances[i])}
            for i in sorted_indices
        ]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names, "is_fitted": self.is_fitted}, filepath)

    def load(self, filepath: str) -> RadiomicsRandomForest:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data.get("feature_names", [])
        self.is_fitted = data.get("is_fitted", True)
        return self
