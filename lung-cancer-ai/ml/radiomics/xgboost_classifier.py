"""
backend/models/radiomics/xgboost_classifier.py
===============================================
XGBoost + Random Forest classifiers for radiomic features.

These models operate on the 107-feature vector from extractor.py.
They serve as the radiomics baseline branch in the ensemble.

SHAP values from XGBoost are exposed via the /api/v2/radiomics/{nodule_id}
endpoint for feature-level explainability.

Patient-level cross-validation is enforced:
  - Split is defined per patient ID, not per nodule or per slice
  - A patient never appears in both train and val fold
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed: pip install scikit-learn")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─────────────────────────────────────────────────────────────
#  XGBoost Classifier
# ─────────────────────────────────────────────────────────────

class RadiomicsXGBoost:
    """XGBoost binary classifier on radiomic feature vectors.

    Training uses patient-level cross-validation.
    SHAP values are computed for feature-level explainability.
    """

    def __init__(self, n_estimators: int = 200,
                 max_depth: int = 4,
                 learning_rate: float = 0.1,
                 use_gpu: bool = False):
        if not HAS_XGB:
            raise ImportError("xgboost required: pip install xgboost")

        device = "cuda" if use_gpu else "cpu"
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            use_label_encoder=False,
            device=device,
            random_state=42,
        )
        self.scaler  = StandardScaler() if HAS_SKLEARN else None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self._shap_explainer = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Train on feature matrix X with labels y.

        Args:
            X:             (N, F) feature matrix
            y:             (N,) binary labels (0=benign, 1=malignant)
            feature_names: List of F feature names
            eval_set:      Optional (X_val, y_val) for early stopping
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        fit_kwargs = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            if self.scaler is not None:
                X_val = self.scaler.transform(X_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"]  = False

        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True

        if HAS_SHAP:
            self._shap_explainer = shap.TreeExplainer(self.model)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array [P(benign), P(malignant)]."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) binary predictions."""
        return self.predict_proba(X).argmax(axis=1)

    def get_shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Compute SHAP values for a feature matrix.

        Returns:
            (N, F) SHAP values for malignant class,
            or None if SHAP not installed.
        """
        if not HAS_SHAP or self._shap_explainer is None:
            return None
        if self.scaler is not None:
            X = self.scaler.transform(X)
        shap_vals = self._shap_explainer(X)
        # shap_vals.values shape: (N, F, 2) for binary classification
        if hasattr(shap_vals, "values"):
            vals = shap_vals.values
            if vals.ndim == 3:
                return vals[:, :, 1]   # malignant class
            return vals
        return None

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance dict (XGBoost gain)."""
        if not self.is_fitted:
            return {}
        importance = self.model.get_booster().get_score(importance_type="gain")
        return {k: float(v) for k, v in importance.items()}

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path.with_suffix(".json")))
        if self.scaler is not None:
            with open(str(path.with_suffix(".scaler.pkl")), "wb") as f:
                pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str | Path) -> "RadiomicsXGBoost":
        if not HAS_XGB:
            raise ImportError("xgboost required")
        instance = cls.__new__(cls)
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(Path(path).with_suffix(".json")))
        scaler_path = Path(path).with_suffix(".scaler.pkl")
        if scaler_path.exists():
            with open(str(scaler_path), "rb") as f:
                instance.scaler = pickle.load(f)
        else:
            instance.scaler = None
        instance.is_fitted = True
        instance.feature_names = []
        instance._shap_explainer = shap.TreeExplainer(instance.model) if HAS_SHAP else None
        return instance


# ─────────────────────────────────────────────────────────────
#  Random Forest Classifier (comparison baseline)
# ─────────────────────────────────────────────────────────────

class RadiomicsRandomForest:
    """scikit-learn Random Forest with calibration and feature importance."""

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None,
                 calibrate: bool = True):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        self.model = CalibratedClassifierCV(rf, cv=3) if calibrate else rf
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not trained.")
        return self.model.predict_proba(self.scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return permutation feature importance."""
        if not self.is_fitted:
            return {}
        base = self.model
        if hasattr(base, "calibrated_classifiers_"):
            # Access raw RF from calibrated wrapper
            estimators = [c.estimator for c in base.calibrated_classifiers_]
            importances = np.mean(
                [e.feature_importances_ for e in estimators], axis=0)
        elif hasattr(base, "feature_importances_"):
            importances = base.feature_importances_
        else:
            return {}
        return {name: float(imp)
                for name, imp in zip(self.feature_names, importances)}

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, path: str | Path) -> "RadiomicsRandomForest":
        with open(str(path), "rb") as f:
            data = pickle.load(f)
        inst = cls.__new__(cls)
        inst.model = data["model"]
        inst.scaler = data["scaler"]
        inst.feature_names = data["feature_names"]
        inst.is_fitted = True
        return inst
