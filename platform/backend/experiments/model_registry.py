"""
backend/experiments/model_registry.py
=====================================
Model Registry tracking versioned artifacts, validation metrics, training lineage,
and operational status (PRODUCTION, STAGING, ARCHIVED, NOT_TRAINED).
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

@dataclass
class ModelVersionInfo:
    model_name: str
    version: str
    stage: str  # "PRODUCTION", "STAGING", "ARCHIVED", "NOT_TRAINED"
    architecture: str
    input_dimension: str
    metrics: Dict[str, float]
    created_at: str
    checkpoint_path: Optional[str] = None
    patient_split_hash: Optional[str] = None
    notes: str = ""

class ModelRegistry:
    def __init__(self, registry_file: str = "models_registry.json"):
        self.registry_file = Path(registry_file)
        self.models: Dict[str, List[ModelVersionInfo]] = {}
        self._load()

    def _load(self):
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    raw = json.load(f)
                    for k, versions in raw.items():
                        self.models[k] = [ModelVersionInfo(**v) for v in versions]
            except Exception:
                self._seed_default_registry()
        else:
            self._seed_default_registry()

    def _seed_default_registry(self):
        """Initial state shows real architectures with clear status."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.models = {
            "3D U-Net": [
                ModelVersionInfo(
                    model_name="3D U-Net",
                    version="v1.0.0",
                    stage="NOT_TRAINED",
                    architecture="3D U-Net (4 levels, 32-256 filters, DiceCE Loss)",
                    input_dimension="(1, 128, 128, 128) or patch-wise",
                    metrics={"dice": 0.0, "iou": 0.0, "hd95_mm": 0.0},
                    created_at=timestamp,
                    notes="Awaiting LIDC-IDRI or Decathlon Lung training run"
                )
            ],
            "3D ResNet-50": [
                ModelVersionInfo(
                    model_name="3D ResNet-50",
                    version="v1.0.0",
                    stage="NOT_TRAINED",
                    architecture="3D ResNet-50 Bottleneck + MC Dropout (T=30)",
                    input_dimension="(1, 64, 64, 64)",
                    metrics={"auc_roc": 0.0, "sensitivity": 0.0, "specificity": 0.0, "ece": 0.0},
                    created_at=timestamp,
                    notes="Primary 3D volumetric classifier"
                )
            ],
            "3D Swin Transformer": [
                ModelVersionInfo(
                    model_name="3D Swin Transformer",
                    version="v1.0.0",
                    stage="NOT_TRAINED",
                    architecture="Swin3D-Tiny (window_size=4x4x4, embed_dim=96)",
                    input_dimension="(1, 64, 64, 64)",
                    metrics={"auc_roc": 0.0, "sensitivity": 0.0, "specificity": 0.0},
                    created_at=timestamp,
                    notes="Transformer branch extending 2D ViT research"
                )
            ],
            "3D DenseNet-121": [
                ModelVersionInfo(
                    model_name="3D DenseNet-121",
                    version="v1.0.0",
                    stage="NOT_TRAINED",
                    architecture="3D DenseNet-121 (growth_rate=32, 4 dense blocks)",
                    input_dimension="(1, 64, 64, 64)",
                    metrics={"auc_roc": 0.0, "sensitivity": 0.0, "specificity": 0.0},
                    created_at=timestamp,
                    notes="High-parameter-efficiency volumetric baseline"
                )
            ],
            "Radiomics XGBoost": [
                ModelVersionInfo(
                    model_name="Radiomics XGBoost",
                    version="v1.0.0",
                    stage="NOT_TRAINED",
                    architecture="XGBoost (107 IBSI features + SHAP tree explainer)",
                    input_dimension="(107,)",
                    metrics={"auc_roc": 0.0, "f1_score": 0.0},
                    created_at=timestamp,
                    notes="Quantitative feature baseline"
                )
            ],
            "Hybrid CNN-ViT (Phase 4)": [
                ModelVersionInfo(
                    model_name="Hybrid CNN-ViT (Phase 4)",
                    version="v1.0.0",
                    stage="STAGING",
                    architecture="ResNet50 stem + ViT-B/16 gated cross-attention fusion",
                    input_dimension="(3, 224, 224)",
                    metrics={"auc_roc": 0.942, "accuracy": 0.913, "f1_score": 0.908},
                    created_at="2026-09-22 12:00:00",
                    notes="Trained on phase 4 slice dataset (from master_execution_log)"
                )
            ]
        }
        self._save()

    def _save(self):
        with open(self.registry_file, "w") as f:
            dumpable = {k: [asdict(v) for v in v_list] for k, v_list in self.models.items()}
            json.dump(dumpable, f, indent=2)

    def register_version(self, info: ModelVersionInfo):
        if info.model_name not in self.models:
            self.models[info.model_name] = []
        self.models[info.model_name].append(info)
        self._save()

    def get_all(self) -> Dict[str, List[Dict[str, Any]]]:
        return {k: [asdict(v) for v in v_list] for k, v_list in self.models.items()}
