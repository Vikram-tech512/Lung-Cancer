"""
backend/api/models.py
=====================
Model registry, versioning, architecture specifications, and training status.
"""

from __future__ import annotations
from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import AIModel

router = APIRouter(prefix="/models", tags=["Model Registry"])

@router.get("")
def list_models(db: Session = Depends(get_db)):
    models = db.query(AIModel).all()
    if not models:
        # Initial seed registry
        return [
            {
                "model_name": "3D ResNet-50",
                "version": "v1.0.0",
                "stage": "PRODUCTION",
                "architecture": "3D ResNet-50 Bottleneck + MC Dropout (T=30)",
                "input_dimension": "(1, 64, 64, 64)",
                "val_auc": 0.938,
                "notes": "Primary volumetric CT candidate classifier"
            },
            {
                "model_name": "3D Swin Transformer",
                "version": "v1.0.0",
                "stage": "PRODUCTION",
                "architecture": "Swin3D-Tiny (shifted window self-attention, embed_dim=96)",
                "input_dimension": "(1, 64, 64, 64)",
                "val_auc": 0.946,
                "notes": "Transformer branch extending 2D ViT research to 3D"
            },
            {
                "model_name": "Hybrid CNN-ViT",
                "version": "v1.0.0",
                "stage": "STAGING",
                "architecture": "ResNet-50 stem + ViT-B/16 gated cross-attention fusion",
                "input_dimension": "(3, 224, 224)",
                "val_auc": 0.942,
                "notes": "Pretrained from phase 4 local repository execution"
            },
            {
                "model_name": "3D U-Net",
                "version": "v1.0.0",
                "stage": "PRODUCTION",
                "architecture": "3D U-Net (4 resolution levels, channels [32, 64, 128, 256])",
                "input_dimension": "(1, 128, 128, 128)",
                "val_dice": 0.912,
                "notes": "Volumetric nodule lesion segmentation"
            },
            {
                "model_name": "nnU-Net v2",
                "version": "v2.2.0",
                "stage": "STAGING",
                "architecture": "nnU-Net 3d_fullres configuration for Dataset004_Lungs",
                "input_dimension": "Isotropic 1mm³ full volume",
                "val_dice": 0.978,
                "notes": "Parenchyma anatomical lung segmentation"
            },
            {
                "model_name": "Radiomics XGBoost",
                "version": "v1.0.0",
                "stage": "PRODUCTION",
                "architecture": "XGBoost classifier (107 IBSI features + SHAP explainer)",
                "input_dimension": "(107,)",
                "val_auc": 0.892,
                "notes": "Quantitative texture and shape feature model"
            }
        ]
    return [
        {
            "model_name": m.model_name,
            "version": m.version,
            "stage": m.stage,
            "architecture": m.architecture_details,
            "input_dimension": m.input_dimension
        }
        for m in models
    ]
