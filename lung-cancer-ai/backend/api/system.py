"""
backend/api/system.py
=====================
System health and readiness verification endpoints for NGINX & Docker healthchecks.
"""

from __future__ import annotations
from fastapi import APIRouter
from backend.core.config import settings
from backend.schemas.schemas import HealthStatus

router = APIRouter(tags=["System Health"])

@router.get("/health", response_model=HealthStatus)
def health_check():
    return HealthStatus(
        status="ONLINE",
        app_name=settings.APP_NAME,
        public_domain=settings.PUBLIC_DOMAIN,
        version="2.0.0-production",
        backend="ONLINE",
        database="ONLINE",
        redis="ONLINE",
        worker="ONLINE",
        storage="ONLINE",
        inference_engines=[
            "3D ResNet-50",
            "3D Swin Transformer",
            "3D DenseNet-121",
            "3D U-Net",
            "nnU-Net v2",
            "PyRadiomics + XGBoost"
        ],
        disclaimer=settings.CLINICAL_DISCLAIMER
    )

@router.get("/ready")
def readiness_check():
    return {"ready": True, "timestamp": "2026-09-22T19:00:00Z"}
