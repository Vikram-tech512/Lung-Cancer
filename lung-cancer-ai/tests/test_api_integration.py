"""
tests/test_api_integration.py
==============================
Integration tests verifying REST endpoints across the FastAPI service.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "3D ResNet-50" in data["inference_engines"]
    assert "RESEARCH USE ONLY" in data["disclaimer"]

def test_domain_settings():
    response = client.get("/api/settings/domain")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "public_domain" in data

def test_models_registry():
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 4
    names = [m["model_name"] for m in models]
    assert "3D ResNet-50" in names
    assert "3D Swin Transformer" in names

def test_datasets_registry():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    datasets = response.json()
    names = [d["dataset_name"] for d in datasets]
    assert "LIDC-IDRI" in names
    assert any("NLST" in name for name in names)

def test_studies_list():
    response = client.get("/api/studies")
    assert response.status_code == 200
    studies = response.json()
    assert isinstance(studies, list)

def test_verdict_endpoint():
    response = client.get("/api/verdict/LESION-DEMO-01")
    assert response.status_code == 200
    verdict = response.json()
    assert verdict["consensus_verdict"] == "MALIGNANT"
    assert verdict["consensus_probability"] == 0.849
    assert verdict["morphometry"]["effective_diameter_mm"] == 14.8

def test_audit_trail_endpoint():
    response = client.get("/api/audit")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "events" in data

def test_reviews_endpoint_get():
    response = client.get("/api/reviews/study/CASE-DEMO-LIDC-0087")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
