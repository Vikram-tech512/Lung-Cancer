# tests/test_api.py
"""Integration tests for the FastAPI application."""
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.inference.service import app

client = TestClient(app)


def _create_dummy_image_buffer():
    img = Image.new("RGB", (64, 64), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestHealth:
    def test_health_endpoint(self):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"


class TestDashboard:
    def test_dashboard_returns_stats(self):
        res = client.get("/api/dashboard")
        assert res.status_code == 200
        data = res.json()
        assert "total_classifications" in data
        assert "benchmarks" in data
        assert "total_users" in data
        assert len(data["benchmarks"]) >= 3


class TestAuth:
    def test_register_and_login(self):
        import uuid
        unique_email = f"test_{uuid.uuid4().hex[:6]}@example.com"
        # Register
        res = client.post("/api/auth/register", json={
            "email": unique_email,
            "password": "TestPassword123!",
            "full_name": "Dr. Test"
        })
        assert res.status_code == 200
        user = res.json()
        assert user["email"] == unique_email

        # Login
        res = client.post("/api/auth/login", data={
            "username": unique_email,
            "password": "TestPassword123!"
        })
        assert res.status_code == 200
        token = res.json()
        assert "access_token" in token

        # Get profile
        res = client.get("/api/auth/me", headers={
            "Authorization": f"Bearer {token['access_token']}"
        })
        assert res.status_code == 200
        assert res.json()["email"] == unique_email

    def test_invalid_login(self):
        res = client.post("/api/auth/login", data={
            "username": "nonexistent_doctor@example.com",
            "password": "wrongpassword"
        })
        assert res.status_code == 401


class TestClinicalEndpoints:
    def test_sample_images(self):
        res = client.get("/api/sample-images")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

    def test_classify_with_dummy_image(self):
        buf = _create_dummy_image_buffer()
        res = client.post(
            "/api/classify",
            data={"architecture": "hybrid"},
            files={"file": ("test_ct.png", buf, "image/png")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "heatmap_base64" in data

    def test_classify_with_sample_id(self):
        res = client.post(
            "/api/classify",
            data={"architecture": "cnn", "sample_id": "malignant"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "predicted_class" in data

    def test_classify_unsupported_missing_input(self):
        res = client.post(
            "/api/classify",
            data={"architecture": "hybrid"},
        )
        assert res.status_code == 400

    def test_xai_explain_ct(self):
        buf = _create_dummy_image_buffer()
        res = client.post(
            "/api/xai",
            data={"architecture": "hybrid"},
            files={"file": ("test_ct.png", buf, "image/png")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "predicted_class" in data
        assert "gradcam_base64" in data
        assert "gradcam_pp_base64" in data
        assert "attention_base64" in data
        assert "lime_base64" in data

    def test_robustness_sweep(self):
        buf = _create_dummy_image_buffer()
        res = client.post(
            "/api/robustness/sweep",
            data={"architecture": "hybrid"},
            files={"file": ("test_ct.png", buf, "image/png")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "perturbations" in data
        assert "mean_robustness_degradation" in data
