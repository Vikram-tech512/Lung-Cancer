"""
backend/tests/test_dicom_upload.py
====================================
Tests for genuine DICOM upload, archive inspection, multi-vendor checking,
quality validation, and data integrity safeguards.
"""

import io
import zipfile
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import numpy as np

from backend.api.service import app, STUDY_STORE
from backend.tests.test_hospital_interop import create_synthetic_slice


@pytest.fixture
def client():
    return TestClient(app)


def test_upload_missing_file_returns_400(client):
    """Uploading without a file payload must be rejected with 400."""
    res = client.post("/api/v2/dicom/upload")
    assert res.status_code == 400
    assert "No DICOM file or archive provided" in res.json()["detail"]


def test_upload_empty_file_returns_400(client):
    """Uploading an empty file must be rejected with 400."""
    files = {"file": ("empty.dcm", b"", "application/octet-stream")}
    res = client.post("/api/v2/dicom/upload", files=files)
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_upload_non_dicom_archive_returns_400(client):
    """Uploading a zip with text files must report no readable DICOM files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("notes.txt", "This is not a DICOM file")
    buf.seek(0)

    files = {"file": ("corrupt.zip", buf.getvalue(), "application/zip")}
    res = client.post("/api/v2/dicom/upload", files=files)
    assert res.status_code == 400
    assert "no readable dicom files" in res.json()["detail"].lower()


def test_upload_genuine_dicom_zip_success(client):
    """Uploading a genuine zip of DICOM slices must compute real slice counts,
    extract real vendor metadata, perform de-identification, and mark is_synthetic=False.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(10):
            ds = create_synthetic_slice(
                z_pos=i * 2.0,
                manufacturer="TOSHIBA",
                slice_thickness=2.0
            )
            slice_buf = io.BytesIO()
            ds.save_as(slice_buf, write_like_original=False)
            zf.writestr(f"slice_{i:03d}.dcm", slice_buf.getvalue())
    buf.seek(0)

    files = {"file": ("toshiba_study.zip", buf.getvalue(), "application/zip")}
    res = client.post("/api/v2/dicom/upload", files=files)
    assert res.status_code == 200
    data = res.json()

    assert "study_id" in data
    study_id = data["study_id"]
    assert study_id in STUDY_STORE

    stored = STUDY_STORE[study_id]
    # Verify metadata is genuinely extracted, NOT hardcoded GE Apex / 196 slices!
    assert stored["is_synthetic"] is False
    assert stored["slice_count"] == 10
    assert stored["slice_thickness"] == 2.0
    assert stored["compatibility_report"]["vendor"] == "CANON/TOSHIBA"
    assert stored["deidentification_audit"]["stage_status"] == "PASS"
    assert stored["patient_id"].startswith("ANON-")
    assert stored["nodules"] == []


def test_upload_mhd_raw_archive_success(client):
    """Uploading a zip containing a .mhd/.raw pair must successfully ingest and register the volume."""
    subset0 = Path("data/LUNA16/subset0")
    mhd_files = list(subset0.glob("*.mhd"))
    if not mhd_files:
        pytest.skip("No LUNA16 .mhd files found on disk.")

    mhd_path = mhd_files[0]
    raw_path = mhd_path.with_suffix(".raw")
    if not raw_path.exists():
        pytest.skip(f"Accompanying raw file {raw_path} not found.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(mhd_path, arcname=mhd_path.name)
        zf.write(raw_path, arcname=raw_path.name)
    buf.seek(0)

    files = {"file": ("luna16_sample.zip", buf.getvalue(), "application/zip")}
    res = client.post("/api/v2/dicom/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert "study_id" in data
    study_id = data["study_id"]
    assert study_id in STUDY_STORE
    stored = STUDY_STORE[study_id]
    assert stored["is_synthetic"] is False
    assert stored["modality"] == "CT"
    assert stored["slice_count"] > 0
    assert stored["mhd_file"] is not None

def test_list_examples_when_env_unset(client, monkeypatch):
    monkeypatch.delenv("LUNA16_DATA_DIR", raising=False)
    res = client.get("/api/v2/examples")
    assert res.status_code == 200
    data = res.json()
    assert data["examples"] == []
    assert "not configured" in data["message"].lower() or "unset" in data["message"].lower()


def test_load_example_rejects_path_traversal(client):
    for bad_name in ["../secret.mhd", "..\\secret.mhd", "subset0/scan.mhd", "sub/../scan.mhd"]:
        res = client.post("/api/v2/examples", json={"filename": bad_name})
        assert res.status_code == 400
        assert "path separators" in res.json()["detail"].lower() or "traversal" in res.json()["detail"].lower()


def test_load_example_missing_file_returns_404(client, monkeypatch, tmp_path):
    monkeypatch.setenv("LUNA16_DATA_DIR", str(tmp_path))
    res = client.post("/api/v2/examples", json={"filename": "non_existent_scan.mhd"})
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()
