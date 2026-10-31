"""
backend/tests/test_continuous_reporting.py
===========================================
Tests for:
1. Upload-differs-per-scan (two different studies produce different slices & hashes)
2. Asynchronous job queue concurrency and stage progression (Stage 0 to 10)
3. Location-to-view synchronization (nodule centroid -> slice -> on-demand XAI)
4. Per-scan compute metering, resource tracking, and cost transparency
5. Human-in-the-loop review actions (accept/reject/adjust) and audit trail persistence
6. Longitudinal follow-up comparison view
"""

import io
import time
import zipfile
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.api.service import app, STUDY_STORE, _seed_demo_case
from backend.tests.test_hospital_interop import create_synthetic_slice


@pytest.fixture(autouse=True)
def setup_demo_store():
    _seed_demo_case()


@pytest.fixture
def client():
    return TestClient(app)


def _create_study_zip(
    base_hu: float = -600.0,
    manufacturer: str = "SIEMENS",
    z_step: float = 1.25,
    nodule_center: tuple = None,
    nodule_radius: int = 8,
    nodule_hu: float = 40.0,
) -> io.BytesIO:
    """Creates an in-memory zip of 6 CT slices with specified HU characteristics and optional embedded nodule."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for z in range(6):
            ds = create_synthetic_slice(
                z_pos=z * z_step,
                manufacturer=manufacturer,
                slice_thickness=z_step,
            )
            # In DICOM CT: Stored Pixel = HU - RescaleIntercept = HU - (-1024) = HU + 1024
            base_pv = np.clip(int(base_hu + 1024.0), 0, 4095).astype(np.uint16)
            arr = np.full((512, 512), base_pv, dtype=np.uint16)
            if nodule_center is not None:
                nz, ny, nx = nodule_center
                if abs(z - nz) <= 1:
                    y_grid, x_grid = np.ogrid[:512, :512]
                    dist_sq = (y_grid - ny) ** 2 + (x_grid - nx) ** 2 + ((z - nz) * 4) ** 2
                    mask = dist_sq <= (nodule_radius ** 2)
                    nod_pv = np.clip(int(nodule_hu + 1024.0), 0, 4095).astype(np.uint16)
                    arr[mask] = nod_pv

            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelData = arr.tobytes()
            ds.Rows, ds.Columns = 512, 512

            slice_buf = io.BytesIO()
            ds.save_as(slice_buf, write_like_original=False)
            zf.writestr(f"slice_{z:03d}.dcm", slice_buf.getvalue())
    buf.seek(0)
    return buf


def test_upload_differs_per_scan(client):
    """
    WORKSTREAM 2 & 6 REQUIREMENT:
    Assert uploading two structurally different CT volumes with different nodule positions produces:
      1. Different slice hashes
      2. Different rendered image bytes
      3. Different detected candidate slice indices (e.g. slice 2 vs 4)
      4. Different bounding boxes
    Proves no sample image, fixed slice, or hardcoded coordinates are reused.
    """
    # 1. Upload Study A (Nodule embedded at Z=2, Y=160, X=160)
    zip_a = _create_study_zip(
        base_hu=-600.0,
        manufacturer="GE",
        z_step=1.25,
        nodule_center=(2, 160, 160),
        nodule_radius=8,
        nodule_hu=50.0,
    )
    res_a = client.post("/api/v2/dicom/upload", files={"file": ("study_a.zip", zip_a.getvalue(), "application/zip")})
    assert res_a.status_code == 200
    study_id_a = res_a.json()["study_id"]

    # 2. Upload Study B (Nodule embedded at Z=4, Y=320, X=360)
    zip_b = _create_study_zip(
        base_hu=-500.0,
        manufacturer="SIEMENS",
        z_step=1.25,
        nodule_center=(4, 320, 360),
        nodule_radius=12,
        nodule_hu=75.0,
    )
    res_b = client.post("/api/v2/dicom/upload", files={"file": ("study_b.zip", zip_b.getvalue(), "application/zip")})
    assert res_b.status_code == 200
    study_id_b = res_b.json()["study_id"]

    assert study_id_a != study_id_b

    # 3. Request slice metadata for slice 2 of both studies
    meta_a = client.get(f"/api/v2/study/{study_id_a}/slice/2?window=lung").json()
    meta_b = client.get(f"/api/v2/study/{study_id_b}/slice/2?window=lung").json()

    assert meta_a["status"] == "available"
    assert meta_b["status"] == "available"
    assert meta_a["slice_hash"] != meta_b["slice_hash"], "Slice content hashes must differ across different uploads!"

    # 4. Request raw PNG renders
    render_a = client.get(meta_a["image_url"])
    render_b = client.get(meta_b["image_url"])

    assert render_a.status_code == 200
    assert render_b.status_code == 200
    assert render_a.headers["content-type"] == "image/png"
    assert render_b.headers["content-type"] == "image/png"
    assert render_a.content != render_b.content, "Rendered PNG bytes must differ across different uploads!"

    # 5. Run inference pipeline on both studies and verify detected coordinates differ
    from backend.pipeline.job_queue import job_queue
    job_a = job_queue.enqueue_sync(study_id_a, mode="fast_triage")
    job_b = job_queue.enqueue_sync(study_id_b, mode="fast_triage")

    t0 = time.time()
    while time.time() - t0 < 10.0 and (job_a.status not in ("TRIAGE_COMPLETED", "FULL_COMPLETED") or job_b.status not in ("TRIAGE_COMPLETED", "FULL_COMPLETED")):
        time.sleep(0.05)

    assert job_a.triage_report is not None, f"Job A failed or timed out: {job_a.status}"
    assert job_b.triage_report is not None, f"Job B failed or timed out: {job_b.status}"

    findings_a = job_a.triage_report.get("findings", [])
    findings_b = job_b.triage_report.get("findings", [])

    assert len(findings_a) > 0, "Study A must have detected candidate nodules"
    assert len(findings_b) > 0, "Study B must have detected candidate nodules"

    cand_a = findings_a[0]
    cand_b = findings_b[0]

    slice_a = int(round(cand_a["center_zyx"][0]))
    slice_b = int(round(cand_b["center_zyx"][0]))

    # Assert detected slices differ
    assert slice_a != slice_b, f"Detected slice index must differ! A: {slice_a}, B: {slice_b}"
    assert slice_a == 2, f"Study A nodule should be detected at slice 2, got {slice_a}"
    assert slice_b == 4, f"Study B nodule should be detected at slice 4, got {slice_b}"

    # Assert bounding boxes differ
    box_a = cand_a.get("bbox_voxel")
    box_b = cand_b.get("bbox_voxel")
    assert box_a is not None and box_b is not None
    assert box_a != box_b, "Bounding boxes must differ across scans with different nodule locations"


def test_location_to_view_sync(client):
    """
    WORKSTREAM 3 REQUIREMENT:
    Downstream views driven by detected candidate nodule location.
    Centroid coordinate -> slice view -> on-demand 3D Grad-CAM computed on that exact crop.
    """
    # Use seeded demo case
    study_id = "CASE-DEMO-LIDC-0087"
    nodule_id = "NODULE-01"

    # 1. Locate nodule coordinates
    res_loc = client.get(f"/api/v2/xai/{nodule_id}/locate?study_id={study_id}")
    assert res_loc.status_code == 200
    loc_data = res_loc.json()

    assert loc_data["found"] is True
    assert loc_data["source"] == "detector"
    assert "centroid_voxel" in loc_data
    best_slice = loc_data["centroid_voxel"][0]
    assert best_slice == 92
    assert "bbox_voxel" in loc_data
    assert loc_data["diameter_mm"] > 0

    # 2. Slice metadata at that exact centroid slice
    res_slice = client.get(f"/api/v2/study/{study_id}/slice/{best_slice}?window=lung")
    assert res_slice.status_code == 200
    slice_meta = res_slice.json()
    assert slice_meta["slice_index"] == best_slice
    assert slice_meta["status"] == "available"

    # 3. Check HU inspection at centroid
    cy, cx = loc_data["centroid_voxel"][1], loc_data["centroid_voxel"][2]
    res_hu = client.get(f"/api/v2/study/{study_id}/voxel_hu?z={best_slice}&y={cy}&x={cx}")
    assert res_hu.status_code == 200
    assert "hu" in res_hu.json()

    # 4. On-demand Grad-CAM computed on that exact crop
    res_xai = client.get(f"/api/v2/xai/{nodule_id}/gradcam3d?study_id={study_id}&plane=axial")
    assert res_xai.status_code == 200
    xai_data = res_xai.json()
    assert xai_data["generation_status"] == "GENERATED"
    assert xai_data["is_placeholder"] is False
    assert len(xai_data["heatmap_matrix"]) == 64
    assert len(xai_data["heatmap_matrix"][0]) == 64
    assert xai_data["peak_intensity"] > 0.0


def test_job_queue_concurrency_and_modes(client):
    """
    WORKSTREAM 1 REQUIREMENT:
    Bounded asynchronous job queue around pipeline.
    Immediate job ticket issuance with visible queue position and stage reporting.
    """
    study_id = "CASE-DEMO-LIDC-0087"

    # Enqueue scan in Fast Triage mode
    res_job = client.post("/api/v2/jobs", data={"study_id": study_id, "mode": "fast_triage"})
    assert res_job.status_code == 200
    data = res_job.json()

    assert "job_id" in data
    job_id = data["job_id"]
    assert data["mode"] == "fast_triage"
    assert data["status"] in ("QUEUED", "PROCESSING")
    assert "queue_position" in data

    # Poll status until triage completed (triage runs in < 2 seconds)
    max_wait = 10.0
    t0 = time.time()
    completed = False
    status_data = None

    while time.time() - t0 < max_wait:
        res_poll = client.get(f"/api/v2/jobs/{job_id}")
        assert res_poll.status_code == 200
        status_data = res_poll.json()
        if status_data["status"] in ("TRIAGE_COMPLETED", "FULL_COMPLETED"):
            completed = True
            break
        time.sleep(0.1)

    assert completed is True, f"Job {job_id} did not finish within {max_wait}s: {status_data}"
    assert status_data["triage_report"] is not None
    assert "findings" in status_data["triage_report"]
    assert "lung_rads" in status_data["triage_report"]


def test_scan_metering_and_usage(client):
    """
    WORKSTREAM 5 REQUIREMENT:
    Per-scan usage metering: wall time, CPU seconds, peak memory, input voxels,
    candidate crops, compute cost $, honest LLM token accounting, and visible time budget.
    """
    study_id = "CASE-DEMO-LIDC-0087"
    res_usage = client.get(f"/api/v2/study/{study_id}/usage")
    assert res_usage.status_code == 200
    usage = res_usage.json()

    assert "total_wall_time_ms" in usage
    assert usage["total_wall_time_ms"] > 0
    assert "total_cpu_seconds" in usage
    assert "peak_memory_mb" in usage
    assert "input_voxels" in usage
    assert usage["input_voxels"] > 0
    assert "candidate_crops_processed" in usage
    assert "compute_cost_usd" in usage
    assert usage["compute_cost_usd"] >= 0
    assert "cost_basis" in usage
    assert "llm_tokens_used" in usage
    assert "no LLM tokens used" in usage["llm_tokens_used"]
    assert "exceeded_time_budget" in usage
    assert "time_budget_s" in usage


def test_human_in_the_loop_audit_trail(client):
    """
    WORKSTREAM 4 REQUIREMENT:
    Doctor actions per finding: accept, reject, adjust box, add note.
    Audit trail stored in database/records with user, action, and timestamp.
    """
    study_id = "CASE-DEMO-LIDC-0087"
    nodule_id = "NODULE-01"

    review_payload = {
        "action": "accept",
        "user_id": "Dr. Sarah Chen, MD",
        "notes": "Spiculated upper lobe nodule confirmed. Recommend 3-month CT follow-up.",
        "adjusted_diameter_mm": 15.2,
        "study_id": study_id,
    }

    res_review = client.post(f"/api/v2/findings/{nodule_id}/review", json=review_payload)
    assert res_review.status_code == 200
    rev_data = res_review.json()

    assert rev_data["status"] == "success"
    assert rev_data["action"] == "accept"
    assert "review_id" in rev_data
    assert "timestamp" in rev_data

    # Verify review appears in study review audit trail
    res_trail = client.get(f"/api/v2/study/{study_id}/reviews")
    assert res_trail.status_code == 200
    reviews = res_trail.json()["reviews"]
    assert len(reviews) > 0
    latest = reviews[-1]
    assert latest["action"] == "ACCEPT"
    assert latest["user_id"] == "Dr. Sarah Chen, MD"
    assert "Spiculated" in latest["notes"]


def test_longitudinal_comparison(client):
    """
    WORKSTREAM 4 REQUIREMENT:
    Follow-up view comparing same patient across study dates.
    """
    res = client.get("/api/v2/patient/LIDC-IDRI-0087/longitudinal")
    assert res.status_code == 200
    data = res.json()
    assert "history" in data
    assert len(data["history"]) >= 1
    assert "patient_id" in data


def test_cloud_ai_prompt_deidentification_enforcement(client, monkeypatch):
    """
    Strict server-side de-identification test for Cloud AI proxy:
    - Disabled by default (HTTP 403)
    - Requires valid authentication (HTTP 401 when absent)
    - When enabled, strictly rejects patient IDs, MRNs, dates, accession numbers, and image uploads
    """
    import os

    test_key = "test-secret-key-12345"
    monkeypatch.setenv("SECRET_KEY", test_key)
    auth_headers = {"X-API-Key": test_key}

    # 1. Unauthenticated request rejected with 401
    res_unauth = client.post(
        "/api/v2/ai/analyze",
        json={"provider": "openrouter", "model": "mistral-7b", "prompt": "General query"}
    )
    assert res_unauth.status_code == 401

    # 2. Authenticated but disabled by default -> rejected with 403
    monkeypatch.setenv("ENABLE_CLOUD_AI", "false")
    res_disabled = client.post(
        "/api/v2/ai/analyze",
        json={"provider": "openrouter", "model": "mistral-7b", "prompt": "General query"},
        headers=auth_headers
    )
    assert res_disabled.status_code == 403
    assert "Cloud AI integration is disabled" in res_disabled.json()["detail"]

    # 3. Enabled state with PHI / identifiers
    monkeypatch.setenv("ENABLE_CLOUD_AI", "true")

    # Prohibited MRN
    res_mrn = client.post(
        "/api/v2/ai/analyze",
        json={"provider": "openrouter", "model": "mistral-7b", "prompt": "Evaluate patient with MRN-98421 nodule"},
        headers=auth_headers
    )
    assert res_mrn.status_code == 400
    assert "De-identification violation" in res_mrn.json()["detail"]

    # Prohibited patient ID / LIDC tag
    res_pid = client.post(
        "/api/v2/ai/analyze",
        json={"provider": "openrouter", "model": "mistral-7b", "prompt": "Findings for patient_id: LIDC-IDRI-0087"},
        headers=auth_headers
    )
    assert res_pid.status_code == 400
    assert "De-identification violation" in res_pid.json()["detail"]

    # Prohibited study date (YYYY-MM-DD)
    res_date = client.post(
        "/api/v2/ai/analyze",
        json={"provider": "openrouter", "model": "mistral-7b", "prompt": "Prior CT scan was performed on 2024-05-12."},
        headers=auth_headers
    )
    assert res_date.status_code == 400
    assert "calendar date" in res_date.json()["detail"]

    # Prohibited accession number
    res_acc = client.post(
        "/api/v2/ai/analyze",
        json={"provider": "openrouter", "model": "mistral-7b", "prompt": "Accession number ACC99281 details"},
        headers=auth_headers
    )
    assert res_acc.status_code == 400
    assert "prohibited identifier tag" in res_acc.json()["detail"]

    # Prohibited image payload
    res_img = client.post(
        "/api/v2/ai/analyze",
        json={
            "provider": "openrouter",
            "model": "mistral-7b",
            "prompt": "Evaluate nodule morphology",
            "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        },
        headers=auth_headers
    )
    assert res_img.status_code == 400
    assert "Images rejected" in res_img.json()["detail"]


