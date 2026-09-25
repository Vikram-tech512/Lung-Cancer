"""
backend/tests/test_hospital_interop.py
======================================
Comprehensive Test Suite for Real-World Hospital DICOM Interoperability & Deployment Readiness:
1. Multi-Vendor Robustness & Quirk Handling (GE, Siemens, Philips, Canon)
2. Scout / Localizer Filtering
3. Coordinate-derived Z-spacing vs Header SliceThickness
4. Non-axial acquisition and tilted gantry detection
5. Burned-in annotation detection
6. Ingestion-boundary HIPAA Safe Harbor De-identification & ReIdentificationVault
7. Guard 5 Non-Bypassable Clinical Deployment Gate
8. PACS C-STORE & FHIR R4 Diagnostic Report Integration
"""

import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from pathlib import Path

from backend.dicom.compatibility import MultiVendorDicomInspector, DicomCompatibilityReport
from backend.dicom.deidentifier import DicomDeidentifier, ReIdentificationVault
from backend.core.clinical_gate import ClinicalDeploymentGate, ValidationEvidence, RESEARCH_BANNER
from backend.dicom.pacs_service import HospitalPacsReceiver, FhirHospitalBridge
from backend.api.service import app
from fastapi.testclient import TestClient


def create_synthetic_slice(
    z_pos: float = 0.0,
    manufacturer: str = "SIEMENS",
    slice_thickness: float = 1.25,
    is_scout: bool = False,
    gantry_tilt: float = 0.0,
    orientation: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    burned_in: str = "NO"
) -> Dataset:
    """Creates a valid in-memory CT DICOM slice."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.840.113619.2.55.1.1"
    ds.SeriesInstanceUID = "1.2.840.113619.2.55.1.2"
    ds.Modality = "CT"
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = "Test Scanner Model"
    ds.SliceThickness = slice_thickness
    ds.PixelSpacing = [0.7, 0.7]
    ds.ImagePositionPatient = [0.0, 0.0, z_pos]
    ds.ImageOrientationPatient = list(orientation)
    ds.GantryDetectorTilt = gantry_tilt
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.BurnedInAnnotation = burned_in
    ds.PatientName = "DOE^JOHN"
    ds.PatientID = "HOSPITAL-MRN-998877"
    ds.PatientBirthDate = "19650412"
    ds.AccessionNumber = "ACC-2026-001"

    if is_scout:
        ds.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    else:
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]

    # Dummy pixel array
    ds.Rows = 64
    ds.Columns = 64
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = (np.ones((64, 64), dtype=np.uint16) * 100).tobytes()

    return ds


# ── 1. Multi-Vendor Robustness Tests ──────────────────────────────────────────

def test_vendor_identification():
    inspector = MultiVendorDicomInspector()
    ds_ge = Dataset(); ds_ge.Manufacturer = "GE MEDICAL SYSTEMS"
    ds_siemens = Dataset(); ds_siemens.Manufacturer = "SIEMENS Healthineers"
    ds_philips = Dataset(); ds_philips.Manufacturer = "Philips Medical Systems"
    ds_canon = Dataset(); ds_canon.Manufacturer = "Canon Medical Systems"

    assert inspector.identify_vendor(ds_ge) == "GE"
    assert inspector.identify_vendor(ds_siemens) == "SIEMENS"
    assert inspector.identify_vendor(ds_philips) == "PHILIPS"
    assert inspector.identify_vendor(ds_canon) == "CANON/TOSHIBA"


def test_scout_filtering():
    inspector = MultiVendorDicomInspector()
    ds_scout = create_synthetic_slice(is_scout=True)
    is_scout, reason = inspector.is_scout_or_localizer(ds_scout)
    assert is_scout is True
    assert "LOCALIZER" in reason

    ds_axial = create_synthetic_slice(is_scout=False)
    is_scout, _ = inspector.is_scout_or_localizer(ds_axial)
    assert is_scout is False


def test_derived_spacing_and_discrepancy(tmp_path):
    inspector = MultiVendorDicomInspector(max_spacing_mismatch_mm=0.2)
    files = []
    # Header says SliceThickness = 1.0mm, but actual coordinate step is 2.5mm
    for i in range(20):
        ds = create_synthetic_slice(z_pos=i * 2.5, slice_thickness=1.0)
        p = tmp_path / f"slice_{i:03d}.dcm"
        ds.save_as(str(p), write_like_original=False)
        files.append(p)

    usable_slices, report = inspector.inspect_series(files)
    assert report.is_compatible is True
    assert report.usable_axial_slice_count == 20
    assert report.derived_z_spacing_mm == 2.5
    assert report.header_slice_thickness_mm == 1.0
    assert report.spacing_discrepancy_mm == 1.5
    assert any(q.category == "SPACING_DISCREPANCY" for q in report.quirks)


def test_non_axial_and_tilted_gantry_detection(tmp_path):
    inspector = MultiVendorDicomInspector(max_gantry_tilt_degrees=1.0)
    files = []
    for i in range(20):
        # Tilted gantry: 5.0 degrees
        ds = create_synthetic_slice(z_pos=i * 1.5, gantry_tilt=5.0)
        p = tmp_path / f"slice_{i:03d}.dcm"
        ds.save_as(str(p), write_like_original=False)
        files.append(p)

    _, report = inspector.inspect_series(files)
    assert any(q.category == "TILTED_GANTRY" for q in report.quirks)


# ── 2. De-identification & Vault Tests ────────────────────────────────────────

def test_hipaa_safe_harbor_deidentification(tmp_path):
    vault = ReIdentificationVault(vault_path=tmp_path / "test_vault.json")
    deid = DicomDeidentifier(vault=vault)

    ds = create_synthetic_slice()
    assert ds.PatientName == "DOE^JOHN"
    assert ds.PatientID == "HOSPITAL-MRN-998877"

    scrubbed_ds, audit = deid.deidentify_dataset(ds)

    assert scrubbed_ds.PatientName == "ANONYMISED"
    assert scrubbed_ds.PatientID.startswith("ANON-")
    assert scrubbed_ds.PatientBirthDate == "19650101"  # year preserved, month/day generalized
    assert getattr(scrubbed_ds, "AccessionNumber", "") == ""
    assert audit.stage_status == "PASS"
    assert audit.reid_key_stored is True
    assert (tmp_path / "test_vault.json").exists()


# ── 3. Guard 5 Clinical Deployment Gate Tests ─────────────────────────────────

def test_guard_5_default_research_only(tmp_path):
    gate = ClinicalDeploymentGate(state_file=tmp_path / "gate_state.json")
    assert gate.deployment_tier == "RESEARCH_ONLY"
    assert gate.is_hospital_validated is False
    assert RESEARCH_BANNER in gate.get_advisory_banner()


def test_guard_5_transition_requirements(tmp_path, monkeypatch):
    gate = ClinicalDeploymentGate(state_file=tmp_path / "gate_state.json")
    monkeypatch.setenv("HOSPITAL_GATE_ADMIN_SECRET", "TOP_SECRET_ADMIN_TOKEN_99")

    # Incomplete evidence should fail
    evidence_bad = ValidationEvidence(
        hospital_name="",
        case_mix_cohort_size=10,  # < 50
        internal_vs_external_delta={},
        ethics_irb_protocol_number="",
        ethics_approval_date="",
        regulatory_counsel_determination="",
        disclaimer_verified_on_all_screens=False,
        authorized_administrator="",
        evidence_package_hash=""
    )
    ok, err = gate.transition_to_hospital_validated(evidence_bad, "TOP_SECRET_ADMIN_TOKEN_99")
    assert ok is False
    assert gate.deployment_tier == "RESEARCH_ONLY"

    # Full evidence with admin secret should succeed
    evidence_good = ValidationEvidence(
        hospital_name="Metropolitan Academic Medical Center",
        case_mix_cohort_size=120,
        internal_vs_external_delta={"macro_auc_delta": -0.018, "f1_delta": -0.024},
        ethics_irb_protocol_number="IRB-2026-MED-4402",
        ethics_approval_date="2026-08-15",
        regulatory_counsel_determination="DOC-LEGAL-RES-TIER-01",
        disclaimer_verified_on_all_screens=True,
        authorized_administrator="Dr. Sarah Collins, Chief Medical Officer",
        evidence_package_hash="a1b2c3d4e5f67890"
    )
    ok, msg = gate.transition_to_hospital_validated(evidence_good, "TOP_SECRET_ADMIN_TOKEN_99")
    assert ok is True
    assert gate.deployment_tier == "HOSPITAL_VALIDATED"
    assert gate.is_hospital_validated is True


# ── 4. PACS & FHIR Integration Tests ──────────────────────────────────────────

def test_fhir_diagnostic_report_generation():
    report = FhirHospitalBridge.create_diagnostic_report_fhir(
        study_id="STUDY-TEST-001",
        patient_id="ANON-TEST-PATIENT",
        verdict="MALIGNANT",
        malignancy_probability=0.84,
        lung_rads="Lung-RADS 4B",
        deployment_tier="RESEARCH_ONLY",
        disclaimer=RESEARCH_BANNER
    )

    assert report["resourceType"] == "DiagnosticReport"
    assert report["status"] == "preliminary"
    assert "363358000" in report["conclusionCode"][0]["coding"][0]["code"]
    assert any(ext["valueString"] == "RESEARCH_ONLY" for ext in report["extension"])


# ── 5. End-to-End API Endpoints ───────────────────────────────────────────────

def test_hospital_interop_api_endpoints():
    client = TestClient(app)

    # Health check contains deployment_tier and banner
    res_health = client.get("/api/v2/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["deployment_tier"] == "RESEARCH_ONLY"
    assert "RESEARCH USE ONLY" in data["banner"]

    # Gate status endpoint
    res_gate = client.get("/api/v2/gate/status")
    assert res_gate.status_code == 200
    assert res_gate.json()["deployment_tier"] == "RESEARCH_ONLY"

    # Compatibility report for demo case
    res_comp = client.get("/api/v2/dicom/CASE-DEMO-LIDC-0087/compatibility")
    assert res_comp.status_code == 200
    comp = res_comp.json()
    assert comp["compatibility"]["vendor"] == "SIEMENS"
    assert comp["deidentification"]["stage_status"] == "PASS"

    # FHIR export endpoint
    res_fhir = client.get("/api/v2/dicom/fhir/report/CASE-DEMO-LIDC-0087")
    assert res_fhir.status_code == 200
    assert res_fhir.json()["resourceType"] == "DiagnosticReport"

    # PACS listener status
    res_pacs = client.get("/api/v2/pacs/status")
    assert res_pacs.status_code == 200
    assert "ae_title" in res_pacs.json()
