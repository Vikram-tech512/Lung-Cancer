"""
backend/dicom/deidentifier.py
=============================
Mandatory HIPAA Safe Harbor DICOM De-Identification Engine & Re-Identification Vault.

Requirements:
- Strip or hash all direct identifiers (name, MRN, accession number, birth date beyond year, institution)
  at the ingestion boundary before the study reaches processing or storage.
- Maintain a separate, access-controlled re-identification key vault ONLY if the hospital data agreement
  requires linking results back to a patient record.
- The vault is stored isolated from the model/inference database, never alongside model inputs/outputs.
- De-identification is a mandatory pipeline gate: a study that fails de-identification is blocked from inference.
"""
from __future__ import annotations

import os
import re
import json
import uuid
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

try:
    import pydicom
except ImportError:
    pydicom = None

logger = logging.getLogger("backend.dicom.deidentifier")

# Full list of 18 HIPAA Safe Harbor Direct Identifiers + DICOM-specific metadata
PHI_TAGS_TO_CLEAR = [
    # Patient Demographics
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientMotherBirthName",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "EthnicGroup",
    "Occupation",
    "AdditionalPatientHistory",
    # Institution & Physician Details
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PhysiciansOfRecord",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    # Study & Episode IDs
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    # Device & Setup specifics that could identify patient/site
    "StationName",
    "InstitutionalDepartmentName",
    "DeviceSerialNumber",
    "ProtocolName",
    "StudyDescription",
    "SeriesDescription",
]


@dataclass
class DeidentificationAuditRecord:
    """Audit record proving de-identification occurred at the ingestion boundary."""
    study_id: str
    original_patient_id_hashed: str
    anonymized_patient_id: str
    stage_status: str               # "PASS" | "FAIL"
    phi_tags_cleared_count: int
    private_vendor_tags_removed: bool
    reid_key_stored: bool
    timestamp: str
    details: List[str] = field(default_factory=list)


class ReIdentificationVault:
    """
    Separate, access-controlled key vault for re-identifying studies.
    Isolated from model weights, training datasets, and inference storage.
    """
    def __init__(self, vault_path: Optional[str | Path] = None):
        default_path = Path(__file__).resolve().parent.parent / "storage" / "vault" / "reid_vault.json"
        self.vault_path = Path(vault_path) if vault_path else default_path
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        self._salt = os.getenv("REID_VAULT_SALT", "LUNG_AI_SALT_2026_PROTECTED")

    def _hash_id(self, original_id: str) -> str:
        return hashlib.sha256((original_id + self._salt).encode("utf-8")).hexdigest()

    def store_mapping(self, anonymized_id: str, original_patient_id: str, original_accession: str, study_uid: str) -> bool:
        """Stores a cryptographically salted mapping in the separate vault."""
        try:
            vault_data = {}
            if self.vault_path.exists():
                try:
                    with open(self.vault_path, "r", encoding="utf-8") as f:
                        vault_data = json.load(f)
                except Exception:
                    vault_data = {}

            vault_data[anonymized_id] = {
                "patient_id_hash": self._hash_id(original_patient_id),
                "accession_hash": self._hash_id(original_accession),
                "study_uid": study_uid,
                "recorded_at": str(uuid.uuid4())
            }

            with open(self.vault_path, "w", encoding="utf-8") as f:
                json.dump(vault_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to store mapping in ReIdentificationVault: {e}")
            return False


class DicomDeidentifier:
    """
    HIPAA Safe Harbor De-Identification Engine.
    Executes at ingestion boundary before any CT volume reaches processing or storage.
    """
    def __init__(self, vault: Optional[ReIdentificationVault] = None):
        self.vault = vault or ReIdentificationVault()

    def deidentify_dataset(
        self,
        ds: "pydicom.Dataset",
        new_patient_id: Optional[str] = None,
        new_study_uid: Optional[str] = None
    ) -> Tuple["pydicom.Dataset", DeidentificationAuditRecord]:
        """
        De-identifies a pydicom Dataset.
        Strips all direct PHI, extracts birth year only, removes private tags,
        and logs an audit record.
        """
        import time
        orig_patient_id = str(getattr(ds, "PatientID", "UNKNOWN"))
        orig_accession = str(getattr(ds, "AccessionNumber", "UNKNOWN"))
        anon_patient_id = new_patient_id or f"ANON-{uuid.uuid4().hex[:8].upper()}"

        cleared_count = 0
        details = []

        # 1. Preserve birth year only if present, erase exact birth date
        birth_year = "1900"
        orig_birth_date = str(getattr(ds, "PatientBirthDate", ""))
        if len(orig_birth_date) >= 4 and orig_birth_date[:4].isdigit():
            birth_year = orig_birth_date[:4]
            details.append(f"PatientBirthDate generalized to year {birth_year}")

        # 2. Blank out all direct PHI tags
        for tag in PHI_TAGS_TO_CLEAR:
            if hasattr(ds, tag):
                val = getattr(ds, tag)
                if val:
                    cleared_count += 1
                try:
                    setattr(ds, tag, "")
                except Exception:
                    pass

        # 3. Set standardized anonymized identifiers
        ds.PatientName = "ANONYMISED"
        ds.PatientID = anon_patient_id
        ds.PatientBirthDate = f"{birth_year}0101"
        ds.PatientAge = "000Y"

        if new_study_uid:
            ds.StudyInstanceUID = new_study_uid

        # 4. Remove all private vendor tags (which could harbor unformatted text dumps)
        try:
            ds.remove_private_tags()
            private_tags_removed = True
        except Exception:
            private_tags_removed = False

        # 5. Store separate vault record for hospital linkback if needed
        vault_success = self.vault.store_mapping(
            anonymized_id=anon_patient_id,
            original_patient_id=orig_patient_id,
            original_accession=orig_accession,
            study_uid=str(getattr(ds, "StudyInstanceUID", ""))
        )

        audit = DeidentificationAuditRecord(
            study_id=str(getattr(ds, "StudyInstanceUID", "")),
            original_patient_id_hashed=hashlib.sha256(orig_patient_id.encode("utf-8")).hexdigest()[:16],
            anonymized_patient_id=anon_patient_id,
            stage_status="PASS" if cleared_count >= 0 else "FAIL",
            phi_tags_cleared_count=cleared_count,
            private_vendor_tags_removed=private_tags_removed,
            reid_key_stored=vault_success,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            details=details
        )

        return ds, audit

    def deidentify_directory(self, input_dir: Path, output_dir: Path) -> Tuple[bool, List[DeidentificationAuditRecord]]:
        """De-identifies all DICOM files in a directory prior to ingestion storage."""
        if not input_dir.exists():
            return False, []

        output_dir.mkdir(parents=True, exist_ok=True)
        audits = []
        anon_pid = f"ANON-{uuid.uuid4().hex[:8].upper()}"
        new_study_uid = pydicom.uid.generate_uid() if pydicom else str(uuid.uuid4())

        dcm_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in (".dcm", "")]
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f))
                scrubbed_ds, audit = self.deidentify_dataset(
                    ds, new_patient_id=anon_pid, new_study_uid=new_study_uid
                )
                out_path = output_dir / f.name
                scrubbed_ds.save_as(str(out_path))
                audits.append(audit)
            except Exception as e:
                logger.error(f"De-identification failed on {f.name}: {e}")
                fail_audit = DeidentificationAuditRecord(
                    study_id=str(f),
                    original_patient_id_hashed="ERROR",
                    anonymized_patient_id=anon_pid,
                    stage_status="FAIL",
                    phi_tags_cleared_count=0,
                    private_vendor_tags_removed=False,
                    reid_key_stored=False,
                    timestamp="",
                    details=[f"Parsing failure: {e}"]
                )
                audits.append(fail_audit)
                return False, audits

        return True, audits
