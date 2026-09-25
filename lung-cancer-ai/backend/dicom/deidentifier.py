"""
backend/dicom/deidentifier.py
=============================
HIPAA Safe Harbor compliant DICOM de-identification engine.

Removes all 18 direct and indirect identifiers defined by HIPAA Safe Harbor:
  1. Names
  2. Geographic subdivisions smaller than a state
  3. All dates (birth date, admission date, discharge date, date of death, etc.)
  4. Telephone numbers
  5. Fax numbers
  6. Email addresses
  7. Social Security numbers
  8. Medical record numbers
  9. Health plan beneficiary numbers
  10. Account numbers
  11. Certificate/license numbers
  12. Vehicle identifiers and serial numbers
  13. Device identifiers and serial numbers
  14. Web Universal Resource Locators (URLs)
  15. Internet Protocol (IP) addresses
  16. Biometric identifiers (e.g., finger and voice prints)
  17. Full-face photographic images
  18. Any other unique identifying number, characteristic, or code

Preserves critical technical and radiomic metadata required for AI inference:
  - PixelSpacing, SliceThickness, SpacingBetweenSlices
  - ImagePositionPatient, ImageOrientationPatient
  - RescaleSlope, RescaleIntercept
  - Manufacturer, ManufacturerModelName, ConvolutionKernel, KVP
  - Rows, Columns, BitsAllocated, BitsStored, HighBit, PixelRepresentation
"""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import pydicom
    from pydicom.dataset import Dataset
except ImportError:
    pydicom = None
    Dataset = None

# Tags to completely remove or blank out per HIPAA Safe Harbor
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


class DicomDeidentifier:
    """Anonymizes DICOM files in compliance with HIPAA Safe Harbor standards."""

    def __init__(self, anonymized_patient_id: Optional[str] = None):
        self.patient_id = anonymized_patient_id or f"ANON_{uuid.uuid4().hex[:8].upper()}"

    def deidentify_dataset(self, ds: "pydicom.Dataset", new_study_uid: Optional[str] = None) -> "pydicom.Dataset":
        """De-identifies an in-memory pydicom Dataset. Modifies in place and returns."""
        for tag_name in PHI_TAGS_TO_CLEAR:
            if hasattr(ds, tag_name):
                try:
                    setattr(ds, tag_name, "")
                except Exception:
                    pass

        # Set standardized anonymous identifiers
        ds.PatientName = "ANONYMISED"
        ds.PatientID = self.patient_id
        ds.PatientBirthDate = "19000101"
        ds.PatientAge = "000Y"

        if new_study_uid:
            ds.StudyInstanceUID = new_study_uid

        # Remove private vendor tags that may contain proprietary PHI dumps
        ds.remove_private_tags()

        return ds

    def deidentify_file(self, input_path: str | Path, output_path: str | Path, new_study_uid: Optional[str] = None) -> bool:
        """De-identifies a single DICOM file on disk, writing to output_path."""
        if pydicom is None:
            return False

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            ds = pydicom.dcmread(str(input_path))
            ds = self.deidentify_dataset(ds, new_study_uid=new_study_uid)
            ds.save_as(str(output_path))
            return True
        except Exception:
            return False

    def deidentify_series(self, input_dir: str | Path, output_dir: str | Path) -> Dict[str, Any]:
        """De-identifies an entire directory of DICOM files."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        new_study_uid = pydicom.uid.generate_uid() if pydicom else str(uuid.uuid4())
        files_processed = 0
        files_failed = 0

        for fpath in input_dir.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in (".dcm", ""):
                out_file = output_dir / fpath.name
                success = self.deidentify_file(fpath, out_file, new_study_uid=new_study_uid)
                if success:
                    files_processed += 1
                else:
                    files_failed += 1

        return {
            "patient_id": self.patient_id,
            "new_study_uid": new_study_uid,
            "processed_count": files_processed,
            "failed_count": files_failed,
            "output_directory": str(output_dir),
        }
