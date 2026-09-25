"""
tests/test_dicom_validator.py
"""
import pytest
import numpy as np
from pathlib import Path
from backend.dicom.validator import validate_dicom_series, QualityResult
from backend.dicom.deidentifier import DicomDeidentifier
from backend.preprocessing.ct_pipeline import CTPipeline

def test_ct_pipeline_windowing():
    pipeline = CTPipeline(default_window="lung")
    raw = np.array([-1200.0, -1000.0, -300.0, 400.0, 1000.0], dtype=np.float32)
    norm = pipeline.apply_window(raw)
    assert norm[0] == 0.0  # clipped below -1000
    assert norm[1] == 0.0  # at -1000
    assert norm[3] == 1.0  # at +400
    assert norm[4] == 1.0  # clipped above +400
    assert 0.0 <= norm[2] <= 1.0

def test_ct_crop_extraction():
    pipeline = CTPipeline()
    vol = np.zeros((64, 64, 64), dtype=np.float32)
    vol[30:34, 30:34, 30:34] = 100.0
    crop = pipeline.extract_nodule_crop(vol, center_zyx=(32, 32, 32), crop_size=(16, 16, 16))
    assert crop.shape == (16, 16, 16)
    assert crop.max() == 100.0

def test_deidentifier_instantiation():
    deid = DicomDeidentifier(anonymized_patient_id="ANON_TEST_123")
    assert deid.patient_id == "ANON_TEST_123"
