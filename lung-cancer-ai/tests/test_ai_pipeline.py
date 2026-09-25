"""
tests/test_ai_pipeline.py
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from ml.segmentation.lung_segmenter import LungSegmenter3D
from ml.segmentation.nodule_segmenter import NoduleSegmenter3D
from ml.morphology.analyzer import MorphologyAnalyzer
from ml.risk.malignancy_model import MalignancyRiskModel, ClinicalVariables
from ml.longitudinal.comparator import LongitudinalComparator, NoduleRecord
from ml.ood.detector import OODDetector
from ml.fp_reduction.analyzer import FalsePositiveAnalyzer


def test_lung_segmenter():
    segmenter = LungSegmenter3D()
    vol = np.full((32, 64, 64), -1000.0, dtype=np.float32)
    # Put lung parenchyma
    vol[:, 10:50, 10:50] = -700.0
    res = segmenter.segment(vol, voxel_spacing_mm=(1.0, 1.0, 1.0))
    assert res.combined_mask.shape == (32, 64, 64)
    assert res.total_volume_cm3 >= 0.0


def test_nodule_segmenter_and_morphology():
    crop = np.full((32, 32, 32), -800.0, dtype=np.float32)
    # Place a small spherical nodule at center
    z, y, x = np.ogrid[:32, :32, :32]
    dist = np.sqrt((z - 16)**2 + (y - 16)**2 + (x - 16)**2)
    crop[dist <= 5.0] = 40.0

    seg = NoduleSegmenter3D()
    res = seg.segment_nodule_roi(crop, voxel_spacing_mm=(1.0, 1.0, 1.0))
    assert res.measurements.max_diameter_mm > 0.0
    assert res.measurements.volume_mm3 > 0.0

    analyzer = MorphologyAnalyzer()
    morph = analyzer.analyze(crop, res.roi_mask, res.measurements)
    assert morph.attenuation_type in ("solid", "part_solid", "ground_glass", "calcified")
    assert morph.margin_type in ("smooth", "lobulated", "spiculated", "ill_defined")


def test_risk_model_clinical_fusion():
    risk_model = MalignancyRiskModel()
    crop = np.zeros((16, 16, 16), dtype=np.float32)
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[6:10, 6:10, 6:10] = 1

    seg = NoduleSegmenter3D()
    meas = seg._calculate_measurements(crop, mask, (1.0, 1.0, 1.0))
    analyzer = MorphologyAnalyzer()
    morph = analyzer.analyze(crop, mask, meas)

    clin = ClinicalVariables(age=65, smoking_status="current", pack_years=40.0)
    risk = risk_model.estimate_risk(meas, morph, deep_features_score=0.8, clinical=clin)
    assert 0.0 <= risk.calibrated_malignancy_risk <= 1.0
    assert risk.lung_rads_category in ("1", "2", "3", "4A", "4B", "4X")


def test_longitudinal_growth_and_vdt():
    comparator = LongitudinalComparator()
    t1 = datetime(2025, 1, 1)
    t2 = datetime(2025, 7, 1)  # ~181 days

    curr_nods = [
        NoduleRecord(nodule_id="N1", center_zyx_mm=(50.0, 100.0, 100.0), diameter_mm=12.0, volume_mm3=904.0, attenuation_type="solid", slice_index=50)
    ]
    prior_nods = [
        NoduleRecord(nodule_id="N1_PRIOR", center_zyx_mm=(51.0, 101.0, 99.0), diameter_mm=10.0, volume_mm3=523.0, attenuation_type="solid", slice_index=51)
    ]

    res = comparator.compare_studies(
        current_study_id="STUDY_CURR",
        current_nodules=curr_nods,
        prior_study_id="STUDY_PRIOR",
        prior_nodules=prior_nods,
        current_date=t2,
        prior_date=t1
    )
    assert len(res.matched_nodules) == 1
    match = res.matched_nodules[0]
    assert match.volume_doubling_time_days is not None
    assert match.volume_change_pct > 0.0
