"""
backend/tests/test_pipeline.py
================================
Smoke tests for the full ML Pipeline layer.

Tests verify:
  - Data augmentation produces correct output shapes
  - CTSplitter enforces patient-level stratification (no leakage)
  - DataLoaderFactory builds a loader with correct batch dimensions
  - ModelEvaluator computes non-trivial classification metrics
  - CalibrationEvaluator computes ECE / Brier score
  - CurveGenerator produces ROC curve with AUC ∈ (0, 1)
  - StatisticalTests McNemar + DeLong produce valid p-values
  - RadiomicsPipeline returns a non-empty feature dict
  - FeatureSelector reduces feature dimensionality
  - PipelineOrchestrator.from_config() instantiates without error
  - MonitoringMiddleware + HealthChecker are importable
"""

from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# 1. Augmentation
# ===========================================================================
class TestAugmentationPipeline:
    def test_train_output_shape(self):
        from backend.pipeline.data.augmentation import AugmentationPipeline
        aug = AugmentationPipeline(mode="train", output_size=(64, 64, 64), seed=42)
        vol  = np.random.randn(80, 80, 80).astype(np.float32)
        mask = (vol > 0).astype(np.float32)
        out_vol, out_mask = aug(vol, mask)
        assert out_vol.shape  == (64, 64, 64), f"Expected (64,64,64), got {out_vol.shape}"
        assert out_mask.shape == (64, 64, 64)

    def test_val_no_augment(self):
        from backend.pipeline.data.augmentation import AugmentationPipeline
        aug = AugmentationPipeline(mode="val", output_size=(48, 48, 48))
        vol = np.zeros((60, 60, 60), dtype=np.float32)
        out_vol, _ = aug(vol)
        assert out_vol.shape == (48, 48, 48)

    def test_small_volume_pads(self):
        from backend.pipeline.data.augmentation import AugmentationPipeline
        aug = AugmentationPipeline(mode="val", output_size=(64, 64, 64))
        vol = np.ones((32, 32, 32), dtype=np.float32)
        out, _ = aug(vol)
        assert out.shape == (64, 64, 64)


# ===========================================================================
# 2. CTSplitter
# ===========================================================================
class TestCTSplitter:
    def _make_records(self, n=100):
        rng = np.random.default_rng(0)
        return [
            {
                "patient_id": f"P{i // 3:04d}",   # 3 scans per patient → ~33 patients
                "label":      int(rng.integers(0, 2)),
                "scan_path":  f"/data/scan_{i}.nii.gz",
            }
            for i in range(n)
        ]

    def test_no_patient_leakage(self):
        from backend.pipeline.data.splitter import CTSplitter
        records = self._make_records(120)
        splitter = CTSplitter(val_frac=0.15, test_frac=0.15, seed=42)
        train, val, test = splitter.split(records)

        train_pids = {r["patient_id"] for r in train}
        val_pids   = {r["patient_id"] for r in val}
        test_pids  = {r["patient_id"] for r in test}

        assert not (train_pids & val_pids),  "Train/Val patient overlap!"
        assert not (train_pids & test_pids), "Train/Test patient overlap!"
        assert not (val_pids & test_pids),   "Val/Test patient overlap!"

    def test_total_records_preserved(self):
        from backend.pipeline.data.splitter import CTSplitter
        records = self._make_records(90)
        splitter = CTSplitter(seed=0)
        train, val, test = splitter.split(records)
        assert len(train) + len(val) + len(test) == len(records)

    def test_kfold(self):
        from backend.pipeline.data.splitter import CTSplitter
        records = self._make_records(90)
        splitter = CTSplitter(seed=0)
        folds = splitter.kfold_split(records, n_folds=5)
        assert len(folds) == 5
        for train_f, val_f in folds:
            assert len(train_f) > 0
            assert len(val_f) > 0

    def test_subset_split_and_split_hash(self):
        from backend.pipeline.data.splitter import CTSplitter
        records = [
            {"patient_id": f"P{i:03d}", "series_uid": f"S{i:03d}", "subset_id": i % 4, "label": i % 2}
            for i in range(40)
        ]
        splitter = CTSplitter(split_mode="subset")
        train, val, test = splitter.split(records)
        assert all(r["subset_id"] in {0, 1} for r in train)
        assert all(r["subset_id"] == 2 for r in val)
        assert all(r["subset_id"] == 3 for r in test)
        assert splitter.last_split_hash is not None
        assert len(splitter.last_split_hash) == 64  # SHA-256 hex string

    def test_patient_leakage_assertion_raises(self):
        import pytest
        from backend.pipeline.data.splitter import CTSplitter
        # Leakage: patient P001 in both train and val
        train = [{"patient_id": "P001", "series_uid": "S001"}]
        val = [{"patient_id": "P001", "series_uid": "S002"}]
        test = [{"patient_id": "P003", "series_uid": "S003"}]
        with pytest.raises(AssertionError, match="leaked across splits"):
            CTSplitter.assert_splits_disjoint(train, val, test)

        # Series UID leakage: series S009 in train and test
        train_s = [{"patient_id": "P010", "series_uid": "S009"}]
        val_s = [{"patient_id": "P011", "series_uid": "S011"}]
        test_s = [{"patient_id": "P012", "series_uid": "S009"}]
        with pytest.raises(AssertionError, match="leaked across splits"):
            CTSplitter.assert_splits_disjoint(train_s, val_s, test_s)


# ===========================================================================
# 3. Evaluator
# ===========================================================================
class TestModelEvaluator:
    def _dummy_probs(self, n=200, seed=7):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, n)
        y_prob  = np.clip(y_true * 0.6 + rng.normal(0, 0.2, n), 0.01, 0.99)
        y_pred  = (y_prob > 0.5).astype(int)
        return y_true.tolist(), y_prob.tolist(), y_pred.tolist()

    def test_classification_metrics_range(self):
        from backend.pipeline.evaluate.evaluator import ModelEvaluator
        y_true, y_prob, y_pred = self._dummy_probs()
        evaluator = ModelEvaluator(task="classification")
        report = evaluator._classification_report(y_prob, y_pred, y_true)
        assert 0.5 <= report["auc"] <= 1.0
        assert 0.0 <= report["sensitivity"] <= 1.0
        assert 0.0 <= report["specificity"] <= 1.0
        assert "confusion_matrix" in report

    def test_mcc_range(self):
        from backend.pipeline.evaluate.evaluator import ModelEvaluator
        y_true, y_prob, y_pred = self._dummy_probs()
        evaluator = ModelEvaluator()
        report = evaluator._classification_report(y_prob, y_pred, y_true)
        assert -1.0 <= report["mcc"] <= 1.0


# ===========================================================================
# 4. Calibration
# ===========================================================================
class TestCalibrationEvaluator:
    def test_temperature_scaling_rejects_non_validation_split(self):
        import pytest
        import torch
        from backend.calibration.temperature_scaling import TemperatureScaler

        scaler = TemperatureScaler()
        with pytest.raises(ValueError, match="only be fit on validation"):
            scaler.fit(torch.zeros((4, 2)), torch.tensor([0, 1, 0, 1]), split_name="test")

    def test_perfect_calibration_low_ece(self):
        from backend.pipeline.evaluate.calibration_eval import CalibrationEvaluator
        # Perfect model: prob = actual frequency
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 500)
        y_prob = y_true * 0.85 + (1 - y_true) * 0.15
        cal = CalibrationEvaluator(n_bins=10)
        report = cal.evaluate(y_true.tolist(), y_prob.tolist())
        assert report["ece"] <= 0.20
        assert 0 <= report["brier_score"] <= 1

    def test_random_calibration(self):
        from backend.pipeline.evaluate.calibration_eval import CalibrationEvaluator
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 2, 300)
        y_prob = rng.uniform(0, 1, 300)
        cal = CalibrationEvaluator()
        report = cal.evaluate(y_true.tolist(), y_prob.tolist())
        assert "reliability_diagram" in report
        assert isinstance(report["reliability_diagram"], list)


# ===========================================================================
# 5. Curve Generator
# ===========================================================================
class TestCurveGenerator:
    def test_roc_curve_auc_range(self):
        from backend.pipeline.evaluate.curves import CurveGenerator
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 300)
        y_prob = np.clip(y_true * 0.7 + rng.normal(0, 0.2, 300), 0.01, 0.99)
        patient_ids = np.repeat(np.arange(60), 5)
        gen = CurveGenerator(n_bootstrap=100, seed=11)
        roc = gen.roc_curve(
            y_true.tolist(), y_prob.tolist(), patient_ids=patient_ids.tolist()
        )
        assert 0.5 <= roc["auc"] <= 1.0
        assert roc["auc_ci_lower"] <= roc["auc"] <= roc["auc_ci_upper"]

    def test_roc_requires_patient_ids_for_bootstrap_ci(self):
        import pytest
        from backend.pipeline.evaluate.curves import CurveGenerator

        gen = CurveGenerator(n_bootstrap=20)
        with pytest.raises(ValueError, match="patient_ids are required"):
            gen.roc_curve([0, 1], [0.2, 0.8])

    def test_patient_cluster_bootstrap_is_seeded(self):
        from backend.pipeline.evaluate.curves import CurveGenerator

        y_true = np.repeat([0, 1, 0, 1], 3)
        y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.7, 0.2, 0.4, 0.3, 0.6, 0.9, 0.8])
        patient_ids = np.repeat(["P1", "P2", "P3", "P4"], 3)
        first = CurveGenerator(n_bootstrap=100, seed=19)._bootstrap_auc(
            y_true, y_prob, patient_ids
        )
        second = CurveGenerator(n_bootstrap=100, seed=19)._bootstrap_auc(
            y_true, y_prob, patient_ids
        )
        assert first == second

    def test_pr_curve_fields(self):
        from backend.pipeline.evaluate.curves import CurveGenerator
        rng = np.random.default_rng(6)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.clip(y_true * 0.65 + rng.normal(0, 0.2, 200), 0.01, 0.99)
        gen = CurveGenerator(n_bootstrap=50)
        pr = gen.pr_curve(y_true.tolist(), y_prob.tolist())
        assert "average_precision" in pr
        assert "curve" in pr

    def test_single_class_evaluation_marks_auc_unavailable(self):
        from backend.pipeline.evaluate.evaluator import ModelEvaluator

        report = ModelEvaluator()._classification_report([0.1, 0.3], [0, 0], [0, 0])
        assert report["auc"] is None
        assert report["sensitivity"] is None
        assert report["specificity"] == 1.0

    def test_empty_evaluation_does_not_substitute_metric_values(self):
        from backend.pipeline.evaluate.evaluator import ModelEvaluator

        report = ModelEvaluator()._classification_report([], [], [])
        assert report["n_samples"] == 0
        assert report["auc"] is None
        assert report["accuracy"] is None
        assert report["sensitivity"] is None


# ===========================================================================
# 6. Statistical Tests
# ===========================================================================
class TestStatisticalTests:
    def _dummy_preds(self, n=300, seed=10):
        rng   = np.random.default_rng(seed)
        y     = rng.integers(0, 2, n)
        pA    = (rng.uniform(0, 1, n) < (0.75 * y + 0.25)).astype(int)
        pB    = (rng.uniform(0, 1, n) < (0.60 * y + 0.30)).astype(int)
        probA = np.clip(y * 0.7 + rng.normal(0, 0.15, n), 0.01, 0.99)
        probB = np.clip(y * 0.55 + rng.normal(0, 0.2, n), 0.01, 0.99)
        return y, pA, pB, probA, probB

    def test_mcnemar_p_value_range(self):
        from backend.pipeline.evaluate.statistical_tests import StatisticalTests
        y, pA, pB, _, _ = self._dummy_preds()
        result = StatisticalTests().mcnemar(y.tolist(), pA.tolist(), pB.tolist())
        assert 0.0 <= result["p_value"] <= 1.0
        assert "significant" in result

    def test_delong_auc_fields(self):
        from backend.pipeline.evaluate.statistical_tests import StatisticalTests
        y, _, _, probA, probB = self._dummy_preds()
        result = StatisticalTests().delong_auc(y.tolist(), probA.tolist(), probB.tolist())
        assert "auc_a" in result
        assert "auc_b" in result
        assert 0.0 <= result["p_value"] <= 1.0


# ===========================================================================
# 7. RadiomicsPipeline must not replace missing extraction with synthetic output
# ===========================================================================
class TestRadiomicsPipeline:
    def test_missing_extractor_does_not_return_synthetic_features(self, monkeypatch):
        from backend.pipeline.radiomics.pipeline import RadiomicsPipeline
        pipe = RadiomicsPipeline(use_cache=False)
        vol  = np.random.randn(64, 64, 64).astype(np.float32)
        mask = np.ones((64, 64, 64), dtype=np.uint8)
        monkeypatch.setattr(pipe, "_run_pyradiomics", lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("missing dependency")))
        with pytest.raises(RuntimeError, match="required for feature extraction"):
            pipe.extract(vol, mask, spacing=(1.0, 1.0, 1.0))

    def test_requires_a_segmentation_mask(self):
        from backend.pipeline.radiomics.pipeline import RadiomicsPipeline
        pipe = RadiomicsPipeline(use_cache=False)
        with pytest.raises(ValueError, match="segmentation mask is required"):
            pipe.extract(np.zeros((4, 4, 4), dtype=np.float32))


# ===========================================================================
# 8. FeatureSelector
# ===========================================================================
class TestFeatureSelector:
    def test_lasso_reduces_features(self):
        from backend.pipeline.radiomics.feature_selector import FeatureSelector
        rng = np.random.default_rng(42)
        X   = rng.normal(size=(150, 80)).astype(np.float32)
        y   = rng.integers(0, 2, 150)
        sel = FeatureSelector(method="lasso", n_features=20)
        X_sel, names = sel.fit_transform(X, y, [f"feat_{i}" for i in range(80)])
        assert X_sel.shape[1] <= 20
        assert len(names) == X_sel.shape[1]

    def test_mutual_info_reduces_features(self):
        from backend.pipeline.radiomics.feature_selector import FeatureSelector
        rng = np.random.default_rng(99)
        X   = rng.normal(size=(100, 50)).astype(np.float32)
        y   = rng.integers(0, 2, 100)
        sel = FeatureSelector(method="mutual_info", n_features=10)
        X_sel, names = sel.fit_transform(X, y)
        assert X_sel.shape[1] <= 10


# ===========================================================================
# 9. PipelineOrchestrator
# ===========================================================================
class TestPipelineOrchestrator:
    def test_instantiation(self):
        from backend.pipeline import PipelineOrchestrator
        orch = PipelineOrchestrator.from_config()
        assert orch is not None

    def test_stage_enum_complete(self):
        from backend.pipeline.orchestrator import PipelineStage
        stages = list(PipelineStage)
        assert len(stages) >= 10


# ===========================================================================
# 10. Monitoring
# ===========================================================================
class TestMonitoring:
    def test_logger_importable(self):
        from backend.monitoring import get_logger
        log = get_logger("test")
        assert log is not None

    def test_metrics_registry_importable(self):
        from backend.monitoring import MetricsRegistry
        # Should not raise even without prometheus_client
        MetricsRegistry.record_inference("TestModel", "benign", 50.0)

    def test_health_checker_instantiates(self):
        from backend.monitoring import HealthChecker
        hc = HealthChecker()
        assert hc is not None

# ===========================================================================
# 11. Empty Scan Verdict (NOT ASSESSED)
# ===========================================================================
class TestEmptyScanVerdict:
    def test_no_candidates_verdict_not_assessed(self):
        from backend.pipeline.job_queue import JobQueueManager, PipelineJob
        from backend.api.service import STUDY_STORE
        from backend.dicom.volume_manager import volume_manager, StudyVolume

        # All-air volume (-1000 HU) will yield 0 detector candidates
        vol = np.full((32, 32, 32), -1000.0, dtype=np.float32)
        study_id = "TEST-EMPTY-SCAN-NOT-ASSESSED"
        volume_manager.register_volume(study_id, StudyVolume(study_id, vol))
        STUDY_STORE[study_id] = {
            "study_id": study_id,
            "patient_id": "TEST-PT",
            "series_id": "TEST-SERIES",
            "slice_count": 32,
            "nodules": [],
        }

        try:
            mgr = JobQueueManager()
            job = PipelineJob(job_id="test-job-empty", study_id=study_id, mode="fast_triage")
            mgr._execute_pipeline(job)

            report = job.triage_report
            assert report is not None
            assert report["primary_verdict"] == "NOT ASSESSED"
            assert report.get("lung_rads") is None or report.get("lung_rads") == "Not assessed"
            assert "No candidates detected. This does not exclude nodules; review the scan manually." in report["action_recommendation"]
            assert "No candidates detected. This does not exclude nodules; review the scan manually." in report["clinical_impression"]
            assert report["primary_verdict"] != "BENIGN"
            assert report["primary_verdict"] != "LOW_PROBABILITY"
        finally:
            STUDY_STORE.pop(study_id, None)
            volume_manager._volumes.pop(study_id, None)


# ===========================================================================
# 11. 3D Region-Growing Nodule Segmentation
# ===========================================================================
class Test3DRegionGrowing:
    def test_physical_to_voxel(self):
        from backend.segmentation.morphology import physical_to_voxel
        coord_xyz = (100.0, -50.0, 30.0)
        origin_xyz = (0.0, -100.0, 10.0)
        spacing_zyx = (2.0, 1.0, 0.5)

        vz, vy, vx = physical_to_voxel(coord_xyz, origin_xyz, spacing_zyx)
        assert vz == int(round((30.0 - 10.0) / 2.0)) == 10
        assert vy == int(round((-50.0 - (-100.0)) / 1.0)) == 50
        assert vx == int(round((100.0 - 0.0) / 0.5)) == 200

    def test_region_grow_synthetic_sphere(self):
        from backend.segmentation.morphology import region_grow_3d
        vol = np.full((32, 32, 32), -800.0, dtype=np.float32)
        center = (16, 16, 16)
        radius_mm = 4.0
        spacing = (1.0, 1.0, 1.0)

        for z in range(32):
            for y in range(32):
                for x in range(32):
                    dist = np.sqrt((z - 16)**2 + (y - 16)**2 + (x - 16)**2)
                    if dist <= radius_mm:
                        vol[z, y, x] = -100.0

        res = region_grow_3d(
            volume_hu=vol,
            seed_zyx=center,
            spacing_zyx_mm=spacing,
            hu_min=-600.0,
            hu_max=200.0,
            max_radius_mm=10.0,
            max_diameter_mm=30.0,
        )

        assert not res['is_empty']
        assert not res['is_oversized']
        assert 7.0 <= res['equivalent_diameter_mm'] <= 9.0
        assert 200.0 <= res['volume_mm3'] <= 350.0

    def test_region_grow_empty_outside_hu(self):
        from backend.segmentation.morphology import region_grow_3d
        vol = np.full((16, 16, 16), -950.0, dtype=np.float32)
        res = region_grow_3d(
            volume_hu=vol,
            seed_zyx=(8, 8, 8),
            hu_min=-600.0,
            hu_max=200.0,
        )
        assert res['is_empty']
        assert res['voxel_count'] == 0
        assert res['equivalent_diameter_mm'] == 0.0

    def test_region_grow_oversized(self):
        from backend.segmentation.morphology import region_grow_3d
        vol = np.full((40, 40, 40), -100.0, dtype=np.float32)
        res = region_grow_3d(
            volume_hu=vol,
            seed_zyx=(20, 20, 20),
            hu_min=-600.0,
            hu_max=200.0,
            max_radius_mm=25.0,
            max_diameter_mm=15.0,
        )
        assert res['is_oversized']
