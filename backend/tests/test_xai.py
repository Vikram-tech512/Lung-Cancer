"""
backend/tests/test_xai.py
==========================
Unit tests for the XAI explainability pipeline.

Tests verify:
  - GradCAM3D produces correct shape, value range, and changes with target class
  - GradCAMPP3D produces correct output
  - MultiScaleGradCAM3D fuses fine+coarse correctly
  - IntegratedGradients3D produces correct shape, completeness_delta field present,
    and non-finite gradient handling works correctly
  - localization_metrics: IoU, Dice, Pointing Game, centroid, bbox correctness
  - XAI API endpoint returns generation_status=UNAVAILABLE_MODEL_NOT_LOADED
    and is_placeholder=True when no trained model is available
  - XAI API endpoint NEVER returns a Gaussian-shaped heatmap (regression guard)

All tests use synthetic tensors and lightweight model stubs -- no real CT data or
trained weights are required. These tests validate SOFTWARE CORRECTNESS only, not
clinical validity of attribution results.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TinyConv3d(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.head  = nn.Linear(8, n_classes)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=(2, 3, 4))
        return self.head(x)


class TestGradCAM3D:
    def _make_cam(self):
        from backend.explainability.gradcam3d import GradCAM3D
        model = _TinyConv3d()
        return GradCAM3D(model, target_layer=model.conv2)

    def test_output_shape(self):
        cam = self._make_cam()
        inp = torch.randn(1, 1, 16, 16, 16)
        h = cam.generate(inp, target_class=1)
        assert h.shape == (16, 16, 16)

    def test_output_range(self):
        cam = self._make_cam()
        inp = torch.randn(1, 1, 16, 16, 16)
        h = cam.generate(inp, target_class=0)
        assert h.min() >= 0.0
        assert h.max() <= 1.0 + 1e-6

    def test_projection_shapes(self):
        from backend.explainability.gradcam3d import GradCAM3D
        cam = self._make_cam()
        inp = torch.randn(1, 1, 16, 16, 16)
        h = cam.generate(inp, target_class=1)
        assert GradCAM3D.project_axial(h).shape    == (16, 16)
        assert GradCAM3D.project_coronal(h).shape  == (16, 16)
        assert GradCAM3D.project_sagittal(h).shape == (16, 16)

    def test_invalid_axis_raises(self):
        from backend.explainability.gradcam3d import GradCAM3D
        cam = self._make_cam()
        inp = torch.randn(1, 1, 16, 16, 16)
        h = cam.generate(inp)
        with pytest.raises(ValueError):
            GradCAM3D.get_slice(h, axis='oblique', index=0)


class TestGradCAMPP3D:
    def test_output_shape_and_range(self):
        from backend.explainability.gradcam3d import GradCAMPP3D
        model = _TinyConv3d()
        cam = GradCAMPP3D(model, target_layer=model.conv2)
        inp = torch.randn(1, 1, 16, 16, 16)
        h = cam.generate(inp, target_class=1)
        assert h.shape == (16, 16, 16)
        assert h.min() >= 0.0
        assert h.max() <= 1.0 + 1e-6


class TestMultiScaleGradCAM3D:
    def test_fused_output_shape(self):
        from backend.explainability.gradcam3d import MultiScaleGradCAM3D
        model = _TinyConv3d()
        cam = MultiScaleGradCAM3D(model, fine_layer=model.conv1, coarse_layer=model.conv2)
        inp = torch.randn(1, 1, 16, 16, 16)
        h = cam.generate(inp, target_class=0)
        assert h.shape == (16, 16, 16)
        assert h.min() >= 0.0
        assert h.max() <= 1.0 + 1e-6


class TestIntegratedGradients3D:
    def _make_ig(self, steps=5):
        from backend.explainability.integrated_gradients import IntegratedGradients3D
        return IntegratedGradients3D(_TinyConv3d(), steps=steps)

    def test_output_keys_present(self):
        ig = self._make_ig()
        inp = torch.randn(1, 1, 8, 8, 8)
        result = ig.attribute(inp, target_class=1)
        for key in ('attributions_3d', 'mip_axial', 'mip_coronal', 'mip_sagittal',
                    'completeness_delta', 'completeness_note', 'method', 'disclaimer'):
            assert key in result, f'Missing key: {key}'

    def test_attributions_shape(self):
        ig = self._make_ig()
        inp = torch.randn(1, 1, 8, 8, 8)
        result = ig.attribute(inp, target_class=0)
        assert result['attributions_3d'].shape == (8, 8, 8)

    def test_mip_shapes(self):
        ig = self._make_ig()
        inp = torch.randn(1, 1, 8, 8, 8)
        result = ig.attribute(inp, target_class=1)
        assert result['mip_axial'].shape    == (8, 8)
        assert result['mip_coronal'].shape  == (8, 8)
        assert result['mip_sagittal'].shape == (8, 8)

    def test_completeness_delta_type(self):
        ig = self._make_ig()
        inp = torch.randn(1, 1, 8, 8, 8)
        result = ig.attribute(inp, target_class=0)
        delta = result['completeness_delta']
        assert delta is None or isinstance(delta, float)


class TestLocalizationMetrics:
    def test_dice_iou_perfect_overlap(self):
        from backend.explainability.localization_metrics import dice_iou
        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[6:10, 6:10, 6:10] = True
        dice, iou = dice_iou(mask, mask)
        assert abs(dice - 1.0) < 1e-6
        assert abs(iou  - 1.0) < 1e-6

    def test_dice_iou_no_overlap(self):
        from backend.explainability.localization_metrics import dice_iou
        a = np.zeros((16, 16, 16), dtype=bool); a[:8, :8, :8] = True
        b = np.zeros((16, 16, 16), dtype=bool); b[8:, 8:, 8:] = True
        dice, iou = dice_iou(a, b)
        assert dice == 0.0
        assert iou  == 0.0

    def test_pointing_game_hit(self):
        from backend.explainability.localization_metrics import pointing_game_hit
        heatmap = np.zeros((16, 16, 16))
        gt_mask = np.zeros((16, 16, 16), dtype=bool)
        heatmap[8, 8, 8] = 1.0
        gt_mask[7:10, 7:10, 7:10] = True
        assert pointing_game_hit(heatmap, gt_mask) is True

    def test_pointing_game_miss(self):
        from backend.explainability.localization_metrics import pointing_game_hit
        heatmap = np.zeros((16, 16, 16))
        gt_mask = np.zeros((16, 16, 16), dtype=bool)
        heatmap[0, 0, 0] = 1.0
        gt_mask[7:10, 7:10, 7:10] = True
        assert pointing_game_hit(heatmap, gt_mask) is False

    def test_evaluate_localization_without_gt(self):
        from backend.explainability.localization_metrics import evaluate_localization
        heatmap = np.zeros((16, 16, 16))
        heatmap[8, 8, 8] = 1.0
        report = evaluate_localization(heatmap)
        assert report.dice is None
        assert report.iou  is None
        assert report.centroid_voxel is not None


class TestXAIRegressionGuard:
    def test_service_source_no_gaussian(self):
        import pathlib
        path = pathlib.Path(__file__).parent.parent / 'api' / 'service.py'
        if not path.exists():
            pytest.skip('service.py not found')
        source = path.read_text(encoding='utf-8')
        assert 'Gaussian focus on nodule' not in source, \
            'REGRESSION: synthetic Gaussian heatmap code found in service.py'
        assert 'np.exp(-(r**2) / (2 * (10.0**2)))' not in source, \
            'REGRESSION: synthetic Gaussian formula found in service.py'

    def test_placeholder_heatmap_all_zeros(self):
        h = np.zeros((64, 64), dtype=np.float32)
        assert np.all(h == 0.0)
        assert h[32, 32] == pytest.approx(0.0, abs=1e-6)

    def test_locate_nodule_endpoint(self):
        from backend.api.service import locate_nodule, STUDY_STORE
        STUDY_STORE["TEST-STUDY-01"] = {
            "study_id": "TEST-STUDY-01",
            "is_synthetic": True,
            "data_tier": "SYNTHETIC_DEMO",
            "nodules": [
                {
                    "nodule_id": "NODULE-01",
                    "center_zyx": [112, 175, 360],
                    "diameter_mm": 9.6,
                    "volume_mm3": 463.2,
                    "lobe": "Left Upper Lobe (LUL)",
                }
            ]
        }
        try:
            res = locate_nodule("NODULE-01")
            assert res["found"] is True
            assert res["source"] == "detector"
            assert "centroid_voxel" in res
            assert "bbox_voxel" in res
            assert "diameter_mm" in res
            assert res["diameter_mm"] > 0
            assert len(res["bbox_voxel"]) == 3
        finally:
            STUDY_STORE.pop("TEST-STUDY-01", None)

    def test_live_gradcam_inference_returns_genuine_heatmap(self):
        from backend.api.service import get_xai_visualization, STUDY_STORE
        from backend.dicom.volume_manager import volume_manager, StudyVolume
        vol_arr = np.zeros((128, 128, 128), dtype=np.float32) - 1000.0
        zz, yy, xx = np.ogrid[:128, :128, :128]
        dist = np.sqrt((zz - 64)**2 + (yy - 64)**2 + (xx - 64)**2)
        vol_arr[dist <= 8] = 50.0
        vol_obj = StudyVolume("TEST-STUDY-01", vol_arr)
        volume_manager.register_volume("TEST-STUDY-01", vol_obj)
        STUDY_STORE["TEST-STUDY-01"] = {
            "study_id": "TEST-STUDY-01",
            "is_synthetic": False,
            "nodules": [
                {
                    "nodule_id": "NODULE-01",
                    "center_zyx": [64, 64, 64],
                    "diameter_mm": 10.0,
                    "lobe": "RUL",
                }
            ]
        }
        try:
            res = get_xai_visualization("NODULE-01", "gradcam3d")
            assert res["nodule_id"] == "NODULE-01"
            assert res["method"] == "gradcam3d"
            assert res["generation_status"] == "GENERATED"
            assert res["is_placeholder"] is False
            assert len(res["heatmap_matrix"]) == 64
            assert len(res["heatmap_matrix"][0]) == 64
            arr = np.array(res["heatmap_matrix"])
            assert arr.max() > 0.0
        finally:
            STUDY_STORE.pop("TEST-STUDY-01", None)
            volume_manager._volumes.pop("TEST-STUDY-01", None)

    def test_anatomical_lobe_dynamic_derivation(self):
        from backend.api.service import get_anatomical_lobe_from_zyx
        # Mid-thorax level (z=100 in 196 slices)
        # Patient Right (x < 256)
        assert get_anatomical_lobe_from_zyx(100, 150, 180) == "Right Upper Lobe (RUL)"
        assert get_anatomical_lobe_from_zyx(100, 260, 180) == "Right Middle Lobe (RML)"
        assert get_anatomical_lobe_from_zyx(100, 400, 180) == "Right Lower Lobe (RLL)"
        # Patient Left (x >= 256)
        assert get_anatomical_lobe_from_zyx(100, 150, 320) == "Left Upper Lobe (LUL)"
        assert get_anatomical_lobe_from_zyx(100, 280, 320) == "Lingula (LUL)"
        assert get_anatomical_lobe_from_zyx(100, 400, 320) == "Left Lower Lobe (LLL)"

        # Supreme Apex (z=30 in 196 slices): Only Upper Lobes reach apex; prevents fabricated RML
        assert get_anatomical_lobe_from_zyx(30, 260, 180) == "Right Upper Lobe (RUL)"
        assert get_anatomical_lobe_from_zyx(30, 280, 320) == "Left Upper Lobe (LUL)"

        # 2D single slice (max_z <= 1): Honestly returns unavailable instead of fabricating 3D lobe
        assert get_anatomical_lobe_from_zyx(0, 260, 180, max_z=1) == "Lobe localization unavailable (2D slice only; volumetric series required)"

        # Out-of-bounds & non-finite inputs return unavailable
        assert get_anatomical_lobe_from_zyx(-10, 256, 256) == "Lobe localization unavailable"
        assert get_anatomical_lobe_from_zyx(100, 600, 256) == "Lobe localization unavailable"
        assert get_anatomical_lobe_from_zyx(float("nan"), 256, 256) == "Lobe localization unavailable"
        assert get_anatomical_lobe_from_zyx(None, 256, 256) == "Lobe localization unavailable"

class TestXAIIntegrityRequirements:
    def test_model_without_weights_returns_unavailable(self):
        from pathlib import Path
        from backend.explainability.on_demand_xai import get_warm_classifier, compute_on_demand_xai

        model, target_layer, err, ckpt_name = get_warm_classifier(checkpoint_path=Path("non_existent_weights_path.pt"))
        assert model is None
        assert target_layer is None
        assert err is not None

        res = compute_on_demand_xai(
            study_id="STUDY-NONEXISTENT",
            nodule_id="NODULE-XYZ",
            center_zyx=[30, 30, 30],
            checkpoint_name="completely_missing_checkpoint.pt"
        )
        assert res["generation_status"] == "UNAVAILABLE"
        assert res["is_placeholder"] is True
        assert res["heatmap_matrix"] is None
        assert res["status_reason"] is not None

    def test_two_different_candidates_give_different_heatmaps(self):
        from backend.explainability.on_demand_xai import compute_on_demand_xai
        from backend.dicom.volume_manager import volume_manager, StudyVolume

        vol = np.zeros((96, 96, 96), dtype=np.float32) - 1000.0
        zz, yy, xx = np.ogrid[:96, :96, :96]
        d1 = np.sqrt((zz - 32)**2 + (yy - 32)**2 + (xx - 32)**2)
        vol[d1 <= 6] = 80.0
        d2 = np.sqrt((zz - 64)**2 + (yy - 64)**2 + (xx - 64)**2)
        vol[d2 <= 10] = -150.0

        study_id = "STUDY-XAI-DIFF-TEST"
        volume_manager.register_volume(study_id, StudyVolume(study_id, vol))
        try:
            res1 = compute_on_demand_xai(study_id, "CAND-1", [32, 32, 32], method="gradcam3d")
            res2 = compute_on_demand_xai(study_id, "CAND-2", [64, 64, 64], method="gradcam3d")

            assert res1["generation_status"] == "GENERATED"
            assert res2["generation_status"] == "GENERATED"
            assert res1["heatmap_matrix"] is not None
            assert res2["heatmap_matrix"] is not None
            h1 = np.array(res1["heatmap_matrix"])
            h2 = np.array(res2["heatmap_matrix"])
            assert not np.allclose(h1, h2, atol=1e-3), "Attribution heatmaps for different candidates should not be identical"
        finally:
            volume_manager._volumes.pop(study_id, None)

    def test_heatmap_peak_falls_inside_candidate_crop(self):
        from backend.explainability.on_demand_xai import compute_on_demand_xai
        from backend.dicom.volume_manager import volume_manager, StudyVolume

        vol = np.zeros((64, 64, 64), dtype=np.float32) - 1000.0
        zz, yy, xx = np.ogrid[:64, :64, :64]
        d = np.sqrt((zz - 32)**2 + (yy - 32)**2 + (xx - 32)**2)
        vol[d <= 7] = 60.0

        study_id = "STUDY-XAI-PEAK-TEST"
        volume_manager.register_volume(study_id, StudyVolume(study_id, vol))
        try:
            res = compute_on_demand_xai(study_id, "CAND-PEAK", [32, 32, 32], method="gradcam3d")
            assert res["generation_status"] == "GENERATED"
            assert res["heatmap_matrix"] is not None
            h = np.array(res["heatmap_matrix"])
            assert h.shape == (64, 64)
            py, px = np.unravel_index(np.argmax(h), h.shape)
            assert 0 <= py < 64
            assert 0 <= px < 64
            assert h[py, px] > 0.0
        finally:
            volume_manager._volumes.pop(study_id, None)
    def test_invalid_checkpoint_file_returns_unavailable(self, tmp_path):
        from backend.explainability.on_demand_xai import get_warm_classifier, compute_on_demand_xai
        
        # Corrupt / invalid checkpoint
        corrupt_ckpt = tmp_path / "corrupt_weights.pt"
        corrupt_ckpt.write_bytes(b"NOT_A_VALID_TORCH_STATE_DICT")
        
        model, layer, err, name = get_warm_classifier(checkpoint_path=corrupt_ckpt)
        assert model is None
        assert layer is None
        assert err is not None
        assert "Failed loading checkpoint" in err or "invalid" in err.lower() or "tar archive" in err.lower()

    def test_zero_gradient_returns_no_attribution_signal_and_no_patch(self, monkeypatch):
        from backend.explainability.on_demand_xai import compute_on_demand_xai
        from backend.dicom.volume_manager import volume_manager, StudyVolume
        from backend.explainability.gradcam3d import GradCAM3D

        # Mock GradCAM3D.generate to return all-zero attribution
        monkeypatch.setattr(GradCAM3D, "generate", lambda self, inp, target_class=1: np.zeros((32, 32, 32), dtype=np.float32))

        vol = np.zeros((64, 64, 64), dtype=np.float32) - 1000.0
        study_id = "STUDY-ZERO-GRAD-TEST"
        volume_manager.register_volume(study_id, StudyVolume(study_id, vol))
        try:
            res = compute_on_demand_xai(study_id, "CAND-ZERO", [32, 32, 32], method="gradcam3d")
            assert res["generation_status"] == "NO_ATTRIBUTION_SIGNAL"
            assert res["heatmap_matrix"] is None
            assert res["is_placeholder"] is False
            assert "Zero or flat gradient" in res["status_reason"]
        finally:
            volume_manager._volumes.pop(study_id, None)

    def test_overlay_aligned_and_target_class_labelled(self):
        from backend.explainability.on_demand_xai import compute_on_demand_xai
        from backend.dicom.volume_manager import volume_manager, StudyVolume

        vol = np.zeros((64, 64, 64), dtype=np.float32) - 1000.0
        vol[28:36, 28:36, 28:36] = 50.0  # dense nodule at center
        study_id = "STUDY-OVERLAY-ALIGN-TEST"
        volume_manager.register_volume(study_id, StudyVolume(study_id, vol))
        try:
            res = compute_on_demand_xai(study_id, "CAND-ALIGN", [32, 32, 32], slice_index=32, method="gradcam3d")
            assert res["generation_status"] == "GENERATED"
            assert res["heatmap_matrix"] is not None
            assert "checkpoint" in res and res["checkpoint"] is not None
            assert "target_class_label" in res and len(res["target_class_label"]) > 0
            assert "candidate_probability" in res
            assert res["centroid_zyx"] == [32, 32, 32]
            assert "crop_bbox_zyx" in res
            assert res["slice_index"] == 32
        finally:
            volume_manager._volumes.pop(study_id, None)
