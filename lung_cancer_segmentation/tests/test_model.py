# tests/test_model.py
"""Unit tests for CNN, ViT, Hybrid models, Perturbations, and XAI."""
import pytest
import torch
import numpy as np
from PIL import Image

from src.model.cnn import LungCancerCNN
from src.model.vit import ViT
from src.model.hybrid import HybridViT
from src.robustness.perturbations import apply_perturbation, get_all_perturbation_names
from src.explainability.gradcam import GradCAM, GradCAMPP
from src.explainability.attention_viz import extract_attention_map, attention_rollout


class TestClassificationModels:
    def test_cnn_output_shape(self):
        model = LungCancerCNN(num_classes=3)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 3)

    def test_cnn_features_for_gradcam(self):
        model = LungCancerCNN(num_classes=3)
        x = torch.randn(1, 3, 224, 224)
        feat = model.get_feature_maps(x)
        assert feat.ndim == 4
        assert feat.shape[0] == 1

    def test_vit_output_shape(self):
        model = ViT(num_classes=3, depth=2, heads=4, embed_dim=64, mlp_dim=128)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 3)

    def test_vit_attention_maps(self):
        model = ViT(num_classes=3, depth=2, heads=4, embed_dim=64, mlp_dim=128)
        x = torch.randn(1, 3, 224, 224)
        maps = model.get_attention_maps(x)
        assert len(maps) == 2
        assert maps[0].shape[0] == 1  # batch size
        assert maps[0].shape[1] == 4  # heads

    def test_hybrid_output_shape(self):
        model = HybridViT(num_classes=3, embed_dim=64, depth=2, heads=4, mlp_dim=128)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 3)

    def test_hybrid_features_and_attention(self):
        model = HybridViT(num_classes=3, embed_dim=64, depth=2, heads=4, mlp_dim=128)
        x = torch.randn(1, 3, 224, 224)
        cnn_feat = model.get_cnn_features(x)
        assert cnn_feat.ndim == 4
        attn_maps = model.get_attention_maps(x)
        assert len(attn_maps) == 2


class TestRobustnessPerturbations:
    def test_all_registered_perturbations_execute(self):
        test_img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        names = get_all_perturbation_names()
        assert len(names) >= 10
        for name in names:
            perturbed = apply_perturbation(test_img, name)
            assert isinstance(perturbed, Image.Image)
            assert perturbed.size == (100, 100)


class TestExplainability:
    def test_gradcam_generation(self):
        model = LungCancerCNN(num_classes=3)
        target_layer = model.features[-1]
        gcam = GradCAM(model, target_layer)
        x = torch.randn(1, 3, 224, 224)
        heatmap = gcam.generate(x, target_class=1)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_attention_rollout(self):
        model = ViT(num_classes=3, depth=2, heads=4, embed_dim=64, mlp_dim=128)
        x = torch.randn(1, 3, 224, 224)
        rollout = attention_rollout(model, x)
        assert isinstance(rollout, np.ndarray)
        assert rollout.min() >= 0.0
        assert rollout.max() <= 1.0
