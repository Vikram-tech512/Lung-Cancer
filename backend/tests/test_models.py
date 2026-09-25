"""
tests/test_models.py
====================
Smoke tests verifying forward pass tensor dimensions for all implemented 3D models:
- 3D U-Net (Segmentation)
- 3D ResNet-50 (Classification)
- 3D DenseNet-121 (Classification)
- 3D Swin Transformer (Classification)
- Multimodal Ensemble
"""

import pytest
import torch
import numpy as np

from backend.models.segmentation.unet3d.model import UNet3D
from backend.models.classification.resnet3d.model import ResNet3D50
from backend.models.classification.densenet3d.model import DenseNet3D121
from backend.models.classification.swin3d.model import SwinTransformer3D
from backend.models.ensemble.ensemble import MultimodalEnsemble, ModelPrediction

def test_unet3d_forward():
    model = UNet3D(in_channels=1, num_classes=1, base_ch=16)
    model.eval()
    # Mini-volume for quick forward pass test
    dummy_input = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 1, 32, 32, 32), f"Expected (1, 1, 32, 32, 32), got {output.shape}"

def test_resnet3d_forward():
    model = ResNet3D50(in_channels=1, num_classes=2, dropout_p=0.3)
    model.eval()
    dummy_input = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 2), f"Expected (1, 2), got {logits.shape}"

def test_densenet3d_forward():
    model = DenseNet3D121(in_channels=1, num_classes=2, growth_rate=16)
    model.eval()
    dummy_input = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 2), f"Expected (1, 2), got {logits.shape}"

def test_swin3d_forward():
    model = SwinTransformer3D(
        patch_size=(4, 4, 4),
        in_channels=1,
        num_classes=2,
        embed_dim=48,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=(4, 4, 4)
    )
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 2), f"Expected (1, 2), got {logits.shape}"

def test_multimodal_ensemble():
    ensemble = MultimodalEnsemble()
    preds = {
        "resnet3d": ModelPrediction(model_name="3D ResNet-50", probability_malignant=0.8),
        "swin3d": ModelPrediction(model_name="3D Swin Transformer", probability_malignant=0.7),
        "radiomics_xgb": ModelPrediction(model_name="Radiomics XGBoost", probability_malignant=0.75)
    }
    result = ensemble.predict(preds)
    assert 0.0 <= result.ensemble_probability <= 1.0
    assert result.ensemble_verdict in ["BENIGN", "MALIGNANT", "INDETERMINATE"]
    assert len(result.individual_predictions) == 3
