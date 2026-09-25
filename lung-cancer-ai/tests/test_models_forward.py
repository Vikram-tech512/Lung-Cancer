"""
tests/test_models_forward.py
============================
Unit smoke tests verifying forward pass tensor dimensions for concrete 3D DL models.
"""

import pytest
import torch

from ml.segmentation.unet3d import UNet3D
from ml.classification.resnet3d import ResNet3D50
from ml.classification.densenet3d import DenseNet3D121
from ml.transformers.swin3d import SwinTransformer3D
from ml.ensemble.ensemble import MultimodalEnsemble, ModelPrediction

def test_unet3d_forward():
    model = UNet3D(in_channels=1, num_classes=1, base_ch=16)
    model.eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 32, 32, 32)

def test_resnet3d_forward():
    model = ResNet3D50(in_channels=1, num_classes=2, dropout_p=0.3)
    model.eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 2)

def test_densenet3d_forward():
    model = DenseNet3D121(in_channels=1, num_classes=2, growth_rate=16)
    model.eval()
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 2)

def test_swin3d_forward():
    model = SwinTransformer3D(
        in_channels=1,
        num_classes=2,
        embed_dim=48,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=(4, 4, 4)
    )
    model.eval()
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 2)

def test_multimodal_ensemble():
    ensemble = MultimodalEnsemble()
    preds = {
        "resnet3d": ModelPrediction(model_name="3D ResNet-50", probability_malignant=0.85),
        "swin3d": ModelPrediction(model_name="3D Swin Transformer", probability_malignant=0.82),
        "radiomics_xgb": ModelPrediction(model_name="Radiomics XGBoost", probability_malignant=0.79)
    }
    result = ensemble.predict(preds)
    assert result.ensemble_verdict == "MALIGNANT"
    assert 0.80 <= result.ensemble_probability <= 0.86
