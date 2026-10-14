"""
backend/models/classification/histonet/model.py
================================================
2D CNN classifier for histopathological lung images.

Provides three model variants via factory function:
  - "resnet50"       : ResNet-50 pretrained on ImageNet, fine-tuned classifier head
  - "efficientnet"   : EfficientNet-B2 pretrained, lightweight & accurate
  - "densenet"       : DenseNet-121 pretrained, strong for medical imaging

All variants share the same interface:
    model = build_histonet(variant="resnet50", num_classes=3, pretrained=True)
    logits = model(imgs)   # imgs: (B, 3, H, W)

Designed for LC25000 lung histopathology:
    Class 0 → lung_aca  (Adenocarcinoma)
    Class 1 → lung_n    (Normal)
    Class 2 → lung_scc  (Squamous Cell Carcinoma)

NOTE: Output is research-grade classification — NOT a clinical diagnosis.
      Always display: "MODEL OUTPUT — NOT CLINICAL EVIDENCE"
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
#  Classifier head (shared)
# ─────────────────────────────────────────────────────────────────────────────

class HistoClassifierHead(nn.Module):
    """Dropout → FC → classification head replacing the backbone's default head."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_histonet(
    variant: Literal["resnet50", "efficientnet", "densenet"] = "resnet50",
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.4,
) -> nn.Module:
    """
    Build a 2D transfer-learning model for histopathology classification.

    Args:
        variant:          Model backbone to use.
        num_classes:      Number of output classes (3 for lung-only, 5 for all).
        pretrained:       Load ImageNet pretrained weights.
        freeze_backbone:  If True, freeze all backbone layers (train head only).
        dropout:          Dropout rate in the classifier head.

    Returns:
        nn.Module ready for fine-tuning.
    """
    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if variant == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = HistoClassifierHead(in_features, num_classes, dropout)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    elif variant == "efficientnet":
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = HistoClassifierHead(in_features, num_classes, dropout)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    elif variant == "densenet":
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = HistoClassifierHead(in_features, num_classes, dropout)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose: resnet50, efficientnet, densenet")

    return model


# ─────────────────────────────────────────────────────────────────────────────
#  MC Dropout inference helper
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    T: int = 30,
    device: torch.device | None = None,
) -> dict:
    """
    Monte Carlo Dropout uncertainty estimation.

    Returns:
        {
          "mean_probs":  (num_classes,) mean softmax probabilities,
          "std_probs":   (num_classes,) std of softmax (epistemic uncertainty),
          "pred_class":  int — argmax of mean_probs,
        }
    """
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)

    # Enable dropout at inference (MC mode)
    model.train()
    with torch.no_grad():
        preds = []
        for _ in range(T):
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds.append(probs.cpu())

    model.eval()
    stacked = torch.stack(preds, dim=0)           # (T, B, C)
    mean_p  = stacked.mean(dim=0)                  # (B, C)
    std_p   = stacked.std(dim=0)                   # (B, C)

    return {
        "mean_probs": mean_p,
        "std_probs":  std_p,
        "pred_class": mean_p.argmax(dim=-1),
    }
