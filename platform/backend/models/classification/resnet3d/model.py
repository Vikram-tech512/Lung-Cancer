"""
backend/models/classification/resnet3d/model.py
================================================
3D ResNet-50 for volumetric nodule classification.

Architecture:
  Stem:   Conv3d(1,64,kernel=(3,7,7),stride=(1,2,2)) → BN → ReLU → MaxPool3d
  Layer1: 3 × Bottleneck3d(64  → 256,   stride=1)
  Layer2: 4 × Bottleneck3d(128 → 512,   stride=2 at first block)
  Layer3: 6 × Bottleneck3d(256 → 1024,  stride=2 at first block)
  Layer4: 3 × Bottleneck3d(512 → 2048,  stride=2 at first block)
  Head:   AdaptiveAvgPool3d(1) → Dropout → FC(2)

Input:  (B, 1, 64, 64, 64)   — 64³ nodule ROI at 1 mm³ spacing
Output: (B, 2)               — [benign_logit, malignant_logit]

MC Dropout:  Call model.enable_mc_dropout() before stochastic inference.
Uncertainty: Run T=30 forward passes, compute std of softmax probs.

NOTE: Output is labelled "malignant-class probability" — NOT a diagnosis.
      Always display: "MODEL OUTPUT — NOT CLINICAL EVIDENCE"
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


# ─────────────────────────────────────────────────────────────
#  Bottleneck block (3D)
# ─────────────────────────────────────────────────────────────

class Bottleneck3d(nn.Module):
    """3D ResNet Bottleneck: 1×1×1 → 3×3×3 → 1×1×1 with residual."""
    expansion: int = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        out_ch = mid_ch * self.expansion

        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(mid_ch)

        self.conv2 = nn.Conv3d(mid_ch, mid_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(mid_ch)

        self.conv3 = nn.Conv3d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(out_ch)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ─────────────────────────────────────────────────────────────
#  3D ResNet-50
# ─────────────────────────────────────────────────────────────

class ResNet3D50(nn.Module):
    """3D ResNet-50 for nodule malignancy classification.

    Args:
        in_channels: CT input channels (1).
        num_classes: Output classes (2: benign / malignant).
        dropout_p:   Dropout before final FC (for MC Dropout uncertainty).

    Model D from the benchmark suite (3D ResNet-50 variant).
    """

    # [layers per stage] for ResNet-50
    LAYERS = [3, 4, 6, 3]
    # [mid_channels] for each stage before ×4 expansion
    MID_CH = [64, 128, 256, 512]

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 dropout_p: float = 0.3):
        super().__init__()
        self.in_ch = 64

        # ── Stem ──────────────────────────────────────────────
        # Asymmetric kernel: temporal dim preserved, spatial downsampled
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                         padding=(0, 1, 1)),
        )

        # ── Residual stages ───────────────────────────────────
        self.layer1 = self._make_layer(self.MID_CH[0], self.LAYERS[0], stride=1)
        self.layer2 = self._make_layer(self.MID_CH[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.MID_CH[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.MID_CH[3], self.LAYERS[3], stride=2)

        # ── Classification head ───────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(self.MID_CH[3] * Bottleneck3d.expansion,
                                 num_classes)

        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────

    def _make_layer(self, mid_ch: int, n_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        out_ch = mid_ch * Bottleneck3d.expansion
        downsample = None

        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

        layers = [Bottleneck3d(self.in_ch, mid_ch, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, n_blocks):
            layers.append(Bottleneck3d(self.in_ch, mid_ch))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def enable_mc_dropout(self):
        """Switch to MC Dropout inference mode.
        Call model.eval() first, then this to keep only Dropout active.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    # ── Forward ───────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return layer4 feature map for 3D Grad-CAM."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


# ─────────────────────────────────────────────────────────────
#  3D ResNet-18 (lightweight alternative for low-memory GPUs)
# ─────────────────────────────────────────────────────────────

class BasicBlock3d(nn.Module):
    """3D Basic block for ResNet-18/34."""
    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D18(nn.Module):
    """Lightweight 3D ResNet-18. Use when GPU memory < 8 GB."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 dropout_p: float = 0.3):
        super().__init__()
        self.in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, (3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch: int, n: int, stride: int) -> nn.Sequential:
        ds = None
        if stride != 1 or self.in_ch != out_ch:
            ds = nn.Sequential(
                nn.Conv3d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers = [BasicBlock3d(self.in_ch, out_ch, stride, ds)]
        self.in_ch = out_ch
        for _ in range(1, n):
            layers.append(BasicBlock3d(out_ch, out_ch))
        return nn.Sequential(*layers)

    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.dropout(self.avgpool(x).flatten(1)))


# ─────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 3D ResNet-50 ===")
    m50 = ResNet3D50(in_channels=1, num_classes=2)
    x = torch.randn(2, 1, 64, 64, 64)
    print(f"Output: {m50(x).shape}")
    print(f"Params: {sum(p.numel() for p in m50.parameters()):,}")

    print("\n=== 3D ResNet-18 ===")
    m18 = ResNet3D18(in_channels=1, num_classes=2)
    print(f"Output: {m18(x).shape}")
    print(f"Params: {sum(p.numel() for p in m18.parameters()):,}")
