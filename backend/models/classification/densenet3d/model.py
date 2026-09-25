"""
backend/models/classification/densenet3d/model.py
==================================================
3D DenseNet-121 for volumetric nodule classification.

Architecture (Huang et al., 2017 — 3D extension):
  Stem:   Conv3d(1,64,kernel=7,stride=2) → BN → ReLU → MaxPool3d
  Dense Block 1:  6 layers,  growth_rate=32
  Transition 1:   BN → Conv1³ → AvgPool3d(2,2,2)
  Dense Block 2: 12 layers
  Transition 2
  Dense Block 3: 24 layers
  Transition 3
  Dense Block 4: 16 layers
  Head:           BN → ReLU → AdaptiveAvgPool3d(1) → Dropout → FC(2)

Input:  (B, 1, 64, 64, 64)
Output: (B, 2)   — [benign_logit, malignant_logit]

Dense connectivity: every layer receives feature maps from ALL preceding layers.
This provides implicit deep supervision and strong gradient flow,
particularly useful when training data is limited (LIDC-IDRI ~1000 patients).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  Dense Layer
# ─────────────────────────────────────────────────────────────

class DenseLayer3d(nn.Module):
    """BN → ReLU → Conv1³(bottleneck) → BN → ReLU → Conv3³ → output k channels."""

    def __init__(self, in_ch: int, growth_rate: int = 32,
                 bn_size: int = 4, dropout_p: float = 0.0):
        super().__init__()
        mid_ch = bn_size * growth_rate
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, growth_rate, kernel_size=3,
                      padding=1, bias=False),
        )
        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_feat = self.drop(self.block(x))
        return torch.cat([x, new_feat], dim=1)


# ─────────────────────────────────────────────────────────────
#  Dense Block
# ─────────────────────────────────────────────────────────────

class DenseBlock3d(nn.Module):
    """Stack of n DenseLayer3d; input grows by growth_rate per layer."""

    def __init__(self, in_ch: int, n_layers: int,
                 growth_rate: int = 32, dropout_p: float = 0.0):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(DenseLayer3d(ch, growth_rate,
                                       dropout_p=dropout_p))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ─────────────────────────────────────────────────────────────
#  Transition Layer
# ─────────────────────────────────────────────────────────────

class TransitionLayer3d(nn.Module):
    """Reduce channels by theta=0.5, halve spatial resolution."""

    def __init__(self, in_ch: int, theta: float = 0.5):
        super().__init__()
        out_ch = int(in_ch * theta)
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────
#  3D DenseNet-121
# ─────────────────────────────────────────────────────────────

class DenseNet3D121(nn.Module):
    """3D DenseNet-121 for nodule malignancy classification.

    Args:
        in_channels:  CT input channels (1).
        num_classes:  2 (benign / malignant).
        growth_rate:  k = 32 (default, per original paper).
        block_layers: [6,12,24,16] for DenseNet-121.
        dropout_p:    Dropout in dense layers and head.

    Benchmark role: Model B (DenseNet vs ResNet comparison).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 growth_rate: int = 32,
                 block_layers: list[int] = None,
                 init_ch: int = 64, dropout_p: float = 0.2):
        super().__init__()
        if block_layers is None:
            block_layers = [6, 12, 24, 16]

        # ── Stem ───────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_ch, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # ── Dense blocks + Transitions ─────────────────────────
        ch = init_ch
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, n_layers in enumerate(block_layers):
            block = DenseBlock3d(ch, n_layers, growth_rate, dropout_p)
            self.blocks.append(block)
            ch = block.out_channels

            if i < len(block_layers) - 1:   # no transition after last block
                trans = TransitionLayer3d(ch)
                self.transitions.append(trans)
                ch = trans.out_channels
            else:
                self.transitions.append(None)   # placeholder

        # ── Classification head ────────────────────────────────
        self.final_bn   = nn.BatchNorm3d(ch)
        self.avgpool    = nn.AdaptiveAvgPool3d(1)
        self.dropout    = nn.Dropout(p=dropout_p)
        self.fc         = nn.Linear(ch, num_classes)
        self._out_ch    = ch

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for i, (block, trans) in enumerate(
                zip(self.blocks, self.transitions)):
            x = block(x)
            if trans is not None:
                x = trans(x)

        x = F.relu(self.final_bn(x), inplace=True)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Last dense block output for Grad-CAM."""
        x = self.stem(x)
        for i, (block, trans) in enumerate(
                zip(self.blocks, self.transitions)):
            x = block(x)
            if trans is not None and i < len(self.blocks) - 1:
                x = trans(x)
        return F.relu(self.final_bn(x), inplace=True)


# ─────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DenseNet3D121(in_channels=1, num_classes=2)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print(f"DenseNet3D-121 output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
