"""
backend/models/segmentation/unet3d/model.py
============================================
3D U-Net for volumetric lung / nodule segmentation.

Architecture (Ronneberger et al., 2015 — 3D extension):
  Encoder: 4 resolution levels, channels [32, 64, 128, 256]
  Bottleneck: 512 channels
  Decoder: 4 upsampling levels with skip connections
  Output: binary/multi-class voxel-wise segmentation mask

Input:  (B, 1, D, H, W)  — single-channel CT volume (normalised HU)
Output: (B, num_classes, D, H, W)  — raw logits

Metrics: Dice, IoU, HD95 (via SimpleITK), Sensitivity, Specificity
Training loss: DiceCELoss from MONAI
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────

class ConvBnRelu3d(nn.Module):
    """Conv3d → BatchNorm3d → ReLU (pre-activation friendly)."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv3d(nn.Module):
    """Two consecutive ConvBnRelu3d blocks (standard U-Net unit)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu3d(in_ch, out_ch),
            ConvBnRelu3d(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """MaxPool3d(2,2,2) → DoubleConv3d."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3d(in_ch, out_ch)

    def forward(self, x: torch.Tensor):
        pooled = self.pool(x)
        out = self.conv(pooled)
        return out


class DecoderBlock(nn.Module):
    """Trilinear upsample ×2 → concat skip → DoubleConv3d."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = DoubleConv3d(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims mismatch (rare at boundary)
        diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
#  3D U-Net
# ─────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """3D U-Net for volumetric CT segmentation.

    Args:
        in_channels: Input channels (1 for CT).
        num_classes:  1 → sigmoid binary; >1 → softmax multi-class.
        base_ch:      Channel count at the first encoder level (default 32).
        dropout_p:    Spatial dropout probability in bottleneck.

    Example::
        model = UNet3D(in_channels=1, num_classes=3)   # left/right/combined
        # CT patch: 128³ at 1mm³ spacing
        x = torch.randn(1, 1, 128, 128, 128)
        logits = model(x)   # (1, 3, 128, 128, 128)
    """

    CHANNEL_MULTIPLIERS = (1, 2, 4, 8)   # relative to base_ch

    def __init__(self, in_channels: int = 1, num_classes: int = 3,
                 base_ch: int = 32, dropout_p: float = 0.2):
        super().__init__()
        c = base_ch
        ch = [c * m for m in self.CHANNEL_MULTIPLIERS]   # [32,64,128,256]

        # Encoder
        self.enc0 = DoubleConv3d(in_channels, ch[0])     # full res
        self.enc1 = EncoderBlock(ch[0], ch[1])
        self.enc2 = EncoderBlock(ch[1], ch[2])
        self.enc3 = EncoderBlock(ch[2], ch[3])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv3d(ch[3], ch[3] * 2),              # 512 ch
            nn.Dropout3d(p=dropout_p),
        )

        # Decoder
        self.dec3 = DecoderBlock(ch[3] * 2, ch[3], ch[3])
        self.dec2 = DecoderBlock(ch[3], ch[2], ch[2])
        self.dec1 = DecoderBlock(ch[2], ch[1], ch[1])
        self.dec0 = DecoderBlock(ch[1], ch[0], ch[0])

        # Segmentation head
        self.head = nn.Conv3d(ch[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b = self.bottleneck(s3)

        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        return self.head(d0)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor,
                     threshold: float = 0.5) -> torch.Tensor:
        """Inference helper: returns binary/argmax mask."""
        logits = self.forward(x)
        if logits.shape[1] == 1:
            return (torch.sigmoid(logits) > threshold).long()
        return torch.argmax(logits, dim=1, keepdim=True)


# ─────────────────────────────────────────────────────────────
#  Dice Loss (training) — standalone, no MONAI dependency
# ─────────────────────────────────────────────────────────────

class DiceLoss3D(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        flat_p = probs.view(-1)
        flat_t = targets.view(-1).float()
        intersection = (flat_p * flat_t).sum()
        return 1.0 - (2 * intersection + self.smooth) / (
            flat_p.sum() + flat_t.sum() + self.smooth)


class DiceCELoss3D(nn.Module):
    """Combined Dice + CrossEntropy loss (standard for medical segmentation)."""
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss3D()
        self.ce = nn.BCEWithLogitsLoss()
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, targets) + \
               self.cw * self.ce(logits, targets.float())


# ─────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = UNet3D(in_channels=1, num_classes=3, base_ch=32)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print(f"UNet3D output: {out.shape}")   # (1, 3, 64, 64, 64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
