"""
backend/models/detection/detector3d/retinanet3d.py
===================================================
3D RetinaNet for pulmonary nodule detection in volumetric CT.

Architecture:
  Backbone: 3D ResNet-18 (lightweight, suitable for full-volume inference)
  Neck:     3D Feature Pyramid Network (FPN) — P3, P4, P5 levels
  Heads:    Classification head (focal loss) + Regression head (smooth L1)
  Anchors:  Multi-scale spherical anchors: diameters [5,8,12,18,25] mm

Input:  (B, 1, D, H, W) — whole lung-masked CT volume at 1mm³
Output: {
    "boxes":   (N, 6) — [x1,y1,z1,x2,y2,z2] in voxel coords
    "scores":  (N,)   — nodule probability
    "centers": (N, 3) — [cx, cy, cz]
    "diameters": (N,) — estimated diameter in mm
}

Training:
  Classification: Focal loss (α=0.25, γ=2.0)
  Regression:     Smooth L1 loss on box deltas
  NMS:            3D IoU-based NMS, iou_threshold=0.3

NOTE: This module provides the detection architecture.
Inference results are stored per-study and per-nodule in the database.
No nodule is classified without first being detected by this model.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  3D ResNet-18 FPN Backbone
# ─────────────────────────────────────────────────────────────

class Conv3dBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class BasicBlock3dFPN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class BackboneFPN3d(nn.Module):
    """ResNet-18 3D backbone producing C3,C4,C5 feature maps for FPN."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        # Stem: output C1 (stride 2)
        self.stem = nn.Sequential(
            Conv3dBnRelu(in_channels, 32, kernel=7, stride=2, padding=3),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        # Stage 2 → C2 (stride 4)
        self.layer1 = nn.Sequential(
            BasicBlock3dFPN(32, 64), BasicBlock3dFPN(64, 64))
        # Stage 3 → C3 (stride 8)
        self.layer2 = nn.Sequential(
            BasicBlock3dFPN(64, 128, stride=2), BasicBlock3dFPN(128, 128))
        # Stage 4 → C4 (stride 16)
        self.layer3 = nn.Sequential(
            BasicBlock3dFPN(128, 256, stride=2), BasicBlock3dFPN(256, 256))
        # Stage 5 → C5 (stride 32)
        self.layer4 = nn.Sequential(
            BasicBlock3dFPN(256, 512, stride=2), BasicBlock3dFPN(512, 512))

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5   # (stride 8, 16, 32)


class FPN3d(nn.Module):
    """3D Feature Pyramid Network merging C3,C4,C5 → P3,P4,P5."""

    def __init__(self, in_chs: Tuple[int, int, int] = (128, 256, 512),
                 fpn_ch: int = 256):
        super().__init__()
        self.lat3 = nn.Conv3d(in_chs[0], fpn_ch, 1, bias=False)
        self.lat4 = nn.Conv3d(in_chs[1], fpn_ch, 1, bias=False)
        self.lat5 = nn.Conv3d(in_chs[2], fpn_ch, 1, bias=False)
        self.out3 = nn.Conv3d(fpn_ch, fpn_ch, 3, padding=1, bias=False)
        self.out4 = nn.Conv3d(fpn_ch, fpn_ch, 3, padding=1, bias=False)
        self.out5 = nn.Conv3d(fpn_ch, fpn_ch, 3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[2:],
                             mode="trilinear", align_corners=False) + y

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self._upsample_add(p5, self.lat4(c4))
        p3 = self._upsample_add(p4, self.lat3(c3))
        return self.out3(p3), self.out4(p4), self.out5(p5)


# ─────────────────────────────────────────────────────────────
#  Anchor Generator (3D spherical anchors)
# ─────────────────────────────────────────────────────────────

class AnchorGenerator3d:
    """Generate spherical anchor boxes for 3D detection.

    Anchor diameters: [5, 8, 12, 18, 25] mm (chosen to cover LIDC-IDRI
    nodule size distribution: 3–30 mm, median ~7 mm).
    """
    DIAMETERS_MM = [5.0, 8.0, 12.0, 18.0, 25.0]

    def __init__(self, strides: List[int] = (8, 16, 32),
                 voxel_spacing_mm: float = 1.0):
        self.strides = strides
        self.spacing = voxel_spacing_mm
        # Convert diameter mm → voxels
        self.radii_vox = [d / (2 * voxel_spacing_mm)
                          for d in self.DIAMETERS_MM]

    def generate(self, feature_maps: List[torch.Tensor],
                 input_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Return anchor boxes (N_anchors, 6) [x1,y1,z1,x2,y2,z2]."""
        D, H, W = input_shape
        all_anchors = []
        for fmap, stride in zip(feature_maps, self.strides):
            _, _, fd, fh, fw = fmap.shape
            # Centers on the original volume grid
            zs = torch.arange(fd, dtype=torch.float32) * stride + stride / 2
            ys = torch.arange(fh, dtype=torch.float32) * stride + stride / 2
            xs = torch.arange(fw, dtype=torch.float32) * stride + stride / 2
            zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
            centers = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

            for r in self.radii_vox:
                boxes = torch.cat([centers - r, centers + r], dim=-1)
                all_anchors.append(boxes)

        return torch.cat(all_anchors, dim=0)   # (N, 6)


# ─────────────────────────────────────────────────────────────
#  Detection Heads
# ─────────────────────────────────────────────────────────────

class ClassHead3d(nn.Module):
    """4× Conv3d → sigmoid classification (focal loss target)."""

    def __init__(self, fpn_ch: int = 256, n_anchors: int = 5,
                 n_classes: int = 1):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv3d(fpn_ch, fpn_ch, 3, padding=1),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv3d(fpn_ch, n_anchors * n_classes, 3, padding=1))
        self.net = nn.Sequential(*layers)
        # Init bias for focal loss stability (p≈0.01 at start)
        nn.init.constant_(self.net[-1].bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        return self.net(x)


class RegHead3d(nn.Module):
    """4× Conv3d → box delta regression."""

    def __init__(self, fpn_ch: int = 256, n_anchors: int = 5):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv3d(fpn_ch, fpn_ch, 3, padding=1),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv3d(fpn_ch, n_anchors * 6, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
#  Focal Loss
# ─────────────────────────────────────────────────────────────

class FocalLoss3D(nn.Module):
    """Binary focal loss for class imbalance in nodule detection."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        p  = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = target * p + (1 - target) * (1 - p)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────
#  3D RetinaNet
# ─────────────────────────────────────────────────────────────

class RetinaNet3D(nn.Module):
    """3D RetinaNet for pulmonary nodule detection.

    Args:
        in_channels:      CT input channels (1).
        fpn_ch:           FPN channel width (256).
        n_anchor_sizes:   Number of anchor diameter scales (5).
        score_thresh:     Minimum score for a detection (0.05).
        nms_iou_thresh:   3D NMS IoU threshold (0.3).

    Usage::
        model = RetinaNet3D()
        # inference
        volume = torch.randn(1, 1, 128, 192, 192)   # (B, 1, D, H, W)
        detections = model.detect(volume)
        # Each detection: {center, diameter, score, bbox}
    """

    def __init__(self, in_channels: int = 1, fpn_ch: int = 256,
                 n_anchor_sizes: int = 5,
                 score_thresh: float = 0.05,
                 nms_iou_thresh: float = 0.3):
        super().__init__()
        self.backbone    = BackboneFPN3d(in_channels)
        self.fpn         = FPN3d(fpn_ch=fpn_ch)
        self.cls_head    = ClassHead3d(fpn_ch, n_anchor_sizes)
        self.reg_head    = RegHead3d(fpn_ch, n_anchor_sizes)
        self.anchor_gen  = AnchorGenerator3d()
        self.focal_loss  = FocalLoss3D()
        self.score_thresh = score_thresh
        self.nms_thresh   = nms_iou_thresh

    def forward(self, x: torch.Tensor):
        """Training forward — returns (cls_logits, reg_deltas) per FPN level."""
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.fpn(c3, c4, c5)
        cls_outs, reg_outs = [], []
        for p in (p3, p4, p5):
            cls_outs.append(self.cls_head(p))
            reg_outs.append(self.reg_head(p))
        return cls_outs, reg_outs

    @torch.no_grad()
    def detect(self, volume: torch.Tensor,
               voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
               ) -> List[Dict]:
        """Inference: return list of detected nodule candidates.

        Returns:
            List of dicts: {
                'center': [cx, cy, cz] in voxel coords,
                'diameter_mm': float,
                'score': float (0–1),
                'bbox': [x1, y1, z1, x2, y2, z2]
            }
        """
        self.eval()
        cls_outs, reg_outs = self.forward(volume)
        D, H, W = volume.shape[2], volume.shape[3], volume.shape[4]
        anchors = self.anchor_gen.generate(
            [volume[:, :, ::8, ::8, ::8],
             volume[:, :, ::16, ::16, ::16],
             volume[:, :, ::32, ::32, ::32]],
            (D, H, W)).to(volume.device)

        # Collect scores and deltas across all FPN levels
        all_scores, all_deltas = [], []
        for cls_out, reg_out in zip(cls_outs, reg_outs):
            B, _, fd, fh, fw = cls_out.shape
            scores = torch.sigmoid(cls_out).permute(0, 2, 3, 4, 1).reshape(B, -1)
            deltas = reg_out.permute(0, 2, 3, 4, 1).reshape(B, -1, 6)
            all_scores.append(scores)
            all_deltas.append(deltas)

        all_scores = torch.cat(all_scores, dim=1)[0]   # (N,)
        all_deltas = torch.cat(all_deltas, dim=1)[0]   # (N, 6)

        # Decode boxes
        if len(anchors) != len(all_scores):
            n = min(len(anchors), len(all_scores))
            anchors = anchors[:n]; all_scores = all_scores[:n]
            all_deltas = all_deltas[:n]

        centers = (anchors[:, :3] + anchors[:, 3:]) / 2
        sizes   = anchors[:, 3:] - anchors[:, :3]
        pred_centers = centers + all_deltas[:, :3] * sizes
        pred_sizes   = sizes * torch.exp(all_deltas[:, 3:].clamp(-3, 3))
        pred_boxes   = torch.cat([pred_centers - pred_sizes/2,
                                   pred_centers + pred_sizes/2], dim=-1)

        # Threshold + 3D NMS (simple greedy)
        keep = all_scores > self.score_thresh
        scores  = all_scores[keep]
        boxes   = pred_boxes[keep]
        cents   = pred_centers[keep]
        diams   = pred_sizes[keep].mean(dim=-1) * voxel_spacing[0]

        # Greedy NMS
        order  = scores.argsort(descending=True)
        picked = []
        while len(order) > 0:
            i = order[0].item()
            picked.append(i)
            if len(order) == 1:
                break
            ious = self._iou3d(boxes[i:i+1], boxes[order[1:]])
            order = order[1:][ious < self.nms_thresh]

        results = []
        for i in picked:
            results.append({
                "center": cents[i].cpu().tolist(),
                "diameter_mm": float(diams[i].cpu()),
                "score": float(scores[i].cpu()),
                "bbox": boxes[i].cpu().tolist(),
            })
        return results

    @staticmethod
    def _iou3d(box1: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """3D IoU between one box and a set of boxes."""
        inter_min = torch.max(box1[:, :3], boxes[:, :3])
        inter_max = torch.min(box1[:, 3:], boxes[:, 3:])
        inter_dims = (inter_max - inter_min).clamp(min=0)
        inter_vol  = inter_dims.prod(dim=-1)
        vol1 = (box1[:, 3:] - box1[:, :3]).clamp(min=0).prod()
        vol2 = (boxes[:, 3:] - boxes[:, :3]).clamp(min=0).prod(dim=-1)
        return inter_vol / (vol1 + vol2 - inter_vol + 1e-8)


# ─────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = RetinaNet3D(in_channels=1)
    vol = torch.randn(1, 1, 64, 64, 64)
    cls_outs, reg_outs = model(vol)
    print("Training output shapes:")
    for c, r in zip(cls_outs, reg_outs):
        print(f"  cls: {c.shape}  reg: {r.shape}")
    detections = model.detect(vol)
    print(f"\nDetected {len(detections)} nodule candidates")
    if detections:
        print(f"Top detection: {detections[0]}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
