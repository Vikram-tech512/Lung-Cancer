"""
backend/models/classification/swin3d/model.py
=============================================
3D Swin Transformer for volumetric nodule classification.

Architecture (Liu et al., "Video Swin Transformer", 2021 — adapted for CT):

  How a CT volume is converted into model input:
  ─────────────────────────────────────────────
  1. Input CT nodule ROI: (B, 1, D, H, W) = (B, 1, 64, 64, 64)  at 1mm³
  2. Patch partition: split into non-overlapping 3D patches of size
     (patch_d, patch_h, patch_w) = (4, 4, 4)
     → token grid: (D/4, H/4, W/4) = (16, 16, 16) tokens = 4096 tokens
     → each token embeds 4×4×4×1 = 64 values → linear project to embed_dim=96
  3. Swin stages with 3D window-based shifted attention:
     Stage 0: window_size=(4,4,4), [W-MSA, SW-MSA] × 2 layers, 96ch
     Stage 1: patch merge (2×2×2 → ×8 tokens) → 192ch, × 2 layers
     Stage 2: patch merge → 384ch, × 6 layers
     Stage 3: patch merge → 768ch, × 2 layers
  4. Head: LayerNorm → AdaptiveAvgPool3d(1) → Dropout → FC(2)

Window attention: queries/keys/values only interact within local 3D windows.
Shifted windows (SW-MSA): cyclic shift creates cross-window connections in
alternate layers without additional memory overhead.

Extends your existing ViT/TransformerBlock code from
lung_cancer_segmentation/src/model/vit.py into the 3D domain.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def window_partition_3d(x: torch.Tensor,
                        ws: Tuple[int, int, int]) -> torch.Tensor:
    """Partition (B, D, H, W, C) into (num_windows*B, wd, wh, ww, C)."""
    B, D, H, W, C = x.shape
    wd, wh, ww = ws
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd, wh, ww, C)


def window_reverse_3d(windows: torch.Tensor, ws: Tuple[int, int, int],
                      D: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition_3d."""
    wd, wh, ww = ws
    B = int(windows.shape[0] / (D * H * W / (wd * wh * ww)))
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


def compute_attn_mask_3d(D: int, H: int, W: int,
                         ws: Tuple[int, int, int],
                         shift: Tuple[int, int, int],
                         device: torch.device) -> torch.Tensor:
    """Compute additive attention mask for SW-MSA."""
    wd, wh, ww = ws
    sd, sh, sw = shift
    img_mask = torch.zeros(1, D, H, W, 1, device=device)
    d_slices = (slice(0, -wd), slice(-wd, -sd), slice(-sd, None))
    h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
    w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
    cnt = 0
    for ds in d_slices:
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[:, ds, hs, ws_, :] = cnt
                cnt += 1
    mask_windows = window_partition_3d(img_mask, ws)   # (nW, wd, wh, ww, 1)
    mask_windows = mask_windows.view(-1, wd * wh * ww)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0, 0.0)


# ─────────────────────────────────────────────────────────────
#  3D Window Multi-Head Self-Attention
# ─────────────────────────────────────────────────────────────

class WindowAttention3d(nn.Module):
    """3D W-MSA / SW-MSA with relative position bias."""

    def __init__(self, dim: int, window_size: Tuple[int, int, int],
                 num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim         = dim
        self.ws          = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5

        # Relative position bias table
        wd, wh, ww = window_size
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2*wd-1)*(2*wh-1)*(2*ww-1), num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Compute relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww),
            indexing="ij")).flatten(1)                        # (3, wd*wh*ww)
        rel = coords[:, :, None] - coords[:, None, :]         # (3, N, N)
        rel = rel.permute(1, 2, 0)                            # (N, N, 3)
        rel[:, :, 0] += wd - 1
        rel[:, :, 1] += wh - 1
        rel[:, :, 2] += ww - 1
        rel[:, :, 0] *= (2*wh-1) * (2*ww-1)
        rel[:, :, 1] *= (2*ww-1)
        self.register_buffer("rel_pos_index", rel.sum(-1))

        self.qkv      = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

        # Store for attention rollout
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        self.attn_weights = attn.detach()   # for attention rollout
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


# ─────────────────────────────────────────────────────────────
#  Swin Transformer Block 3D
# ─────────────────────────────────────────────────────────────

class SwinBlock3d(nn.Module):
    """Pre-norm Swin block. Even idx → W-MSA, odd idx → SW-MSA."""

    def __init__(self, dim: int, input_resolution: Tuple[int, int, int],
                 num_heads: int, window_size: Tuple[int, int, int] = (4, 4, 4),
                 shift: bool = False, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.dim        = dim
        self.res        = input_resolution
        self.ws         = window_size
        self.shift      = shift
        shift_size      = tuple(s // 2 for s in window_size) if shift else (0, 0, 0)
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention3d(dim, window_size, num_heads,
                                       attn_drop=attn_drop,
                                       proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

        if any(s > 0 for s in shift_size):
            D, H, W = input_resolution
            self.register_buffer(
                "attn_mask",
                compute_attn_mask_3d(D, H, W, window_size,
                                     shift_size, torch.device("cpu"))
            )
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, H, W = self.res
        B, L, C = x.shape
        assert L == D * H * W

        shortcut = x
        x = self.norm1(x).view(B, D, H, W, C)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0],
                                       -self.shift_size[1],
                                       -self.shift_size[2]),
                           dims=(1, 2, 3))

        windows = window_partition_3d(x, self.ws)
        wd, wh, ww = self.ws
        windows = windows.view(-1, wd * wh * ww, C)

        mask = self.attn_mask
        if mask is not None:
            mask = mask.to(x.device)
        attn_out = self.attn(windows, mask=mask)

        attn_out = attn_out.view(-1, *self.ws, C)
        x = window_reverse_3d(attn_out, self.ws, D, H, W)

        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))

        x = x.view(B, L, C) + shortcut
        return x + self.mlp(self.norm2(x))


# ─────────────────────────────────────────────────────────────
#  Patch Merging (3D downsampling)
# ─────────────────────────────────────────────────────────────

class PatchMerging3d(nn.Module):
    """Concat 2×2×2 neighbours → Linear projection to halve resolution."""

    def __init__(self, in_ch: int, out_ch: int | None = None):
        super().__init__()
        out_ch = out_ch or in_ch * 2
        self.norm  = nn.LayerNorm(8 * in_ch)
        self.proj  = nn.Linear(8 * in_ch, out_ch, bias=False)

    def forward(self, x: torch.Tensor,
                res: Tuple[int, int, int]) -> torch.Tensor:
        D, H, W = res
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        x = torch.cat([
            x[:, 0::2, 0::2, 0::2, :],
            x[:, 1::2, 0::2, 0::2, :],
            x[:, 0::2, 1::2, 0::2, :],
            x[:, 0::2, 0::2, 1::2, :],
            x[:, 1::2, 1::2, 0::2, :],
            x[:, 1::2, 0::2, 1::2, :],
            x[:, 0::2, 1::2, 1::2, :],
            x[:, 1::2, 1::2, 1::2, :],
        ], dim=-1)   # (B, D/2, H/2, W/2, 8C)
        return self.proj(self.norm(x.view(B, -1, 8 * C)))


# ─────────────────────────────────────────────────────────────
#  3D Swin Transformer (Swin-T configuration)
# ─────────────────────────────────────────────────────────────

class SwinTransformer3D(nn.Module):
    """3D Swin Transformer for nodule malignancy classification.

    Configuration: Swin-T  (embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24])

    Args:
        in_channels:   CT input channels (1).
        num_classes:   2 (benign / malignant).
        patch_size:    3D patch partition size (4,4,4).
        embed_dim:     Initial embedding dimension (96).
        depths:        Swin blocks per stage.
        num_heads:     Attention heads per stage.
        window_size:   Local attention window (4,4,4).
        dropout:       Dropout rate.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 patch_size: Tuple[int, int, int] = (4, 4, 4),
                 embed_dim: int = 96,
                 depths: list[int] = None,
                 num_heads: list[int] = None,
                 window_size: Tuple[int, int, int] = (4, 4, 4),
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 attn_drop: float = 0.0):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.depths     = depths

        # ── Patch Embedding ─────────────────────────────────────
        pd, ph, pw = patch_size
        self.patch_embed = nn.Conv3d(in_channels, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size, bias=False)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_drop   = nn.Dropout(p=dropout)

        # ── Swin Stages ─────────────────────────────────────────
        self.stages      = nn.ModuleList()
        self.merges      = nn.ModuleList()
        dim = embed_dim

        # initial token resolution after patch embed (assuming 64³ input)
        # 64 / 4 = 16  →  (16, 16, 16)
        res = (64 // pd, 64 // ph, 64 // pw)

        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage = nn.ModuleList([
                SwinBlock3d(dim, res, heads, window_size,
                            shift=(j % 2 == 1),
                            mlp_ratio=mlp_ratio,
                            dropout=dropout, attn_drop=attn_drop)
                for j in range(depth)
            ])
            self.stages.append(stage)

            if i < len(depths) - 1:
                merge = PatchMerging3d(dim)
                self.merges.append(merge)
                dim *= 2
                res  = tuple(r // 2 for r in res)
            else:
                self.merges.append(None)

        # ── Head ────────────────────────────────────────────────
        self.norm    = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.head    = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _patch_embed_forward(self, x: torch.Tensor):
        """Embed patches, flatten to sequence."""
        x = self.patch_embed(x)              # (B, C, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, N, C)
        x = self.patch_norm(x)
        return x, (D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, res = self._patch_embed_forward(x)
        x = self.pos_drop(x)

        for i, (stage, merge) in enumerate(zip(self.stages, self.merges)):
            for blk in stage:
                x = blk(x)
            if merge is not None:
                x = merge(x, res)
                res = tuple(r // 2 for r in res)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).squeeze(-1)
        x = self.dropout(x)
        return self.head(x)

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return attention weights from the last Swin stage (for rollout)."""
        x, res = self._patch_embed_forward(x)
        x = self.pos_drop(x)
        attn_maps = []
        for i, (stage, merge) in enumerate(zip(self.stages, self.merges)):
            for blk in stage:
                x = blk(x)
                if blk.attn.attn_weights is not None:
                    attn_maps.append(blk.attn.attn_weights)
            if merge is not None:
                x = merge(x, res)
                res = tuple(r // 2 for r in res)
        return attn_maps


# ─────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SwinTransformer3D(in_channels=1, num_classes=2)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print(f"Swin3D output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    attn = model.get_attention_maps(x)
    print(f"Attention maps collected: {len(attn)}")
