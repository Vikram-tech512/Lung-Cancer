# src/model/vit_seg.py
"""Vision Transformer segmentation head.
Uses the ViT encoder defined in src/model/vit.py and adds a simple 1x1 convolution
followed by bilinear up‑sampling to produce a pixel‑wise binary mask.
"""
import torch
import torch.nn as nn
from .vit import ViT

class ViTSeg(nn.Module):
    """ViT‑based segmentation model.
    Args:
        img_size (int): Input image size (assumes square).
        patch_size (int): Patch size used by the backbone ViT.
        num_classes (int): Number of output channels (1 for binary mask).
        embed_dim (int): Embedding dimension of the ViT backbone.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of MLP in transformer.
        dropout (float): Dropout probability.
    """
    def __init__(self, img_size=224, patch_size=16, num_classes=1, embed_dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        # Backbone ViT (returns patch embeddings without cls token)
        self.backbone = ViT(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        # 1x1 conv to map embedding dim to num_classes per patch
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        # Upsample to original image resolution (patch_size * patch grid size = img_size)
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        # x shape: (B, C, H, W)
        patch_emb = self.backbone(x)               # (B, N, embed_dim)
        B, N, D = patch_emb.shape
        # Compute spatial dimensions of patch grid
        h = w = int(N ** 0.5)
        # Reshape to (B, embed_dim, h, w)
        feats = patch_emb.permute(0, 2, 1).reshape(B, D, h, w)
        seg = self.head(feats)                     # (B, num_classes, h, w)
        seg = self.upsample(seg)                   # (B, num_classes, H, W)
        # For binary mask we squeeze the channel dimension
        return seg.squeeze(1)  # (B, H, W)

if __name__ == "__main__":
    dummy = torch.randn(1, 3, 224, 224)
    model = ViTSeg()
    out = model(dummy)
    print(f"Segmentation output shape: {out.shape}")
