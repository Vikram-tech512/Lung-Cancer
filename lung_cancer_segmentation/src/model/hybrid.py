# src/model/hybrid.py
"""Hybrid CNN–Vision Transformer for lung cancer classification.
Combines local feature extraction (CNN backbone) with global contextual
modeling (Transformer encoder) as described in the project synopsis.
"""
import torch
import torch.nn as nn
from src.model.vit import TransformerBlock


class CNNBackbone(nn.Module):
    """Lightweight CNN feature extractor that reduces spatial resolution
    and produces patch-like feature tokens for the transformer."""
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 224 → 112
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
            # 56 → 28
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            # 28 → 14
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.proj = nn.Conv2d(256, embed_dim, 1)  # 1x1 conv to embed_dim

    def forward(self, x):
        x = self.conv_layers(x)       # (B, 256, 14, 14)
        x = self.proj(x)              # (B, embed_dim, 14, 14)
        return x


class HybridViT(nn.Module):
    """Hybrid CNN–ViT classifier.
    The CNN backbone extracts local features into a 14×14 grid (196 tokens),
    which are then processed by a Transformer encoder for global reasoning.

    Args:
        num_classes (int): Output classes.
        embed_dim (int): Token embedding dimension.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        mlp_dim (int): MLP hidden dim.
        dropout (float): Dropout rate.
    """
    def __init__(self, num_classes=3, in_channels=3, embed_dim=256,
                 depth=6, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.cnn_backbone = CNNBackbone(in_channels, embed_dim)
        self.n_patches = 14 * 14  # 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        # CNN feature extraction
        feat = self.cnn_backbone(x)                    # (B, embed_dim, 14, 14)
        tokens = feat.flatten(2).transpose(1, 2)       # (B, 196, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, 197, embed_dim)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # Transformer encoder
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out)

    def get_cnn_features(self, x):
        """Return the CNN backbone feature map for Grad-CAM."""
        return self.cnn_backbone.conv_layers(x)

    def get_attention_maps(self, x):
        """Return attention weights from all transformer blocks."""
        B = x.shape[0]
        feat = self.cnn_backbone(x)
        tokens = feat.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        attn_maps = []
        for block in self.blocks:
            tokens = block(tokens)
            attn_maps.append(block.attn.attn_weights)
        return attn_maps


if __name__ == "__main__":
    model = HybridViT(num_classes=3)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
