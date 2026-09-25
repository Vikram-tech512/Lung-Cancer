# src/model/cnn.py
"""Convolutional Neural Network (CNN) for lung cancer classification.
A custom deep CNN with residual-style blocks, global average pooling, and dropout.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → Conv → BatchNorm + residual shortcut."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class LungCancerCNN(nn.Module):
    """Deep CNN classifier for lung cancer images.
    Args:
        num_classes (int): Number of output classes (e.g., 3 for benign/malignant/normal).
        in_channels (int): Number of input image channels.
    """
    def __init__(self, num_classes=3, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),

            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),

            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        """Return the final conv feature map (before GAP) for Grad-CAM."""
        return self.features(x)


if __name__ == "__main__":
    model = LungCancerCNN(num_classes=3)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (2, 3)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
