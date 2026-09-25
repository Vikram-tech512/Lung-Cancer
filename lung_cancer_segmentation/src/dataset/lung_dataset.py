# src/dataset/lung_dataset.py
"""PyTorch Dataset for lung cancer classification.
Supports loading from a directory of labelled images (class-per-folder or CSV label file)
and optional on-the-fly perturbation for robustness analysis.
"""
import os
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Default preprocessing (ImageNet normalisation)
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Training augmentations
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LungCancerDataset(Dataset):
    """Lung cancer classification dataset.

    Directory structure (class-per-folder):
        img_dir/
            class_0/  (e.g., "benign")
                img001.png
                img002.png
            class_1/  (e.g., "malignant")
                ...
            class_2/  (e.g., "normal")
                ...

    Or provide a label_file (CSV) with columns: filename, label (int).

    Args:
        img_dir: Root image directory.
        label_file: Optional CSV path (filename,label).
        transform: torchvision transforms. Defaults to DEFAULT_TRANSFORM.
        perturbation: Optional perturbation name to apply before transform.
    """

    def __init__(
        self,
        img_dir: str,
        label_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        perturbation: Optional[str] = None,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform or DEFAULT_TRANSFORM
        self.perturbation = perturbation
        self.samples = []    # list of (path, label)
        self.class_names = []

        if label_file and os.path.exists(label_file):
            # CSV mode
            import csv
            with open(label_file, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    fname, label = row[0], int(row[1])
                    self.samples.append((self.img_dir / fname, label))
        else:
            # Class-per-folder mode
            classes = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
            self.class_names = classes
            for label, cls_name in enumerate(classes):
                cls_dir = self.img_dir / cls_name
                for img_path in sorted(cls_dir.glob("*")):
                    if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # Apply perturbation if specified (for robustness testing)
        if self.perturbation:
            from src.robustness.perturbations import apply_perturbation
            img = apply_perturbation(img, self.perturbation)

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    # Quick sanity check
    ds = LungCancerDataset("data/images")
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        img, label = ds[0]
        print(f"Image shape: {img.shape}, label: {label}")
