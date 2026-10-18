"""
backend/datasets/histopathology.py
====================================
PyTorch Dataset for the LC25000 Lung & Colon Cancer Histopathological Image Dataset
(Kaggle: andrewmvd/lung-and-colon-cancer-histopathological-images).

Classes (lung only — default):
    0 → lung_aca   (Lung Adenocarcinoma)
    1 → lung_n     (Lung Normal)
    2 → lung_scc   (Lung Squamous Cell Carcinoma)

All 5 classes (lung + colon):
    0 → colon_aca  (Colon Adenocarcinoma)
    1 → colon_n    (Colon Normal)
    2 → lung_aca
    3 → lung_n
    4 → lung_scc

Usage::
    from backend.datasets.histopathology import HistopathologyDataset, get_dataloaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        dataset_root=r"C:/Users/ashok/.cache/kagglehub/datasets/andrewmvd/
                       lung-and-colon-cancer-histopathological-images/versions/1",
        mode="lung"   # or "all"
    )
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

KAGGLE_DATASET_ROOT = Path(
    r"C:\Users\ashok\.cache\kagglehub\datasets"
    r"\andrewmvd\lung-and-colon-cancer-histopathological-images\versions\1"
    r"\lung_colon_image_set"
)

# Sub-folder structure inside the dataset root
LUNG_CLASSES: Dict[str, int] = {
    "lung_aca": 0,
    "lung_n":   1,
    "lung_scc": 2,
}

ALL_CLASSES: Dict[str, int] = {
    "colon_aca": 0,
    "colon_n":   1,
    "lung_aca":  2,
    "lung_n":    3,
    "lung_scc":  4,
}

# Sub-folder names → containing directory
_CLASS_TO_SUBDIR: Dict[str, str] = {
    "lung_aca":  "lung_image_sets",
    "lung_n":    "lung_image_sets",
    "lung_scc":  "lung_image_sets",
    "colon_aca": "colon_image_sets",
    "colon_n":   "colon_image_sets",
}

IMG_EXTENSIONS = {".jpeg", ".jpg", ".png", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def _train_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats (RGB)
                    std=[0.229, 0.224, 0.225]),
    ])


def _val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class HistopathologyDataset(Dataset):
    """
    Histopathological image dataset (LC25000).

    Args:
        samples:    List of (image_path, label_int) tuples.
        transform:  torchvision transform pipeline.
        class_names: Ordered list of class name strings.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[T.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.class_names = class_names or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @classmethod
    def from_root(
        cls,
        dataset_root: Path,
        class_map: Dict[str, int],
        split: str = "train",
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        seed: int = 42,
        img_size: int = 224,
    ) -> "HistopathologyDataset":
        """
        Build a split-specific dataset from the raw folder layout.

        Args:
            dataset_root:  Path to `lung_colon_image_set/`
            class_map:     Dict mapping subfolder name → integer label
            split:         "train" | "val" | "test"
            val_frac:      Fraction of images for validation
            test_frac:     Fraction of images for test
            seed:          Random seed for reproducible splits
            img_size:      Resize target (pixels)
        """
        rng = random.Random(seed)
        all_samples: List[Tuple[Path, int]] = []

        for class_name, label in class_map.items():
            subdir = _CLASS_TO_SUBDIR[class_name]
            class_dir = dataset_root / subdir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}\n"
                    f"Check that dataset_root points to the correct path."
                )
            paths = sorted([
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMG_EXTENSIONS
            ])
            rng.shuffle(paths)
            n = len(paths)
            n_val  = int(n * val_frac)
            n_test = int(n * test_frac)
            n_train = n - n_val - n_test

            if split == "train":
                chosen = paths[:n_train]
            elif split == "val":
                chosen = paths[n_train:n_train + n_val]
            else:  # test
                chosen = paths[n_train + n_val:]

            all_samples.extend([(p, label) for p in chosen])

        transform = (
            _train_transforms(img_size)
            if split == "train"
            else _val_transforms(img_size)
        )

        class_names = sorted(class_map, key=class_map.get)
        return cls(all_samples, transform=transform, class_names=class_names)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for loss weighting on imbalanced splits."""
        labels = [lbl for _, lbl in self.samples]
        counts = np.bincount(labels, minlength=len(self.class_names)).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def sampler_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler (balanced mini-batches)."""
        cw = self.class_weights().numpy()
        return [float(cw[lbl]) for _, lbl in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    dataset_root: Optional[str | Path] = None,
    mode: str = "lung",
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
    balanced_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Returns (train_loader, val_loader, test_loader, class_names).

    Args:
        dataset_root:       Path to `lung_colon_image_set/` directory.
                            Defaults to the kagglehub cache path.
        mode:               "lung" (3-class) or "all" (5-class).
        batch_size:         Mini-batch size.
        num_workers:        DataLoader worker processes.
        img_size:           Input resolution (square).
        val_frac:           Fraction for validation.
        test_frac:          Fraction for test.
        seed:               Random seed.
        balanced_sampling:  Use WeightedRandomSampler for balanced training batches.
    """
    root = Path(dataset_root) if dataset_root else KAGGLE_DATASET_ROOT
    class_map = LUNG_CLASSES if mode == "lung" else ALL_CLASSES

    kwargs = dict(
        dataset_root=root,
        class_map=class_map,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        img_size=img_size,
    )

    train_ds = HistopathologyDataset.from_root(split="train", **kwargs)
    val_ds   = HistopathologyDataset.from_root(split="val",   **kwargs)
    test_ds  = HistopathologyDataset.from_root(split="test",  **kwargs)

    # Balanced sampler so rare classes get equal representation in each batch
    if balanced_sampling:
        sample_weights = train_ds.sampler_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[HistopathologyDataset] mode={mode} | classes={train_ds.class_names}")
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    return train_loader, val_loader, test_loader, train_ds.class_names
