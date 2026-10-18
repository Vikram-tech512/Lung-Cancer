"""
backend/pipeline/data/loader_factory.py
=========================================
DataLoaderFactory — builds production-grade PyTorch DataLoaders
from any CT dataset class with consistent settings.

Features
--------
* Automatic num_workers selection (CPU count cap at 8)
* Weighted random sampler for class imbalance
* Pin-memory auto-enabled when CUDA is available
* Prefetch factor tuning
* Collate function with variable-size volume padding

Usage
-----
>>> from backend.pipeline.data import DataLoaderFactory
>>> from backend.datasets.luna16 import LUNA16Dataset

>>> factory = DataLoaderFactory(batch_size=4, output_size=(96, 96, 96))
>>> train_loader = factory.build(LUNA16Dataset, split="train", records=train_records)
>>> val_loader   = factory.build(LUNA16Dataset, split="val",   records=val_records)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Type

import numpy as np

from backend.monitoring import get_logger

log = get_logger(__name__)

# Lazy PyTorch imports
_TORCH_AVAILABLE = False
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class DataLoaderFactory:
    """
    Centralised factory for creating PyTorch DataLoaders.

    Parameters
    ----------
    batch_size  : samples per mini-batch
    output_size : (D, H, W) — volumes will be cropped/padded to this size
    num_workers : override automatic worker count selection
    augment     : whether to apply 3D augmentation for train splits
    """

    def __init__(
        self,
        batch_size: int = 4,
        output_size: tuple = (96, 96, 96),
        num_workers: Optional[int] = None,
        augment: bool = True,
    ) -> None:
        self.batch_size  = batch_size
        self.output_size = output_size
        self.augment     = augment
        self._num_workers = num_workers or min(int(os.cpu_count() or 1), 8)

    # ------------------------------------------------------------------
    # Public factory method
    # ------------------------------------------------------------------
    def build(
        self,
        dataset_cls: "Type[Dataset]",
        split: str,
        records: List[Dict[str, Any]],
        balance_classes: bool = True,
        **dataset_kwargs: Any,
    ) -> "DataLoader":
        """
        Instantiate `dataset_cls(records=records, split=split, ...)` and
        wrap in a DataLoader with the right settings.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DataLoaderFactory.build()")

        from backend.pipeline.data.augmentation import AugmentationPipeline

        aug = AugmentationPipeline(
            mode=split,
            output_size=self.output_size,
        ) if self.augment else None

        dataset: Dataset = dataset_cls(
            records=records,
            augmentation=aug,
            **dataset_kwargs,
        )

        sampler = None
        if split == "train" and balance_classes:
            sampler = self._weighted_sampler(records)

        pin_memory = torch.cuda.is_available()
        prefetch   = 2 if self._num_workers > 0 else None

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and split == "train"),
            num_workers=self._num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch,
            drop_last=(split == "train"),
            collate_fn=self._pad_collate,
        )
        log.info(
            "dataloader_built",
            split=split,
            n_records=len(records),
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            balanced=balance_classes and split == "train",
        )
        return loader

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _weighted_sampler(self, records: List[Dict]) -> "WeightedRandomSampler":
        """
        Build a WeightedRandomSampler that oversamples the minority class.
        Expects each record to have a 'label' key (0 = benign, 1 = malignant).
        """
        labels = np.array([int(r.get("label", 0)) for r in records])
        class_counts = np.bincount(labels)
        # Weight per sample = 1 / class_frequency
        weights = 1.0 / class_counts[labels]
        weights_tensor = torch.from_numpy(weights).float()
        return WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(records),
            replacement=True,
        )

    @staticmethod
    def _pad_collate(batch: list) -> dict:
        """
        Custom collate that stacks variable-size volumes by padding to max dims.
        Each item must be a dict with at least a "volume" key (torch.Tensor).
        """
        if not batch:
            return {}

        # Determine max spatial dims in batch
        volumes = [item["volume"] for item in batch]
        max_shape = tuple(
            max(v.shape[i] for v in volumes) for i in range(volumes[0].dim())
        )

        padded_vols = []
        for v in volumes:
            pads = []
            for i in range(v.dim() - 1, -1, -1):
                diff = max_shape[i] - v.shape[i]
                pads += [0, diff]
            import torch.nn.functional as F
            padded_vols.append(F.pad(v, pads))

        collated = {
            "volume": torch.stack(padded_vols),
        }

        # Pass through other keys (labels, masks, patient_ids, ...)
        for key in batch[0]:
            if key == "volume":
                continue
            vals = [item[key] for item in batch]
            if isinstance(vals[0], torch.Tensor):
                # Try to stack; fall back to list on shape mismatch
                try:
                    collated[key] = torch.stack(vals)
                except Exception:
                    collated[key] = vals
            else:
                collated[key] = vals

        return collated
