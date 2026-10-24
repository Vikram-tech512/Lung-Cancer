"""
backend/training/train_step3.py
================================
Step 3: Full Training on LUNA16 Subsets 0, 1 with Subset 2 Validation.

Architectural & Operational Details:
- Model: ResNet3D18Half (3D ResNet-18 Half-Width, 8.29M parameters, CPU-optimized).
- Input: 32x32x32 voxel crops at 1.0 mm isotropic spacing, HU windowed [-1000, 400] to [0, 1].
- Splits:
    - Train: Subsets 0, 1 (178 scans, 308 positives + 1,540 balanced negatives = 1,848 crops).
    - Validation: Subset 2 (89 scans, 181 positives + 905 balanced negatives = 1,086 crops).
- Training:
    - Batch Size: 16 (optimized for CPU vectorization).
    - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4).
    - Scheduler: CosineAnnealingLR.
    - Loss: Class-weighted CrossEntropyLoss.
    - Augmentations: Random 3D spatial flips & 90-deg axial rotations.
- Guards & Checkpointing:
    - 5-epoch guard: Asserts val_auc > 0.50 after epoch 5.
    - Early stopping: Patience = 4 epochs on validation AUC.
    - Checkpoint: checkpoints_3d/resnet3d_luna16_best.pt (model weights, optimizer, epoch, val_auc).
    - Results: results/training_history_step3.json.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import SimpleITK as sitk
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.models.classification.resnet3d.model import BasicBlock3d


# ─────────────────────────────────────────────────────────────────────────────
#  Model Definition: ResNet3D18Half
# ─────────────────────────────────────────────────────────────────────────────

class ResNet3D18Half(nn.Module):
    """
    Half-width 3D ResNet-18 (8.29M parameters).
    Channels: 32 -> 64 -> 128 -> 256.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout_p: float = 0.3):
        super().__init__()
        self.in_ch = 32
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = self._make_layer(32,  2, stride=1)
        self.layer2 = self._make_layer(64,  2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc      = nn.Linear(256, num_classes)

    def _make_layer(self, out_ch: int, n: int, stride: int) -> nn.Sequential:
        ds = None
        if stride != 1 or self.in_ch != out_ch:
            ds = nn.Sequential(
                nn.Conv3d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers = [BasicBlock3d(self.in_ch, out_ch, stride, ds)]
        self.in_ch = out_ch
        for _ in range(1, n):
            layers.append(BasicBlock3d(out_ch, out_ch))
        return nn.Sequential(*layers)

    def enable_mc_dropout(self):
        """Keep dropout active during inference for MC Dropout uncertainty."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return layer4 feature maps for 3D Grad-CAM."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Crop Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_32_crop(
    image: sitk.Image,
    center_xyz: Tuple[float, float, float],
    crop_size: int = 32,
    target_spacing: float = 1.0,
    window_min: float = -1000.0,
    window_max: float = 400.0,
) -> np.ndarray:
    """Extract a 32x32x32 crop at 1.0mm isotropic spacing with [-1000, 400] HU windowing."""
    target_size = [crop_size, crop_size, crop_size]
    spacing = [target_spacing, target_spacing, target_spacing]
    target_origin = [center_xyz[i] - (crop_size - 1) * target_spacing / 2.0 for i in range(3)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(-1000.0)
    resampler.SetInterpolator(sitk.sitkLinear)

    crop_img = resampler.Execute(image)
    arr = sitk.GetArrayFromImage(crop_img).astype(np.float32)

    clipped = np.clip(arr, window_min, window_max)
    return (clipped - window_min) / (window_max - window_min)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset & Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def augment_crop_3d(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label-preserving 3D data augmentation."""
    out = crop.copy()
    # Random axis flips
    for ax in range(3):
        if rng.random() < 0.5:
            out = np.flip(out, axis=ax)
    # Random 90 deg rotation in axial plane (axes 1, 2)
    k = int(rng.integers(0, 4))
    if k > 0:
        out = np.rot90(out, k=k, axes=(1, 2))
    # Subtle intensity scaling (+- 5%)
    scale = rng.uniform(0.95, 1.05)
    out = np.clip(out * scale, 0.0, 1.0)
    return out.astype(np.float32)


class LUNA16PreloadedDataset(Dataset):
    """In-memory dataset of pre-extracted 32^3 crops for high CPU throughput."""

    def __init__(self, crops: List[np.ndarray], labels: List[int], augment: bool = False, seed: int = 42):
        self.crops = crops
        self.labels = labels
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        crop = self.crops[idx]
        if self.augment:
            rng = np.random.default_rng(self.seed + idx * 10007)
            crop = augment_crop_3d(crop, rng)
        vol_t = torch.from_numpy(crop.copy()).unsqueeze(0).float()
        lbl_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return vol_t, lbl_t


# ─────────────────────────────────────────────────────────────────────────────
#  Preloading Helper
# ─────────────────────────────────────────────────────────────────────────────

def preload_candidates_crops(
    luna_root: Path,
    scans_list: List[str],
    candidates_csv_path: Path,
    neg_ratio: int = 5,
    seed: int = 42,
    desc: str = "dataset",
) -> Tuple[List[np.ndarray], List[int]]:
    scan_set = set(scans_list)
    print(f"[{desc.upper()}] Indexing candidates for {len(scan_set)} scans...")

    # Locate MHD files
    scan_mhd_map = {}
    for subset_dir in luna_root.glob("subset*"):
        for mhd in subset_dir.glob("*.mhd"):
            if mhd.stem in scan_set:
                scan_mhd_map[mhd.stem] = mhd

    pos_cands = []
    neg_cands = []
    with open(candidates_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            suid = r["seriesuid"]
            if suid in scan_set:
                entry = {
                    "seriesuid": suid,
                    "coord_x": float(r["coordX"]),
                    "coord_y": float(r["coordY"]),
                    "coord_z": float(r["coordZ"]),
                    "class": int(r["class"]),
                }
                if entry["class"] == 1:
                    pos_cands.append(entry)
                else:
                    neg_cands.append(entry)

    # Balance negatives
    np.random.seed(seed)
    n_pos = len(pos_cands)
    n_neg = min(len(neg_cands), n_pos * neg_ratio)
    sampled_neg = [neg_cands[i] for i in np.random.choice(len(neg_cands), size=n_neg, replace=False)]
    all_cands = pos_cands + sampled_neg
    np.random.shuffle(all_cands)

    print(f"[{desc.upper()}] Selected {n_pos} positives + {n_neg} negatives = {len(all_cands)} total crops.")

    # Group by scan to extract
    by_scan: Dict[str, List] = {}
    for c in all_cands:
        by_scan.setdefault(c["seriesuid"], []).append(c)

    crops = []
    labels = []
    t0 = time.time()
    for suid, c_list in by_scan.items():
        mhd = scan_mhd_map[suid]
        img = sitk.ReadImage(str(mhd))
        for c in c_list:
            coord = (c["coord_x"], c["coord_y"], c["coord_z"])
            crop_32 = extract_32_crop(img, coord, crop_size=32)
            crops.append(crop_32)
            labels.append(c["class"])

    dt = time.time() - t0
    print(f"[{desc.upper()}] Preloaded {len(crops)} crops in {dt:.1f}s ({len(crops)/dt:.1f} crops/sec).")
    return crops, labels


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation Routine
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    for volumes, labels in loader:
        logits = model(volumes)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        total_loss += loss.item() * volumes.size(0)
        preds = logits.argmax(dim=-1)
        all_targets.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    n = max(1, len(all_targets))
    mean_loss = total_loss / n
    acc = accuracy_score(all_targets, all_preds)

    # Sensitivity & Specificity
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auc = float(roc_auc_score(all_targets, all_probs))
    except Exception:
        auc = 0.50

    return {
        "val_loss": round(mean_loss, 4),
        "val_accuracy": round(acc, 4),
        "val_auc": round(auc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Loop (Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def train_luna16_step3(
    luna_root: Path = Path(r"C:\LUNA16_DATA"),
    manifest_path: Path = Path("results/split_manifest.json"),
    max_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 4,
    neg_ratio: int = 5,
    seed: int = 42,
) -> Dict:
    print("=================================================================")
    print(" Step 3: Full Training on LUNA16 Subsets 0, 1 (Val: Subset 2)")
    print("=================================================================")

    # 1. Load Split Manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_scans = manifest["scans"]["train_scans"]  # 178 scans
    val_scans = manifest["scans"]["val_scans"]      # 89 scans
    cand_csv = luna_root / "candidates_V2.csv"

    print(f"[SPLIT] Train: {len(train_scans)} scans (subsets 0, 1) | Val: {len(val_scans)} scans (subset 2)")

    # 2. Preload Datasets into Memory
    t0_preload = time.time()
    train_crops, train_labels = preload_candidates_crops(
        luna_root, train_scans, cand_csv, neg_ratio=neg_ratio, seed=seed, desc="train"
    )
    val_crops, val_labels = preload_candidates_crops(
        luna_root, val_scans, cand_csv, neg_ratio=neg_ratio, seed=seed + 1, desc="val"
    )
    print(f"[PRELOAD DONE] Total preloading elapsed time: {time.time() - t0_preload:.1f}s")

    train_ds = LUNA16PreloadedDataset(train_crops, train_labels, augment=True, seed=seed)
    val_ds = LUNA16PreloadedDataset(val_crops, val_labels, augment=False, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer & Scheduler
    model = ResNet3D18Half(in_channels=1, num_classes=2, dropout_p=0.3)
    
    # Class weights for loss weighting
    counts = np.bincount(train_labels, minlength=2)
    weights = torch.tensor([len(train_labels) / (2.0 * counts[0]), len(train_labels) / (2.0 * counts[1])], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    checkpoints_dir = Path("checkpoints_3d")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoints_dir / "resnet3d_luna16_best.pt"
    final_ckpt_path = checkpoints_dir / "resnet3d_luna16_final.pt"

    # 4. Training Loop
    history = []
    best_val_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    total_training_time = 0.0

    print(f"\n--- Starting Training ({max_epochs} epochs max, early stopping patience={patience}) ---")

    for epoch in range(1, max_epochs + 1):
        t0_ep = time.time()
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for volumes, targets in train_loader:
            optimizer.zero_grad()
            logits = model(volumes)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * volumes.size(0)
            train_preds.extend(logits.argmax(dim=-1).tolist())
            train_targets.extend(targets.tolist())

        scheduler.step()
        dt_ep = time.time() - t0_ep
        total_training_time += dt_ep

        train_acc = accuracy_score(train_targets, train_preds)
        mean_train_loss = train_loss / len(train_targets)

        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion)
        val_auc = val_metrics["val_auc"]

        ep_record = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["val_accuracy"],
            "val_auc": val_auc,
            "sensitivity": val_metrics["sensitivity"],
            "specificity": val_metrics["specificity"],
            "epoch_sec": round(dt_ep, 1),
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        history.append(ep_record)

        is_best = val_auc > best_val_auc
        marker = " [*BEST]" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{max_epochs:2d} ({dt_ep:4.1f}s) | "
            f"Train Loss: {mean_train_loss:.4f} Acc: {train_acc*100:4.1f}% | "
            f"Val Loss: {val_metrics['val_loss']:.4f} Acc: {val_metrics['val_accuracy']*100:4.1f}% "
            f"AUC: {val_auc:.4f} (Sens: {val_metrics['sensitivity']:.3f}, Spec: {val_metrics['specificity']:.3f}){marker}"
        )

        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_metrics": val_metrics,
                "architecture": "ResNet3D18Half",
                "split_hash": manifest["split_hash_sha256"],
                "crop_size": 32,
            }, best_ckpt_path)
        else:
            epochs_no_improve += 1

        # 5-Epoch Guard: Assert val_auc > 0.50 after epoch 5
        if epoch == 5:
            assert best_val_auc > 0.50, (
                f"GUARD FAILED: Best Validation AUC after 5 epochs is {best_val_auc:.4f} <= 0.50. "
                "Model is failing to learn discriminating features on LUNA16 candidates."
            )
            print(f"[GUARD PASSED] 5-Epoch Guard: Best Validation AUC is {best_val_auc:.4f} > 0.50.")

        # Early Stopping
        if epochs_no_improve >= patience and epoch >= 5:
            print(f"[EARLY STOP] Validation AUC did not improve for {patience} consecutive epochs. Stopping at epoch {epoch}.")
            break

    # Save final checkpoint
    torch.save({
        "epoch": len(history),
        "model_state_dict": model.state_dict(),
        "final_history": history,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
    }, final_ckpt_path)

    # 5. Save Training History Manifest
    results_manifest = {
        "step": 3,
        "title": "LUNA16 3D ResNet-18 Half Training Run",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "architecture": "ResNet3D18Half",
        "parameters": sum(p.numel() for p in model.parameters()),
        "split_hash": manifest["split_hash_sha256"],
        "training_config": {
            "max_epochs": max_epochs,
            "actual_epochs": len(history),
            "batch_size": batch_size,
            "initial_lr": lr,
            "neg_ratio": neg_ratio,
            "train_samples": len(train_crops),
            "val_samples": len(val_crops),
            "total_training_sec": round(total_training_time, 1),
            "total_training_min": round(total_training_time / 60.0, 2),
        },
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "history": history,
        "checkpoint_paths": {
            "best": str(best_ckpt_path),
            "final": str(final_ckpt_path),
        },
        "five_epoch_guard_passed": best_val_auc > 0.50,
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "training_history_step3.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_manifest, f, indent=2)

    print("\n=================================================================")
    print(f" Step 3 Training Complete! Best Epoch: {best_epoch} (Val AUC: {best_val_auc:.4f})")
    print(f" Best Checkpoint: {best_ckpt_path}")
    print(f" Training Manifest: {out_file}")
    print("=================================================================")
    return results_manifest


if __name__ == "__main__":
    train_luna16_step3()
