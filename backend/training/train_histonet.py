"""
backend/training/train_histonet.py
====================================
Training pipeline for histopathological lung cancer image classification.

Trains a 2D transfer-learning model (ResNet-50 / EfficientNet-B2 / DenseNet-121)
on the LC25000 dataset (Kaggle: andrewmvd/lung-and-colon-cancer-histopathological-images).

Features:
  - Pretrained ImageNet backbone + custom dropout head
  - Two-phase training: head warmup → full fine-tune with differential LR
  - Cosine annealing LR schedule with warm restarts
  - Class-weighted cross-entropy loss
  - Early stopping on validation AUC
  - MLflow experiment tracking
  - Checkpoint saving (best val AUC + last epoch)
  - Confusion matrix + per-class metrics at test time

Usage::
    # Train resnet50 on lung classes (3-class), GPU:
    python -m backend.training.train_histonet \\
        --model resnet50 --mode lung --epochs 30 --batch-size 32

    # Train efficientnet on all 5 classes, CPU dry-run:
    python -m backend.training.train_histonet \\
        --model efficientnet --mode all --epochs 2 --dry-run

    # Resume from checkpoint:
    python -m backend.training.train_histonet \\
        --model resnet50 --resume checkpoints/histonet_resnet50_best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.amp import autocast, GradScaler
except ImportError as e:
    raise SystemExit(
        "PyTorch not found. Install requirements: pip install -r backend/requirements.txt"
    ) from e

from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)

from backend.datasets.histopathology import get_dataloaders, KAGGLE_DATASET_ROOT
from backend.models.classification.histonet.model import build_histonet


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU (training will be slow — GPU recommended)")
    return dev


def _mixup_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp augmentation for better generalisation."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def _compute_auc(all_labels: np.ndarray, all_probs: np.ndarray, num_classes: int) -> float:
    """Macro-averaged one-vs-rest AUC (handles both binary and multi-class)."""
    try:
        if num_classes == 2:
            return float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            return float(roc_auc_score(
                all_labels, all_probs,
                multi_class="ovr", average="macro"
            ))
    except ValueError:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Training core
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    use_mixup: bool = True,
    mixup_alpha: float = 0.3,
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup:
            imgs, labels_a, labels_b, lam = _mixup_batch(imgs, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(imgs)
            if use_mixup:
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy() if not use_mixup else labels_a.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return {
        "loss":     total_loss / len(loader),
        "accuracy": acc,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=-1)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    auc = _compute_auc(all_labels, all_probs, num_classes)

    return {
        "loss":     total_loss / len(loader),
        "accuracy": acc,
        "auc":      auc,
        "_labels":  all_labels,
        "_probs":   all_probs,
        "_preds":   all_preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main trainer
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device    = _get_device()
    num_classes = 3 if args.mode == "lung" else 5
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[Data] Loading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        dataset_root  = args.data_root or None,
        mode          = args.mode,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        img_size      = args.img_size,
        balanced_sampling = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[Model] Building {args.model} (pretrained={not args.no_pretrain})...")
    model = build_histonet(
        variant         = args.model,
        num_classes     = num_classes,
        pretrained      = not args.no_pretrain,
        freeze_backbone = True,   # Phase 1: head-only warmup
        dropout         = args.dropout,
    )

    if args.resume:
        print(f"[Model] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Class-weighted CE to handle any imbalance in splits
    class_weights = train_loader.dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    # Phase 1 — warmup: train only the head for warmup_epochs
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(head_params, lr=args.lr * 10, weight_decay=1e-4)
    scheduler  = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    scaler     = GradScaler() if device.type == "cuda" else None

    best_auc     = 0.0
    patience_cnt = 0
    history      = []
    best_ckpt    = output_dir / f"histonet_{args.model}_best.pt"
    last_ckpt    = output_dir / f"histonet_{args.model}_last.pt"

    print(f"\n{'='*65}")
    print(f"  Lung Cancer Histopathology Training — {args.model.upper()}")
    print(f"  Mode: {args.mode} | Classes: {class_names}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | Device: {device}")
    print(f"{'='*65}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Phase 2: unfreeze backbone after warmup ───────────────────────────
        if epoch == args.warmup_epochs + 1:
            print(f"[Epoch {epoch}] Unfreezing backbone for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # Differential LR: backbone gets 10× smaller LR than head
            all_params = [
                {"params": [p for n, p in model.named_parameters()
                            if "fc" not in n and "classifier" not in n],
                 "lr": args.lr},
                {"params": [p for n, p in model.named_parameters()
                            if "fc" in n or "classifier" in n],
                 "lr": args.lr * 10},
            ]
            optimizer = AdamW(all_params, weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_mixup=args.use_mixup, mixup_alpha=0.3,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        elapsed = time.time() - t0
        row = {
            "epoch":      epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc":  round(train_metrics["accuracy"], 4),
            "val_loss":   round(val_metrics["loss"], 4),
            "val_acc":    round(val_metrics["accuracy"], 4),
            "val_auc":    round(val_metrics["auc"], 4),
            "lr":         round(optimizer.param_groups[0]["lr"], 6),
            "elapsed_s":  round(elapsed, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"TrainLoss={row['train_loss']:.4f} TrainAcc={row['train_acc']:.3f} | "
            f"ValLoss={row['val_loss']:.4f} ValAcc={row['val_acc']:.3f} ValAUC={row['val_auc']:.4f} | "
            f"LR={row['lr']:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt_payload = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_auc":     val_metrics["auc"],
            "class_names": class_names,
            "args":        vars(args),
        }

        torch.save(ckpt_payload, last_ckpt)

        if val_metrics["auc"] > best_auc:
            best_auc     = val_metrics["auc"]
            patience_cnt = 0
            torch.save(ckpt_payload, best_ckpt)
            print(f"  [NEW BEST] AUC: {best_auc:.4f} -- checkpoint saved.")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping.")
                break

        # Dry-run exits after 1 epoch
        if args.dry_run and epoch >= 1:
            print("[Dry Run] 1 epoch complete — exiting early.")
            break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print(f"\n[Test] Loading best checkpoint: {best_ckpt}")
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model_state"])

    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    print(f"\n{'='*65}")
    print(f"  TEST RESULTS")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  AUC      : {test_metrics['auc']:.4f}")
    print(f"{'='*65}")

    print("\nClassification Report:")
    print(classification_report(
        test_metrics["_labels"],
        test_metrics["_preds"],
        target_names=class_names,
        digits=4,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(test_metrics["_labels"], test_metrics["_preds"]))

    # ── Save history & test results ───────────────────────────────────────────
    results = {
        "best_val_auc":  best_auc,
        "test_accuracy": test_metrics["accuracy"],
        "test_auc":      test_metrics["auc"],
        "class_names":   class_names,
        "history":       history,
        "args":          vars(args),
    }
    results_path = output_dir / f"histonet_{args.model}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {results_path}")
    print(f"[Checkpoints] Best: {best_ckpt} | Last: {last_ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HistoNet on LC25000 lung histopathology")
    p.add_argument("--model",        type=str,   default="resnet50",
                   choices=["resnet50", "efficientnet", "densenet"],
                   help="Backbone architecture")
    p.add_argument("--mode",         type=str,   default="lung",
                   choices=["lung", "all"],
                   help="'lung' = 3 classes, 'all' = 5 classes")
    p.add_argument("--data-root",    type=str,   default=None,
                   help="Path to lung_colon_image_set/ directory (default: kagglehub cache)")
    p.add_argument("--output-dir",   type=str,   default="checkpoints",
                   help="Directory to save checkpoints and results")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--warmup-epochs",type=int,   default=3,
                   help="Epochs to train head-only before unfreezing backbone")
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--img-size",     type=int,   default=224)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--patience",     type=int,   default=7,
                   help="Early stopping patience (epochs)")
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--no-pretrain",  action="store_true",
                   help="Do not load ImageNet pretrained weights")
    p.add_argument("--use-mixup",    action="store_true", default=True,
                   help="Enable MixUp augmentation (default: True)")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint .pt to resume from")
    p.add_argument("--dry-run",      action="store_true",
                   help="Run 1 batch/epoch to verify pipeline (no real training)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
