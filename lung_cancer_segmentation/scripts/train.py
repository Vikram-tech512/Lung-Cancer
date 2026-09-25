# scripts/train.py
"""Training script for lung cancer classification.
Supports CNN, ViT, and Hybrid CNN-ViT architectures.
Automatically uses GPU when available; otherwise CPU.
"""
import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset.lung_dataset import LungCancerDataset, TRAIN_TRANSFORM, DEFAULT_TRANSFORM


def get_model(arch: str, num_classes: int):
    """Instantiate model by architecture name."""
    if arch == "cnn":
        from src.model.cnn import LungCancerCNN
        return LungCancerCNN(num_classes=num_classes)
    elif arch == "vit":
        from src.model.vit import ViT
        return ViT(num_classes=num_classes, depth=6, heads=8, embed_dim=256, mlp_dim=512, dropout=0.1)
    elif arch == "hybrid":
        from src.model.hybrid import HybridViT
        return HybridViT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Use cnn, vit, or hybrid.")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train lung cancer classifier")
    parser.add_argument("--data-dir", required=True, help="Root data directory (class-per-folder)")
    parser.add_argument("--arch", choices=["cnn", "vit", "hybrid"], default="hybrid")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output", default="checkpoints/best_model.pt")
    parser.add_argument("--log", default="results/training_log.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    full_ds = LungCancerDataset(args.data_dir, transform=TRAIN_TRANSFORM)
    num_classes = len(set(label for _, label in full_ds.samples))
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    # Override transform for val set
    val_ds.dataset.transform = DEFAULT_TRANSFORM

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Classes: {num_classes}, Train: {train_size}, Val: {val_size}")

    # Model
    model = get_model(args.arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Paths
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(elapsed, 1),
        }
        history.append(log)
        print(f"Epoch {epoch}/{args.epochs} — "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # Save training log
    with open(args.log, "w") as f:
        json.dump({"architecture": args.arch, "best_val_acc": best_val_acc, "history": history}, f, indent=2)
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {args.output}")
    print(f"Log: {args.log}")


if __name__ == "__main__":
    main()
