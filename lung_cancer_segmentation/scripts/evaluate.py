# scripts/evaluate.py
"""Comprehensive evaluation script for lung cancer classification.
Computes: Accuracy, Precision, Recall, F1, Sensitivity, Specificity, ROC-AUC, Confusion Matrix.
Supports all three architectures: CNN, ViT, Hybrid.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.dataset.lung_dataset import LungCancerDataset, DEFAULT_TRANSFORM


def get_model(arch: str, num_classes: int):
    if arch == "cnn":
        from src.model.cnn import LungCancerCNN
        return LungCancerCNN(num_classes=num_classes)
    elif arch == "vit":
        from src.model.vit import ViT
        return ViT(num_classes=num_classes, depth=6, heads=8, embed_dim=256, mlp_dim=512)
    elif arch == "hybrid":
        from src.model.hybrid import HybridViT
        return HybridViT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_specificity(cm):
    """Compute per-class specificity from confusion matrix."""
    specificity = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(spec)
    return specificity


def save_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True Label", xlabel="Predicted Label",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # Write numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate lung cancer classifier")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", choices=["cnn", "vit", "hybrid"], default="hybrid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LungCancerDataset(args.data_dir, transform=DEFAULT_TRANSFORM)
    num_classes = len(set(label for _, label in ds.samples))
    class_names = ds.class_names or [f"class_{i}" for i in range(num_classes)]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.arch, num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    y_true, y_pred, y_probs = predict_all(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    spec = compute_specificity(cm)

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    results = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity_per_class": [round(s, 4) for s in spec],
        "roc_auc": round(auc, 4) if auc else None,
        "confusion_matrix": cm.tolist(),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{args.arch}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save confusion matrix figure
    cm_path = os.path.join(args.output_dir, f"{args.arch}_confusion_matrix.png")
    save_confusion_matrix(cm, class_names, cm_path)

    # Print report
    print(f"\n{'='*50}")
    print(f"Architecture: {args.arch}")
    print(f"{'='*50}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall/Sens: {rec:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}" if auc else "ROC-AUC:     N/A")
    print(f"\nConfusion Matrix saved to: {cm_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


if __name__ == "__main__":
    main()
