# src/robustness/analyzer.py
"""Robustness analyzer – evaluates model accuracy under every registered perturbation
and computes the Robustness Degradation (RD) score defined in the synopsis as:
    RD = Performance_original − Performance_perturbed
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.robustness.perturbations import PERTURBATION_REGISTRY, apply_perturbation
from src.dataset.lung_dataset import LungCancerDataset


def evaluate_accuracy(model, loader, device="cpu"):
    """Compute top-1 accuracy on a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def robustness_analysis(
    model,
    img_dir: str,
    label_file: str,
    perturbations: Optional[List[str]] = None,
    batch_size: int = 16,
    device: str = "cpu",
    output_path: Optional[str] = None,
) -> Dict:
    """Run full robustness analysis.

    Returns:
        dict mapping perturbation name → {accuracy, degradation}
        plus an "original" entry.
    """
    if perturbations is None:
        perturbations = list(PERTURBATION_REGISTRY.keys())

    # Original accuracy
    ds_orig = LungCancerDataset(img_dir, label_file, perturbation=None)
    loader_orig = DataLoader(ds_orig, batch_size=batch_size, shuffle=False)
    acc_orig = evaluate_accuracy(model, loader_orig, device)

    results = {
        "original": {"accuracy": round(acc_orig, 4), "degradation": 0.0}
    }

    for pname in tqdm(perturbations, desc="Robustness sweep"):
        ds_pert = LungCancerDataset(img_dir, label_file, perturbation=pname)
        loader_pert = DataLoader(ds_pert, batch_size=batch_size, shuffle=False)
        acc_pert = evaluate_accuracy(model, loader_pert, device)
        rd = round(acc_orig - acc_pert, 4)
        results[pname] = {"accuracy": round(acc_pert, 4), "degradation": rd}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Robustness results saved to {output_path}")

    return results
