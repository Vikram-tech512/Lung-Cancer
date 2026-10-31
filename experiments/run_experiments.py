"""Reproduce candidate-classification metrics on a verified LUNA16 split.

This runner evaluates nodule-candidate versus non-nodule-candidate labels. It
does not evaluate scan-level detection, malignancy, diagnosis, or clinical benefit.
It refuses synthetic data, missing/incompatible checkpoints, and split mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score

from backend.calibration.temperature_scaling import TemperatureScaler
from backend.datasets.luna16 import LUNA16Loader, get_luna16_loaders, luna16_split_hash
from backend.models.classification.resnet3d.model import ResNet3D18Half


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_bootstrap(labels, probs, patient_ids, seed, repeats):
    patients = np.unique(patient_ids)
    rows_by_patient = [np.flatnonzero(patient_ids == patient) for patient in patients]
    rng = np.random.default_rng(seed)
    auc_values, brier_values = [], []
    for _ in range(repeats):
        chosen = rng.integers(0, len(patients), size=len(patients))
        rows = np.concatenate([rows_by_patient[index] for index in chosen])
        sample_labels = labels[rows]
        if len(np.unique(sample_labels)) != 2:
            continue
        sample_probs = probs[rows]
        auc_values.append(float(roc_auc_score(sample_labels, sample_probs)))
        brier_values.append(float(brier_score_loss(sample_labels, sample_probs)))

    def interval(values):
        if not values:
            return None
        low, high = np.percentile(values, [2.5, 97.5])
        return [float(low), float(high)]

    return {"auc_95_ci": interval(auc_values), "brier_95_ci": interval(brier_values)}


def expected_calibration_error(labels, probs, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        in_bin = (probs > edges[index]) & (probs <= edges[index + 1])
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(labels[in_bin].mean()) - float(probs[in_bin].mean()))
    return ece


def collect_logits(model, loader, device):
    all_logits, all_labels, all_patients = [], [], []
    with torch.inference_mode():
        for volumes, labels, metadata in loader:
            all_logits.append(model(volumes.to(device)).cpu())
            all_labels.extend(int(label) for label in labels)
            all_patients.extend(str(item.get("patient_id") or item["series_uid"]) for item in metadata)
    if not all_logits:
        raise RuntimeError("The requested split contains no candidates.")
    return torch.cat(all_logits), np.asarray(all_labels, dtype=np.int64), np.asarray(all_patients)


def run(args):
    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not LUNA16Loader(data_root).is_available():
        raise RuntimeError("Real LUNA16 files are unavailable; no result was written.")
    if not checkpoint_path.is_file():
        raise RuntimeError("Checkpoint not found: " + str(checkpoint_path))
    if output_path.exists() and not args.overwrite:
        raise RuntimeError("Refusing to overwrite existing result: " + str(output_path))

    train_loader, val_loader, test_loader = get_luna16_loaders(
        luna_root=data_root, cache_dir=args.cache_dir, crop_size=32,
        batch_size=args.batch_size, num_workers=0, max_neg_per_pos=args.max_neg_per_pos,
        val_frac=0.15, test_frac=0.15, seed=args.seed, split_mode=args.split_mode,
        balanced_sampling=False,
    )
    train_ds, val_ds, test_ds = train_loader.dataset, val_loader.dataset, test_loader.dataset
    split_hash = luna16_split_hash(train_ds.candidates, val_ds.candidates, test_ds.candidates)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = {
        "dataset": "LUNA16", "synthetic": False, "split_mode": args.split_mode,
        "split_hash_sha256": split_hash, "crop_size": 32,
    }
    mismatches = {key: [checkpoint.get(key), value] for key, value in expected.items()
                  if checkpoint.get(key) != value}
    if mismatches:
        raise RuntimeError("Checkpoint provenance mismatch (checkpoint, requested): " + repr(mismatches))
    if checkpoint.get("architecture") not in {"resnet3d18_half", "ResNet3D18Half"}:
        raise RuntimeError("Only a ResNet3D18Half checkpoint is supported.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError("Checkpoint has no model_state_dict.")

    device = torch.device(args.device)
    model = ResNet3D18Half(in_channels=1, num_classes=2, dropout_p=0.3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    val_logits, val_labels, _ = collect_logits(model, val_loader, device)
    if len(np.unique(val_labels)) != 2:
        raise RuntimeError("Validation split must contain both candidate classes.")
    scaler = TemperatureScaler()
    temperature = float(scaler.fit(val_logits, torch.from_numpy(val_labels), split_name="validation"))

    test_logits, test_labels, patient_ids = collect_logits(model, test_loader, device)
    if len(np.unique(test_labels)) != 2:
        raise RuntimeError("Test split must contain both candidate classes.")
    probs = scaler.calibrate(test_logits)[:, 1].detach().cpu().numpy()
    predictions = (probs >= 0.5).astype(np.int64)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(test_labels, predictions, labels=[0, 1]).ravel())
    metrics = {
        "task": "LUNA16 nodule-candidate vs non-nodule-candidate classification",
        "n_test_candidates": int(len(test_labels)),
        "n_test_series": int(len(np.unique(patient_ids))),
        "n_positive_candidates": int(np.sum(test_labels == 1)),
        "n_negative_candidates": int(np.sum(test_labels == 0)),
        "temperature_fitted_on": "validation",
        "temperature": temperature,
        "threshold": 0.5,
        "roc_auc": float(roc_auc_score(test_labels, probs)),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "brier_score": float(brier_score_loss(test_labels, probs)),
        "expected_calibration_error_10_bins": expected_calibration_error(test_labels, probs),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "patient_cluster_bootstrap_95_ci": cluster_bootstrap(test_labels, probs, patient_ids, args.seed, args.bootstrap_repeats),
    }
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if revision.returncode:
        raise RuntimeError("Could not resolve current Git commit.")
    result = {
        "schema_version": 1, "status": "completed", "synthetic": False,
        "dataset": "LUNA16",
        "task_scope": "candidate classification only; not scan-level detection, malignancy, or diagnosis",
        "split_mode": args.split_mode, "split_hash_sha256": split_hash,
        "git_commit_hash": revision.stdout.strip(), "seed": args.seed,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                       "architecture": checkpoint["architecture"]},
        "split_counts": {"train_candidates": len(train_ds), "validation_candidates": len(val_ds),
                         "test_candidates": len(test_ds)},
        "metrics": metrics,
        "unavailable_endpoints": {
            "scan_level_froc": "Not available: this runner evaluates candidate classification, not a scan-level detector.",
            "malignancy": "Not available: candidate labels are not pathology-confirmed malignancy labels.",
            "reader_benefit": "Not available: no reader study has been conducted.",
            "subgroup_analysis": "Not available: no prespecified subgroup analysis is implemented.",
            "xai_faithfulness": "Not available: attribution faithfulness has not been evaluated.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a verified LUNA16 candidate-classification checkpoint.")
    parser.add_argument("--data-root", default="data/LUNA16")
    parser.add_argument("--cache-dir", default="data/luna16_crops")
    parser.add_argument("--checkpoint", default="checkpoints_3d/resnet3d18_half_luna16_best.pt")
    parser.add_argument("--output", default="results/reproduced_candidate_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-neg-per-pos", type=int, default=4)
    parser.add_argument("--split-mode", choices=["subset", "random"], default="subset")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print("Evaluation completed. Scope: candidate classification only.")
    print("Dataset: " + result["dataset"] + "; split hash: " + result["split_hash_sha256"])
    print("Result file: " + args.output)


if __name__ == "__main__":
    main()
