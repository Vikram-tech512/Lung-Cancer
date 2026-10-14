"""
backend/training/train_classifier.py
=====================================
Real training entry point for the 3D nodule classifiers. This is the file
that turns backend/models/classification/* from architecture definitions
with no trained weights into models with real, checkpointed, calibrated,
robustness-tested weights.

THIS SCRIPT NEEDS A GPU-CAPABLE MACHINE WITH TORCH INSTALLED — it will not
run inside a network/compute-restricted sandbox. See the module docstring
of backend/datasets/lidc_idri.py for how to obtain LIDC-IDRI and configure
pylidc first.

Usage (real training)::
    python -m backend.training.train_classifier \\
        --model resnet3d --data-root /data/LIDC-IDRI \\
        --epochs 50 --batch-size 4 --output-dir checkpoints_3d

Usage (pipeline dry run — SYNTHETIC data, proves the code runs, proves
NOTHING about real accuracy, and refuses to write a checkpoint that looks
like a real one)::
    python -m backend.training.train_classifier --model resnet3d --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise SystemExit(
        "torch is required to run training. This is expected to fail inside "
        "a sandboxed/no-GPU environment — run this on your own machine or a "
        "cloud GPU instance with `pip install -r backend/requirements.txt`."
    ) from e

from sklearn.metrics import roc_auc_score

from backend.datasets.lidc_idri import LIDCIDRILoader, LIDCIDRIVolumeDataset, patient_level_split
from backend.datasets.synthetic_volumes import make_synthetic_cohort
from backend.preprocessing.ct_pipeline import CTPipeline
from backend.calibration.temperature_scaling import TemperatureScaler, evaluate_calibration
from backend.robustness.ct_perturbations import CTPerturbationSuite
from backend.experiments.mlflow_tracker import ExperimentTracker
from backend.experiments.model_registry import ModelRegistry, ModelVersionInfo

from backend.models.classification.resnet3d.model import ResNet3D50
from backend.models.classification.densenet3d.model import DenseNet3D121
from backend.models.classification.swin3d.model import SwinTransformer3D


MODEL_BUILDERS = {
    "resnet3d": lambda: ResNet3D50(in_channels=1, num_classes=2),
    "densenet3d": lambda: DenseNet3D121(in_channels=1, num_classes=2),
    "swin3d": lambda: SwinTransformer3D(in_channels=1, num_classes=2),  # requires 64^3 input
}


class _SyntheticTorchDataset(Dataset):
    """Wraps SyntheticNodule objects (numpy-only) as a torch Dataset, using
    the SAME CTPipeline windowing real data goes through, so the dry run
    actually exercises the real preprocessing code path.
    """

    def __init__(self, nodules, crop_size: Tuple[int, int, int] = (64, 64, 64)):
        self.nodules = nodules
        self.pipeline = CTPipeline(target_spacing=(1.0, 1.0, 1.0))
        self.crop_size = crop_size

    def __len__(self):
        return len(self.nodules)

    def __getitem__(self, idx):
        n = self.nodules[idx]
        crop = self.pipeline.extract_nodule_crop(n.volume, n.centroid_zyx, self.crop_size,
                                                   pad_value=-1000.0)
        crop = self.pipeline.apply_window(crop)
        volume_t = torch.from_numpy(crop).unsqueeze(0).float()
        label_t = torch.tensor(n.binary_label, dtype=torch.long)
        return volume_t, label_t, {"patient_id": n.patient_id, "synthetic": True}


def build_datasets(args) -> Tuple[Dataset, Dataset, Dataset, List[str], List[str], List[str]]:
    crop_size = (args.crop_size, args.crop_size, args.crop_size)

    if args.dry_run:
        print("!! --dry-run: using SYNTHETIC data. This proves the pipeline "
              "runs end-to-end. It proves NOTHING about real-world accuracy. !!")
        cohort = make_synthetic_cohort(n_patients=max(24, args.batch_size * 6), seed=args.seed)
        patient_ids = [n.patient_id for n in cohort]
        train_ids, val_ids, test_ids = patient_level_split(
            patient_ids, args.val_frac, args.test_frac, seed=args.seed)
        by_patient = {n.patient_id: n for n in cohort}
        train_ds = _SyntheticTorchDataset([by_patient[p] for p in train_ids], crop_size)
        val_ds = _SyntheticTorchDataset([by_patient[p] for p in val_ids], crop_size)
        test_ds = _SyntheticTorchDataset([by_patient[p] for p in test_ids], crop_size)
        return train_ds, val_ds, test_ds, train_ids, val_ids, test_ids

    loader = LIDCIDRILoader(data_root=args.data_root)
    if not loader.is_dataset_present():
        raise SystemExit(
            f"LIDC-IDRI not found at {args.data_root}. See "
            "backend/datasets/lidc_idri.py module docstring for how to get "
            "it, or pass --dry-run to test the pipeline on synthetic data."
        )
    annotations = loader.list_annotations()
    patient_ids = sorted({a.patient_id for a in annotations if a.binary_label is not None})
    train_ids, val_ids, test_ids = patient_level_split(
        patient_ids, args.val_frac, args.test_frac, seed=args.seed)

    def subset(ids):
        ids_set = set(ids)
        return [a for a in annotations if a.patient_id in ids_set and a.binary_label is not None]

    train_ds = LIDCIDRIVolumeDataset(subset(train_ids), crop_size=crop_size, augment=True)
    val_ds = LIDCIDRIVolumeDataset(subset(val_ids), crop_size=crop_size, augment=False)
    test_ds = LIDCIDRIVolumeDataset(subset(test_ids), crop_size=crop_size, augment=False)
    return train_ds, val_ds, test_ds, train_ids, val_ids, test_ids


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: str
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for volumes, labels, _ in loader:
        volumes = volumes.to(device)
        logits = model(volumes)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def _collate_keep_metadata(batch):
    """Custom collate: stack volumes/labels normally, but leave the
    per-sample metadata dicts as a plain list. Needed because
    NoduleAnnotation.malignancy_votes is a variable-length list (LIDC-IDRI
    nodules have 1-4 annotating radiologists) — PyTorch's default
    collation tries to stack same-shaped tensors across the batch and
    raises on ragged lists like this.
    """
    volumes = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return volumes, labels, metadata


def train_one_model(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    train_ds, val_ds, test_ds, train_ids, val_ids, test_ids = build_datasets(args)
    print(f"Patients — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"Samples  — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size,
                               collate_fn=_collate_keep_metadata)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=_collate_keep_metadata)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=_collate_keep_metadata)

    model = MODEL_BUILDERS[args.model]().to(device)
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for volumes, labels, _ in train_loader:
            volumes, labels = volumes.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(volumes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * volumes.size(0)
        scheduler.step()

        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_probs = torch.softmax(val_logits, dim=-1)[:, 1].numpy()
        val_labels_np = val_labels.numpy()
        try:
            val_auc = roc_auc_score(val_labels_np, val_probs)
        except ValueError:
            val_auc = float("nan")  # only one class present in this val split

        print(f"epoch {epoch+1}/{args.epochs}  train_loss={running_loss/len(train_ds):.4f}  "
              f"val_auc={val_auc:.4f}")

        if val_auc == val_auc and val_auc > best_val_auc:  # NaN-safe comparison
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Calibration on the validation set ───────────────────────────
    val_logits, val_labels = collect_logits(model, val_loader, device)
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels)

    # ── Held-out test evaluation (never used for model selection) ──
    test_logits, test_labels = collect_logits(model, test_loader, device)
    cal_result = evaluate_calibration(test_logits, test_labels, model_name=args.model,
                                       scaler=scaler, dataset="LIDC-IDRI" if not args.dry_run
                                       else "SYNTHETIC_DRY_RUN", split="test")
    test_probs = torch.softmax(test_logits, dim=-1)[:, 1].numpy()
    test_labels_np = test_labels.numpy()
    test_preds = (test_probs >= 0.5).astype(int)

    tp = int(((test_preds == 1) & (test_labels_np == 1)).sum())
    tn = int(((test_preds == 0) & (test_labels_np == 0)).sum())
    fp = int(((test_preds == 1) & (test_labels_np == 0)).sum())
    fn = int(((test_preds == 0) & (test_labels_np == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    try:
        test_auc = roc_auc_score(test_labels_np, test_probs)
    except ValueError:
        test_auc = float("nan")

    print(f"\nTEST — auc={test_auc:.4f} sensitivity={sensitivity:.4f} "
          f"specificity={specificity:.4f} ece={cal_result.ece:.4f} "
          f"temperature={temperature:.3f}")

    # ── Robustness suite on a sample of the test set ────────────────
    def make_predict_fn(pipeline: CTPipeline):
        def predict_fn(raw_hu_volume: np.ndarray) -> float:
            windowed = pipeline.apply_window(raw_hu_volume)
            t = torch.from_numpy(windowed).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                probs = torch.softmax(model(t), dim=-1)[0, 1].item()
            return probs
        return predict_fn

    robustness_results = []
    if not args.dry_run and len(test_ds) > 0:
        pipeline = CTPipeline(target_spacing=(1.0, 1.0, 1.0))
        predict_fn = make_predict_fn(pipeline)
        n_sample = min(args.robustness_sample_size, len(test_ds))
        for i in range(n_sample):
            # Re-derive the raw (pre-window) crop for perturbation testing —
            # perturbations (hu_shift, noise, etc.) operate in HU space.
            ann = test_ds.annotations[i]
            raw_crop = _get_raw_crop_for_robustness(test_ds, i)
            results = CTPerturbationSuite.run_suite(raw_crop, predict_fn)
            robustness_results.append({"nodule_id": ann.nodule_id, "results": results})

    # ── Persist checkpoint with full provenance metadata ────────────
    split_hash = hashlib.sha256(
        json.dumps({"train": sorted(train_ids), "val": sorted(val_ids),
                     "test": sorted(test_ids)}, sort_keys=True).encode()
    ).hexdigest()[:16]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = "synthetic_dryrun_" if args.dry_run else ""
    ckpt_path = out_dir / f"{name_prefix}{args.model}_3d_best.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": args.model,
        "in_channels": 1,
        "num_classes": 2,
        "crop_size": args.crop_size,
        "target_spacing": (1.0, 1.0, 1.0),
        "class_names": ["benign", "malignant"],
        "temperature": temperature,
        "val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "test_ece": cal_result.ece,
        "patient_split_hash": split_hash,
        "dataset": "SYNTHETIC_DRY_RUN — NOT REAL DATA, DO NOT DEPLOY" if args.dry_run else "LIDC-IDRI",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
    }, ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")

    # ── Log run + register model version ────────────────────────────
    tracker = ExperimentTracker(experiment_name="lung_cancer_3d_classification")
    tracker.log_run(
        run_name=f"{args.model}_{'dryrun' if args.dry_run else 'lidc_idri'}",
        model_name=args.model,
        params={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "crop_size": args.crop_size, "seed": args.seed, "dry_run": args.dry_run},
        metrics={"val_auc": float(best_val_auc), "test_auc": float(test_auc),
                 "test_sensitivity": float(sensitivity), "test_specificity": float(specificity),
                 "test_ece": float(cal_result.ece)},
        tags={"dataset": "SYNTHETIC_DRY_RUN" if args.dry_run else "LIDC-IDRI"},
        artifacts=[str(ckpt_path)],
    )

    if not args.dry_run:
        registry = ModelRegistry(registry_file=str(out_dir / "models_registry.json"))
        registry.register_version(ModelVersionInfo(
            model_name=args.model,
            version=time.strftime("%Y%m%d_%H%M%S"),
            stage="STAGING",  # never auto-PRODUCTION — a human reviews test metrics first
            architecture=args.model,
            input_dimension=f"{args.crop_size}^3",
            metrics={"val_auc": float(best_val_auc), "test_auc": float(test_auc),
                     "test_sensitivity": float(sensitivity), "test_specificity": float(specificity)},
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            checkpoint_path=str(ckpt_path),
            patient_split_hash=split_hash,
            notes="Trained via backend/training/train_classifier.py",
        ))

    if robustness_results:
        rob_path = out_dir / f"{args.model}_robustness_report.json"
        rob_path.write_text(json.dumps(robustness_results, indent=2))
        print(f"Saved robustness report: {rob_path}")


def _get_raw_crop_for_robustness(dataset: "LIDCIDRIVolumeDataset", idx: int) -> np.ndarray:
    """Re-extract the same ROI as __getitem__ but WITHOUT HU windowing, so
    perturbations (which operate in HU space) see realistic values. This
    duplicates a couple of lines of __getitem__ rather than changing its
    return contract for every caller — training/eval loops need the
    windowed tensor, robustness testing needs the raw HU crop.
    """
    import pylidc as pl
    ann = dataset.annotations[idx]
    scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == ann.series_uid).first()
    vol = np.transpose(scan.to_volume(), (2, 0, 1)).astype(np.float32)
    spacing = (float(scan.slice_thickness), float(scan.pixel_spacing), float(scan.pixel_spacing))
    resampled, _ = dataset.pipeline.resample_volume(vol, spacing, order=1)
    scale = np.array(spacing) / np.array(dataset.pipeline.target_spacing)
    center_zyx = tuple(int(round(c * s)) for c, s in zip(ann.centroid_zyx, scale))
    return dataset.pipeline.extract_nodule_crop(resampled, center_zyx, dataset.crop_size,
                                                 pad_value=-1000.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODEL_BUILDERS.keys()), required=True)
    p.add_argument("--data-root", default="data/LIDC-IDRI")
    p.add_argument("--output-dir", default="checkpoints_3d")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--crop-size", type=int, default=64,
                    help="Swin3D hardcodes a 64^3 token-grid assumption — keep at 64 unless "
                         "training resnet3d/densenet3d only.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--robustness-sample-size", type=int, default=20)
    p.add_argument("--dry-run", action="store_true",
                    help="Use synthetic data to smoke-test the full torch pipeline. "
                         "Produces no valid accuracy numbers.")
    return p.parse_args()


if __name__ == "__main__":
    train_one_model(parse_args())
