"""
backend/training/train_luna16_3d.py
=====================================
Full training pipeline for 3D lung nodule classification on LUNA16.

Trains your existing 3D models (ResNet3D-50, DenseNet3D-121, SwinTransformer3D)
on real LUNA16 CT scans (.mhd/.raw format). Fully integrates with the
existing platform infrastructure:

  - backend.datasets.luna16           → LUNA16Dataset + loaders
  - backend.preprocessing.ct_pipeline → HU windowing, isotropic resampling
  - backend.models.classification.*   → ResNet3D50, DenseNet3D121, SwinTransformer3D
  - backend.calibration               → Temperature scaling
  - backend.robustness                → CT perturbation stress tests
  - backend.experiments               → MLflow tracking + model registry

FEATURES:
  ✅ Patient/scan-level splits (no leakage)
  ✅ Balanced WeightedRandomSampler (handles 1:550 pos:neg imbalance)
  ✅ 3D augmentation (flips, 90° rotations, noise, intensity jitter)
  ✅ Disk-based crop cache (avoids re-reading 12 GB .mhd on every epoch)
  ✅ Mixed precision (AMP) on CUDA
  ✅ Cosine annealing + warm restarts LR schedule
  ✅ Class-weighted cross-entropy + label smoothing
  ✅ Early stopping on validation AUC
  ✅ Temperature scaling calibration post-training
  ✅ Sensitivity / Specificity / ECE / AUC on held-out test set
  ✅ Robustness suite (CT perturbations)
  ✅ MLflow experiment logging
  ✅ Model registry (STAGING stage, no auto-PRODUCTION)

HOW TO GET LUNA16 DATA (Free):
  Option A — Kaggle subset (fastest, ~15 GB):
    pip install kagglehub
    python -c "import kagglehub; print(kagglehub.dataset_download('fanbyprinciple/luna-lung-cancer'))"

  Option B — Official full dataset (~120 GB, 10 subsets):
    pip install zenodo-get
    zenodo_get 3723295   # downloads all subsets to ./

USAGE:
  # Quick pipeline smoke-test (synthetic 64³ volumes, no real data needed):
  python -m backend.training.train_luna16_3d --model resnet3d --dry-run

  # Train on real LUNA16 data (GPU recommended):
  python -m backend.training.train_luna16_3d \\
      --model resnet3d \\
      --data-root /path/to/luna16 \\
      --cache-dir /path/to/luna16_crops_cache \\
      --epochs 50 --batch-size 4

  # Resume from checkpoint:
  python -m backend.training.train_luna16_3d \\
      --model resnet3d --data-root /path/to/luna16 \\
      --resume checkpoints_3d/resnet3d_luna16_best.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import sys
# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for special chars)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from torch.amp import autocast, GradScaler
except ImportError as e:
    raise SystemExit(
        "PyTorch not found. Install: pip install -r backend/requirements.txt"
    ) from e

from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)

# ── Platform imports ────────────────────────────────────────────────────────
from backend.datasets.luna16 import (
    LUNA16Loader, LUNA16Dataset, get_luna16_loaders, luna16_collate
)
from backend.datasets.synthetic_volumes import make_synthetic_cohort
from backend.preprocessing.ct_pipeline import CTPipeline
from backend.calibration.temperature_scaling import TemperatureScaler, evaluate_calibration
from backend.robustness.ct_perturbations import CTPerturbationSuite
from backend.experiments.mlflow_tracker import ExperimentTracker
from backend.experiments.model_registry import ModelRegistry, ModelVersionInfo

# ── 3D Model zoo ────────────────────────────────────────────────────────────
from backend.models.classification.resnet3d.model   import ResNet3D50
from backend.models.classification.densenet3d.model import DenseNet3D121
from backend.models.classification.swin3d.model     import SwinTransformer3D

MODEL_BUILDERS = {
    "resnet3d":   lambda: ResNet3D50(in_channels=1, num_classes=2),
    "densenet3d": lambda: DenseNet3D121(in_channels=1, num_classes=2),
    "swin3d":     lambda: SwinTransformer3D(in_channels=1, num_classes=2),
}

CLASS_NAMES = ["non-nodule", "nodule"]


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic dry-run dataset (no real data required)
# ─────────────────────────────────────────────────────────────────────────────

class _SyntheticLUNA16Dataset(torch.utils.data.Dataset):
    """Wraps synthetic nodule volumes as a LUNA16-compatible dataset for dry-runs."""

    def __init__(self, nodules, crop_size=(64, 64, 64)):
        from backend.datasets.synthetic_volumes import SyntheticNodule
        self.nodules   = nodules
        self.pipeline  = CTPipeline(target_spacing=(1.0, 1.0, 1.0))
        self.crop_size = crop_size

    def __len__(self):
        return len(self.nodules)

    def __getitem__(self, idx):
        n = self.nodules[idx]
        crop = self.pipeline.extract_nodule_crop(
            n.volume, n.centroid_zyx, self.crop_size, pad_value=-1000.0
        )
        crop = self.pipeline.apply_window(crop)
        vol_t   = torch.from_numpy(crop).unsqueeze(0).float()
        label_t = torch.tensor(n.binary_label, dtype=torch.long)
        meta    = {"series_uid": n.patient_id, "synthetic": True,
                   "label": n.binary_label, "diameter_mm": 0.0,
                   "coord_x": 0.0, "coord_y": 0.0, "coord_z": 0.0, "subset_id": -1}
        return vol_t, label_t, meta

    def class_weights(self):
        labels = [n.binary_label for n in self.nodules]
        counts = np.bincount(labels, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (2.0 * counts)
        return torch.tensor(w, dtype=torch.float32)


def _build_dry_run_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build synthetic loaders for pipeline validation (--dry-run)."""
    print(
        "\n!! --dry-run: SYNTHETIC 64³ volumes. Proves pipeline runs. "
        "Proves NOTHING about real accuracy. !!\n"
    )
    n_total = max(24, args.batch_size * 6)
    cohort  = make_synthetic_cohort(n_patients=n_total, seed=args.seed)
    from backend.datasets.lidc_idri import patient_level_split
    pids        = [n.patient_id for n in cohort]
    tr, va, te  = patient_level_split(pids, args.val_frac, args.test_frac, args.seed)
    by_pid      = {n.patient_id: n for n in cohort}
    cs          = (args.crop_size, args.crop_size, args.crop_size)

    tr_ds = _SyntheticLUNA16Dataset([by_pid[p] for p in tr], cs)
    va_ds = _SyntheticLUNA16Dataset([by_pid[p] for p in va], cs)
    te_ds = _SyntheticLUNA16Dataset([by_pid[p] for p in te], cs)

    kw = dict(batch_size=args.batch_size, num_workers=0, collate_fn=luna16_collate)
    return (
        DataLoader(tr_ds, shuffle=True,  **kw),
        DataLoader(va_ds, shuffle=False, **kw),
        DataLoader(te_ds, shuffle=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    Optional[GradScaler],
    grad_clip: float = 1.0,
) -> dict:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for volumes, labels, _ in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(volumes)
            loss   = criterion(logits, labels)

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
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss":     total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
    }


@torch.no_grad()
def _evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_preds, all_probs, all_labels = [], [], [], []

    for volumes, labels, _ in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        logits = model(volumes)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=-1)

        total_loss += loss.item()
        all_logits.extend(logits.cpu())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    labels_np = np.array(all_labels)
    probs_np  = np.array(all_probs)
    preds_np  = np.array(all_preds)

    try:
        auc = float(roc_auc_score(labels_np, probs_np[:, 1]))
    except ValueError:
        auc = float("nan")

    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    tn = int(((preds_np == 0) & (labels_np == 0)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())

    return {
        "loss":        total_loss / max(len(loader), 1),
        "accuracy":    accuracy_score(labels_np, preds_np),
        "auc":         auc,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "_logits":     torch.stack(all_logits),
        "_labels":     torch.tensor(labels_np, dtype=torch.long),
        "_preds":      preds_np,
        "_probs":      probs_np,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main trainer
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU — training will be slow. GPU strongly recommended for 3D CNNs.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[Data] Building LUNA16 loaders...")
    if args.dry_run:
        train_loader, val_loader, test_loader = _build_dry_run_loaders(args)
        dataset_tag = "SYNTHETIC_DRY_RUN"
    else:
        train_loader, val_loader, test_loader = get_luna16_loaders(
            luna_root        = args.data_root,
            crop_size        = args.crop_size,
            batch_size       = args.batch_size,
            num_workers      = args.num_workers,
            max_neg_per_pos  = args.max_neg_per_pos,
            min_diameter_mm  = args.min_diameter_mm,
            val_frac         = args.val_frac,
            test_frac        = args.test_frac,
            seed             = args.seed,
            cache_dir        = args.cache_dir,
            balanced_sampling= True,
        )
        dataset_tag = "LUNA16"

    print(
        f"  Batches — train: {len(train_loader)} | "
        f"val: {len(val_loader)} | test: {len(test_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[Model] Building {args.model}...")
    if args.model not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{args.model}'. Choose: {list(MODEL_BUILDERS)}")
    model = MODEL_BUILDERS[args.model]().to(device)

    if args.resume:
        print(f"[Model] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {total_params:.1f}M")

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Class-weighted CE — positives are rare even with neg subsampling
    if not args.dry_run:
        cw = train_loader.dataset.class_weights().to(device)
    else:
        cw = torch.tensor([1.0, 4.0]).to(device)  # rough 1:4 ratio for dry-run

    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler    = GradScaler() if device.type == "cuda" else None

    best_auc     = 0.0
    best_ckpt    = output_dir / f"{args.model}_luna16_best.pt"
    last_ckpt    = output_dir / f"{args.model}_luna16_last.pt"
    patience_cnt = 0
    history      = []

    print(f"\n{'='*70}")
    print(f"  LUNA16 3D Nodule Classification — {args.model.upper()}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | CropSize: {args.crop_size}³")
    print(f"  LR: {args.lr} | Device: {device} | Dataset: {dataset_tag}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = _train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_m   = _evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        row = {
            "epoch":       epoch,
            "train_loss":  round(train_m["loss"], 4),
            "train_acc":   round(train_m["accuracy"], 4),
            "val_loss":    round(val_m["loss"], 4),
            "val_acc":     round(val_m["accuracy"], 4),
            "val_auc":     round(val_m["auc"], 4),
            "val_sens":    round(val_m["sensitivity"], 4),
            "val_spec":    round(val_m["specificity"], 4),
            "lr":          round(optimizer.param_groups[0]["lr"], 6),
            "elapsed_s":   round(elapsed, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"TrainLoss={row['train_loss']:.4f} Acc={row['train_acc']:.3f} | "
            f"ValLoss={row['val_loss']:.4f} Acc={row['val_acc']:.3f} AUC={row['val_auc']:.4f} "
            f"Sens={row['val_sens']:.3f} Spec={row['val_spec']:.3f} | "
            f"LR={row['lr']:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "architecture":     args.model,
            "in_channels":      1,
            "num_classes":      2,
            "crop_size":        args.crop_size,
            "target_spacing":   (1.0, 1.0, 1.0),
            "class_names":      CLASS_NAMES,
            "val_auc":          val_m["auc"],
            "epoch":            epoch,
            "args":             vars(args),
            "dataset":          dataset_tag,
            "trained_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(ckpt_payload, last_ckpt)

        if val_m["auc"] > best_auc:
            best_auc     = val_m["auc"]
            patience_cnt = 0
            torch.save(ckpt_payload, best_ckpt)
            print(f"  [NEW BEST] AUC: {best_auc:.4f} -- saved to {best_ckpt.name}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping.")
                break

        if args.dry_run:
            print("[Dry Run] 1 epoch complete -- exiting.")
            break

    # ── Temperature Scaling Calibration ───────────────────────────────────
    print(f"\n[Calibration] Loading best checkpoint: {best_ckpt}")
    best = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best["model_state_dict"])

    val_m_final = _evaluate(model, val_loader, criterion, device)
    ts_scaler   = TemperatureScaler()
    temperature = ts_scaler.fit(val_m_final["_logits"], val_m_final["_labels"])
    print(f"  Calibration temperature: {temperature:.3f}")

    # ── Test Evaluation ────────────────────────────────────────────────────
    print("\n[Test] Evaluating on held-out test set...")
    test_m = _evaluate(model, test_loader, criterion, device)

    cal_result = evaluate_calibration(
        test_m["_logits"], test_m["_labels"],
        model_name=args.model,
        scaler=ts_scaler,
        dataset=dataset_tag,
        split="test",
    )

    print(f"\n{'='*70}")
    print(f"  TEST RESULTS — {args.model.upper()} on {dataset_tag}")
    print(f"  AUC:         {test_m['auc']:.4f}")
    print(f"  Accuracy:    {test_m['accuracy']:.4f}")
    print(f"  Sensitivity: {test_m['sensitivity']:.4f}  (nodule recall)")
    print(f"  Specificity: {test_m['specificity']:.4f}  (non-nodule recall)")
    print(f"  ECE:         {cal_result.ece:.4f}  (calibration error)")
    print(f"  Temperature: {temperature:.3f}")
    print(f"{'='*70}")

    print("\nClassification Report:")
    print(classification_report(
        test_m["_labels"].numpy(), test_m["_preds"],
        target_names=CLASS_NAMES, digits=4,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(test_m["_labels"].numpy(), test_m["_preds"]))

    # ── Robustness Suite ──────────────────────────────────────────────────
    robustness_results = []
    if not args.dry_run and args.robustness_sample_size > 0:
        print(f"\n[Robustness] Testing {args.robustness_sample_size} samples...")
        pipeline    = CTPipeline(target_spacing=(1.0, 1.0, 1.0))

        def predict_fn(raw_hu: np.ndarray) -> float:
            windowed = pipeline.apply_window(raw_hu)
            t = torch.from_numpy(windowed).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                return float(torch.softmax(model(t), dim=-1)[0, 1].item())

        test_ds = test_loader.dataset
        n_sample = min(args.robustness_sample_size, len(test_ds))
        for i in range(n_sample):
            vol_t, _, meta = test_ds[i]
            raw = vol_t.squeeze(0).numpy()     # already windowed; perturbations work on it
            results = CTPerturbationSuite.run_suite(raw, predict_fn)
            robustness_results.append({"series_uid": meta["series_uid"], "results": results})

        rob_path = output_dir / f"{args.model}_luna16_robustness.json"
        rob_path.write_text(json.dumps(robustness_results, indent=2))
        print(f"  Robustness report → {rob_path}")

    # ── Save final checkpoint with all metrics ────────────────────────────
    split_hash = hashlib.sha256(
        f"{args.seed}_{dataset_tag}_{args.model}".encode()
    ).hexdigest()[:16]

    name_prefix = "dryrun_" if args.dry_run else ""
    final_ckpt  = output_dir / f"{name_prefix}{args.model}_luna16_final.pt"
    torch.save({
        **best,
        "temperature":      temperature,
        "test_auc":         test_m["auc"],
        "test_accuracy":    test_m["accuracy"],
        "test_sensitivity": test_m["sensitivity"],
        "test_specificity": test_m["specificity"],
        "test_ece":         cal_result.ece,
        "patient_split_hash": split_hash,
        "dataset":          ("SYNTHETIC_DRY_RUN — NOT REAL DATA" if args.dry_run else dataset_tag),
    }, final_ckpt)
    print(f"\n[Checkpoint] Final -> {final_ckpt}")

    # ── Save training history ─────────────────────────────────────────────
    results_json = {
        "best_val_auc":     best_auc,
        "test_auc":         test_m["auc"],
        "test_accuracy":    test_m["accuracy"],
        "test_sensitivity": test_m["sensitivity"],
        "test_specificity": test_m["specificity"],
        "test_ece":         cal_result.ece,
        "temperature":      temperature,
        "dataset":          dataset_tag,
        "model":            args.model,
        "history":          history,
        "args":             vars(args),
    }
    hist_path = output_dir / f"{args.model}_luna16_results.json"
    with open(hist_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[Results]    History -> {hist_path}")

    # ── MLflow + Model Registry ───────────────────────────────────────────
    if not args.dry_run:
        tracker = ExperimentTracker(experiment_name="lung_cancer_3d_luna16")
        tracker.log_run(
            run_name   = f"{args.model}_luna16_{time.strftime('%Y%m%d_%H%M%S')}",
            model_name = args.model,
            params     = vars(args),
            metrics    = {
                "val_auc": best_auc,
                "test_auc": test_m["auc"],
                "test_sensitivity": test_m["sensitivity"],
                "test_specificity": test_m["specificity"],
                "test_ece": cal_result.ece,
            },
            tags       = {"dataset": "LUNA16"},
            artifacts  = [str(final_ckpt)],
        )

        registry = ModelRegistry(registry_file=str(output_dir / "models_registry.json"))
        registry.register_version(ModelVersionInfo(
            model_name        = args.model,
            version           = time.strftime("%Y%m%d_%H%M%S"),
            stage             = "STAGING",  # human reviews before PRODUCTION
            architecture      = args.model,
            input_dimension   = f"{args.crop_size}^3",
            metrics           = {
                "val_auc":          best_auc,
                "test_auc":         test_m["auc"],
                "test_sensitivity": test_m["sensitivity"],
                "test_specificity": test_m["specificity"],
            },
            created_at        = time.strftime("%Y-%m-%d %H:%M:%S"),
            checkpoint_path   = str(final_ckpt),
            patient_split_hash= split_hash,
            notes             = "Trained via backend/training/train_luna16_3d.py on LUNA16",
        ))

    print("\n[DONE] Training complete!")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 3D nodule classifier on LUNA16",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",             choices=list(MODEL_BUILDERS), default="resnet3d")
    p.add_argument("--data-root",         type=str, default="data/LUNA16",
                   help="Path to LUNA16 root dir (must contain subset0..N + candidates CSV)")
    p.add_argument("--cache-dir",         type=str, default="data/luna16_crops",
                   help="Directory to cache extracted 3D crops as .npy (strongly recommended)")
    p.add_argument("--output-dir",        type=str, default="checkpoints_3d")
    p.add_argument("--epochs",            type=int, default=50)
    p.add_argument("--batch-size",        type=int, default=4)
    p.add_argument("--crop-size",         type=int, default=64,
                   help="Cubic crop side length in voxels. SwinTransformer3D requires 64.")
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--patience",          type=int, default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--max-neg-per-pos",   type=int, default=5,
                   help="Max ratio of negative to positive candidates")
    p.add_argument("--min-diameter-mm",   type=float, default=0.0,
                   help="Exclude nodules smaller than this diameter")
    p.add_argument("--val-frac",          type=float, default=0.15)
    p.add_argument("--test-frac",         type=float, default=0.15)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--num-workers",       type=int, default=2)
    p.add_argument("--device",            type=str, default="cuda")
    p.add_argument("--robustness-sample-size", type=int, default=20)
    p.add_argument("--resume",            type=str, default=None,
                   help="Path to .pt checkpoint to resume from")
    p.add_argument("--dry-run",           action="store_true",
                   help="Use synthetic data to smoke-test the pipeline (no LUNA16 needed)")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
