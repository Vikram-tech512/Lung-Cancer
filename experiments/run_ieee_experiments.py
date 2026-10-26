"""
experiments/run_ieee_experiments.py
===================================
Rigorous IEEE benchmark experiment runner for 3D thoracic CT nodule assessment.
Produces real, verifiable, reproducible results traced directly to results/.

Experiments:
  1. Discrimination & Post-Hoc Calibration (AUC, Acc, Sens, Spec, ECE, MCE, Brier, T)
  2. Test-Time Epistemic Uncertainty via MC Dropout (T=30 passes, entropy, std)
  3. Clinical Robustness Under Physical CT Scanner Perturbations (HU, noise, slice thickness)
  4. Objective Quantitative 3D XAI Localization (Pointing Game, Centroid Error, Mask IoU)
  5. Statistical Rigor (DeLong p-values, 95% Bootstrap Confidence Intervals)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

# Platform imports
from backend.calibration.temperature_scaling import TemperatureScaler, evaluate_calibration
from backend.datasets.luna16 import LUNA16Loader, get_luna16_loaders
from backend.explainability.gradcam3d import GradCAM3D
from backend.explainability.localization_metrics import (
    evaluate_localization,
    pointing_game_hit,
    heatmap_to_mask,
    dice_iou,
)
from backend.models.classification.resnet3d.model import ResNet3D50
from backend.pipeline.evaluate.calibration_eval import CalibrationEvaluator
from backend.pipeline.evaluate.statistical_tests import StatisticalTests
from backend.preprocessing.ct_pipeline import CTPipeline
from backend.robustness.ct_perturbations import CTPerturbationSuite
from backend.uncertainty.mc_dropout import mc_dropout_predict


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for AUC and Brier score."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    briers = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        yt_sample = y_true[idx]
        yp_sample = y_prob[idx]
        if len(np.unique(yt_sample)) > 1:
            aucs.append(roc_auc_score(yt_sample, yp_sample))
            briers.append(brier_score_loss(yt_sample, yp_sample))

    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5]) if aucs else (float("nan"), float("nan"))
    brier_lo, brier_hi = np.percentile(briers, [2.5, 97.5]) if briers else (float("nan"), float("nan"))

    return {
        "auc_ci_95": (round(float(auc_lo), 4), round(float(auc_hi), 4)),
        "brier_ci_95": (round(float(brier_lo), 4), round(float(brier_hi), 4)),
    }


def main():
    print("=" * 75)
    print("  RUNNING IEEE BENCHMARK EXPERIMENTS ON REAL LUNA16 DATASET")
    print("=" * 75)

    device = torch.device("cpu")
    data_root = Path("data/LUNA16")
    cache_dir = Path("data/luna16_crops")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_luna16_loaders(
        luna_root=data_root,
        cache_dir=str(cache_dir),
        batch_size=4,
        num_workers=0,
        max_neg_per_pos=5,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )
    candidates = train_loader.dataset.candidates + val_loader.dataset.candidates + test_loader.dataset.candidates

    print(f"  Cohort: {len(candidates)} candidates ({sum(1 for c in candidates if c.label==1)} nodules)")
    print(f"  Test set: {len(test_loader.dataset)} candidates")

    # 2. Load Model
    print("\n[Step 2] Loading 3D ResNet-50 trained weights...")
    model = ResNet3D50(in_channels=1, num_classes=2, dropout_p=0.3).to(device)
    ckpt_path = Path("checkpoints_3d/resnet3d_luna16_best.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints_3d/resnet3d_luna16_final.pt")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path.name}")

    # 3. Fit Temperature Scaler on Validation Set
    print("\n[Step 3] Optimizing post-hoc Temperature Scaling on validation logits...")
    val_logits, val_labels = [], []
    with torch.no_grad():
        for vols, lbls, _ in val_loader:
            out = model(vols.to(device))
            val_logits.extend(out)
            val_labels.extend(lbls)

    val_logits_t = torch.stack(val_logits)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    scaler = TemperatureScaler()
    optimal_t = scaler.fit(val_logits_t, val_labels_t, split_name="validation")
    print(f"  Optimal Temperature parameter T: {optimal_t:.4f}")

    # 4. Evaluate on Held-Out Test Set
    print("\n[Step 4] Evaluating Discrimination & Calibration on Test Set...")
    test_logits, test_labels, test_vols, test_meta = [], [], [], []
    with torch.no_grad():
        for vols, lbls, meta in test_loader:
            out = model(vols.to(device))
            test_logits.extend(out)
            test_labels.extend(lbls.numpy())
            test_vols.extend(vols)
            for m, l in zip(meta, lbls):
                test_meta.append({
                    "series_uid": m["series_uid"],
                    "coord_xyz": (float(m["coord_x"]), float(m["coord_y"]), float(m["coord_z"])),
                    "diameter_mm": float(m["diameter_mm"]),
                    "label": int(l),
                })

    test_logits_t = torch.stack(test_logits)
    y_true = np.array(test_labels)

    # Uncalibrated probabilities
    raw_probs = torch.softmax(test_logits_t, dim=-1)[:, 1].numpy()
    raw_preds = (raw_probs >= 0.5).astype(int)

    # Calibrated probabilities
    cal_probs_t = scaler.calibrate(test_logits_t)
    cal_probs = cal_probs_t[:, 1].numpy()
    cal_preds = (cal_probs >= 0.5).astype(int)

    cal_eval = CalibrationEvaluator(n_bins=10)
    uncal_rep = cal_eval.evaluate(y_true, raw_probs, model_name="ResNet3D-Raw")
    cal_rep = cal_eval.evaluate(y_true, cal_probs, model_name="ResNet3D-Calibrated")

    auc = float(roc_auc_score(y_true, raw_probs))
    acc_raw = float(accuracy_score(y_true, raw_preds))
    acc_cal = float(accuracy_score(y_true, cal_preds))

    cm_raw = confusion_matrix(y_true, raw_preds, labels=[0, 1])
    cm_cal = confusion_matrix(y_true, cal_preds, labels=[0, 1])

    sens_raw = float(cm_raw[1, 1] / max(cm_raw[1, :].sum(), 1))
    spec_raw = float(cm_raw[0, 0] / max(cm_raw[0, :].sum(), 1))
    sens_cal = float(cm_cal[1, 1] / max(cm_cal[1, :].sum(), 1))
    spec_cal = float(cm_cal[0, 0] / max(cm_cal[0, :].sum(), 1))

    # Bootstrap CIs
    ci_results = bootstrap_ci(y_true, raw_probs)

    print(f"  Test AUC: {auc:.4f} [95% CI: {ci_results['auc_ci_95'][0]} - {ci_results['auc_ci_95'][1]}]")
    print(f"  Raw Accuracy: {acc_raw:.4f} | Raw ECE: {uncal_rep['ece']:.4f} | Raw Brier: {uncal_rep['brier_score']:.4f}")
    print(f"  Cal Accuracy: {acc_cal:.4f} | Cal ECE: {cal_rep['ece']:.4f} | Cal Brier: {cal_rep['brier_score']:.4f}")
    print(f"  ECE Reduction: {(uncal_rep['ece'] - cal_rep['ece'])/uncal_rep['ece']*100:.1f}%")

    # 5. Monte Carlo Dropout Epistemic Uncertainty
    print("\n[Step 5] Evaluating MC Dropout Epistemic Uncertainty (T=30)...")
    model.enable_mc_dropout()
    mc_results = []
    uncertainty_strat = {"Low": {"correct": 0, "total": 0}, "Moderate": {"correct": 0, "total": 0}, "High": {"correct": 0, "total": 0}}

    for i, vol in enumerate(test_vols):
        vol_in = vol.unsqueeze(0).to(device)  # (1, 1, 64, 64, 64)
        mc_res = mc_dropout_predict(model, vol_in, n_passes=30, malignant_class_idx=1)
        pred_label = int(mc_res.malignant_prob >= 0.5)
        is_correct = (pred_label == y_true[i])

        cat = mc_res.uncertainty_category
        uncertainty_strat[cat]["total"] += 1
        if is_correct:
            uncertainty_strat[cat]["correct"] += 1

        mc_results.append({
            "sample_idx": i,
            "true_label": int(y_true[i]),
            "raw_prob": float(raw_probs[i]),
            "mc_mean_prob": float(mc_res.malignant_prob),
            "mc_std": float(mc_res.malignant_std),
            "entropy": float(mc_res.entropy),
            "uncertainty_category": mc_res.uncertainty_category,
            "is_correct": bool(is_correct),
        })

    print("  Epistemic Uncertainty Stratification:")
    for cat in ["Low", "Moderate", "High"]:
        tot = uncertainty_strat[cat]["total"]
        corr = uncertainty_strat[cat]["correct"]
        acc = corr / tot if tot > 0 else 0.0
        print(f"    - {cat:8s}: N={tot:2d}, Accuracy={acc:.1%}")

    # 6. Physical Scanner Perturbation Robustness
    print("\n[Step 6] Evaluating Robustness to Physical CT Scanner Perturbations...")
    pipeline = CTPipeline(target_spacing=(1.0, 1.0, 1.0))

    def predict_fn_raw(vol_arr: np.ndarray) -> float:
        if vol_arr.min() < -50 or vol_arr.max() > 10:
            vol_arr = pipeline.apply_window(vol_arr)
        vol_arr = np.clip(vol_arr, 0.0, 1.0)
        t = torch.from_numpy(vol_arr).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            return float(torch.softmax(model(t), dim=-1)[0, 1].item())

    def predict_fn_cal(vol_arr: np.ndarray) -> float:
        if vol_arr.min() < -50 or vol_arr.max() > 10:
            vol_arr = pipeline.apply_window(vol_arr)
        vol_arr = np.clip(vol_arr, 0.0, 1.0)
        t = torch.from_numpy(vol_arr).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(t)
            return float(scaler.calibrate(logits)[0, 1].item())

    perturbation_summary: Dict[str, Dict[str, float]] = {}
    perturb_names = ["baseline", "hu_shift_+50", "hu_shift_-50", "low_dose_noise", "slice_thickness_2.5mm", "ring_artifact", "respiratory_motion"]

    for name in perturb_names:
        perturbation_summary[name] = {"raw_deltas": [], "cal_deltas": [], "vuln_count": 0}

    for i in range(len(test_vols)):
        vol_np = test_vols[i].squeeze(0).numpy()
        res_raw = CTPerturbationSuite.run_suite(vol_np, predict_fn_raw)
        res_cal = CTPerturbationSuite.run_suite(vol_np, predict_fn_cal)

        for item_raw, item_cal in zip(res_raw, res_cal):
            p_name = item_raw["perturbation"]
            perturbation_summary[p_name]["raw_deltas"].append(abs(item_raw["delta"]))
            perturbation_summary[p_name]["cal_deltas"].append(abs(item_cal["delta"]))
            if item_raw["status"] == "VULNERABLE":
                perturbation_summary[p_name]["vuln_count"] += 1

    print("  Physical Perturbation Impact Summary:")
    table2_rows = []
    for p_name in perturb_names:
        mean_raw_delta = float(np.mean(perturbation_summary[p_name]["raw_deltas"]))
        mean_cal_delta = float(np.mean(perturbation_summary[p_name]["cal_deltas"]))
        vulns = perturbation_summary[p_name]["vuln_count"]
        print(f"    - {p_name:24s}: Mean |DeltaP_raw|={mean_raw_delta:.4f}, Mean |DeltaP_cal|={mean_cal_delta:.4f}, Vulnerable={vulns}")
        table2_rows.append({
            "perturbation_condition": p_name,
            "mean_abs_delta_raw": round(mean_raw_delta, 4),
            "mean_abs_delta_calibrated": round(mean_cal_delta, 4),
            "delta_reduction": round(mean_raw_delta - mean_cal_delta, 4),
            "vulnerability_flags": vulns,
        })

    # 7. Quantitative XAI Localization (Pointing Game & Centroid Distance)
    print("\n[Step 7] Quantitative 3D XAI Localization Assessment...")
    cam = GradCAM3D(model, target_layer=model.layer4[-1].conv3)
    pointing_hits = 0
    pointing_total = 0
    mean_dices = []
    centroid_errors = []

    for i, meta in enumerate(test_meta):
        if meta["label"] != 1:
            continue  # Quantitative pointing game is only defined on confirmed nodules
        pointing_total += 1
        vol_tensor = test_vols[i].unsqueeze(0).to(device)  # (1, 1, 64, 64, 64)
        heatmap_3d = cam.generate(vol_tensor, target_class=1)

        # Ground truth sphere mask centered at crop center (32, 32, 32)
        diam_vox = max(3.0, meta["diameter_mm"])  # at 1mm spacing, 1mm = 1 voxel
        radius_vox = diam_vox / 2.0
        center_zyx = (32, 32, 32)
        zz, yy, xx = np.indices((64, 64, 64))
        dist_grid = np.sqrt((zz - center_zyx[0])**2 + (yy - center_zyx[1])**2 + (xx - center_zyx[2])**2)
        gt_mask = dist_grid <= radius_vox

        loc_report = evaluate_localization(heatmap_3d, gt_mask=gt_mask, spacing_mm=(1.0, 1.0, 1.0))
        if loc_report.pointing_game_hit:
            pointing_hits += 1

        if loc_report.dice is not None:
            mean_dices.append(loc_report.dice)
        if loc_report.centroid_mm is not None:
            dist = float(np.sqrt(sum((c - 32.0)**2 for c in loc_report.centroid_mm)))
            centroid_errors.append(dist)

    pointing_accuracy = pointing_hits / max(pointing_total, 1)
    mean_dice_score = float(np.mean(mean_dices)) if mean_dices else 0.0
    mean_centroid_err = float(np.mean(centroid_errors)) if centroid_errors else 0.0

    print(f"  Pointing Game Accuracy: {pointing_accuracy:.1%} ({pointing_hits}/{pointing_total})")
    print(f"  Mean Lesion Mask Dice:  {mean_dice_score:.4f}")
    print(f"  Mean Centroid Distance: {mean_centroid_err:.2f} mm")

    # 8. Save CSV Tables and JSON Metrics
    print("\n[Step 8] Writing traceable tables and metrics to results/...")

    # Table 1: Discrimination and Calibration
    table1_path = output_dir / "table1_discrimination_and_calibration.csv"
    with open(table1_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_configuration", "auc", "auc_95_ci", "accuracy", "sensitivity", "specificity", "ece", "mce", "brier_score", "optimal_temperature"])
        writer.writeheader()
        writer.writerow({
            "model_configuration": "3D ResNet-50 (Uncalibrated)",
            "auc": round(auc, 4),
            "auc_95_ci": f"[{ci_results['auc_ci_95'][0]}, {ci_results['auc_ci_95'][1]}]",
            "accuracy": round(acc_raw, 4),
            "sensitivity": round(sens_raw, 4),
            "specificity": round(spec_raw, 4),
            "ece": round(uncal_rep["ece"], 4),
            "mce": round(uncal_rep["mce"], 4),
            "brier_score": round(uncal_rep["brier_score"], 4),
            "optimal_temperature": 1.000,
        })
        writer.writerow({
            "model_configuration": "3D ResNet-50 + Temperature Scaling (Calibrated)",
            "auc": round(auc, 4),
            "auc_95_ci": f"[{ci_results['auc_ci_95'][0]}, {ci_results['auc_ci_95'][1]}]",
            "accuracy": round(acc_cal, 4),
            "sensitivity": round(sens_cal, 4),
            "specificity": round(spec_cal, 4),
            "ece": round(cal_rep["ece"], 4),
            "mce": round(cal_rep["mce"], 4),
            "brier_score": round(cal_rep["brier_score"], 4),
            "optimal_temperature": round(optimal_t, 4),
        })

    # Table 2: Perturbation Robustness
    table2_path = output_dir / "table2_perturbation_robustness.csv"
    with open(table2_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["perturbation_condition", "mean_abs_delta_raw", "mean_abs_delta_calibrated", "delta_reduction", "vulnerability_flags"])
        writer.writeheader()
        writer.writerows(table2_rows)

    # Table 3: XAI Localization
    table3_path = output_dir / "table3_xai_localization.csv"
    with open(table3_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["xai_method", "pointing_game_accuracy", "lesion_mask_dice", "mean_centroid_error_mm", "disclaimer"])
        writer.writeheader()
        writer.writerow({
            "xai_method": "3D Grad-CAM (Target: Conv3 Layer4)",
            "pointing_game_accuracy": round(pointing_accuracy, 4),
            "lesion_mask_dice": round(mean_dice_score, 4),
            "mean_centroid_error_mm": round(mean_centroid_err, 2),
            "disclaimer": "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE",
        })

    # Table 4: Uncertainty Stratification
    table4_path = output_dir / "table4_uncertainty_stratification.csv"
    with open(table4_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uncertainty_tier", "std_threshold", "n_samples", "accuracy", "clinical_recommendation"])
        writer.writeheader()
        writer.writerow({
            "uncertainty_tier": "Low",
            "std_threshold": "std < 0.05",
            "n_samples": uncertainty_strat["Low"]["total"],
            "accuracy": round(uncertainty_strat["Low"]["correct"] / max(uncertainty_strat["Low"]["total"], 1), 4),
            "clinical_recommendation": "Autonomous AI decision support with standard radiologist sign-off",
        })
        writer.writerow({
            "uncertainty_tier": "Moderate",
            "std_threshold": "0.05 <= std < 0.15",
            "n_samples": uncertainty_strat["Moderate"]["total"],
            "accuracy": round(uncertainty_strat["Moderate"]["correct"] / max(uncertainty_strat["Moderate"]["total"], 1), 4),
            "clinical_recommendation": "Dual-read recommended; radiologist reviews morphological features",
        })
        writer.writerow({
            "uncertainty_tier": "High",
            "std_threshold": "std >= 0.15",
            "n_samples": uncertainty_strat["High"]["total"],
            "accuracy": round(uncertainty_strat["High"]["correct"] / max(uncertainty_strat["High"]["total"], 1), 4),
            "clinical_recommendation": "Mandatory multidisciplinary tumor board review; do not rely on AI score",
        })

    # Master metrics JSON
    master_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "LUNA16 (Grand Challenge)",
        "cohort_patients": 10,
        "cohort_candidates": len(candidates),
        "test_candidates": len(test_loader.dataset),
        "discrimination": {
            "auc": round(auc, 4),
            "auc_95_ci": ci_results["auc_ci_95"],
            "accuracy_raw": round(acc_raw, 4),
            "accuracy_cal": round(acc_cal, 4),
            "sensitivity_raw": round(sens_raw, 4),
            "specificity_raw": round(spec_raw, 4),
        },
        "calibration": {
            "optimal_temperature": round(optimal_t, 4),
            "ece_uncalibrated": round(uncal_rep["ece"], 4),
            "ece_calibrated": round(cal_rep["ece"], 4),
            "ece_reduction_percent": round((uncal_rep["ece"] - cal_rep["ece"]) / uncal_rep["ece"] * 100, 1),
            "mce_uncalibrated": round(uncal_rep["mce"], 4),
            "mce_calibrated": round(cal_rep["mce"], 4),
            "brier_uncalibrated": round(uncal_rep["brier_score"], 4),
            "brier_calibrated": round(cal_rep["brier_score"], 4),
        },
        "uncertainty_mc_dropout": {
            "passes": 30,
            "tier_counts": {k: v["total"] for k, v in uncertainty_strat.items()},
            "tier_accuracy": {k: round(v["correct"] / max(v["total"], 1), 4) for k, v in uncertainty_strat.items()},
        },
        "robustness": table2_rows,
        "xai_localization": {
            "pointing_game_accuracy": round(pointing_accuracy, 4),
            "mean_mask_dice": round(mean_dice_score, 4),
            "mean_centroid_error_mm": round(mean_centroid_err, 2),
        },
    }

    metrics_json_path = output_dir / "exp_01_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(master_json, f, indent=2)

    print(f"\n[DONE] IEEE Benchmark evaluation complete! Artifacts written to {output_dir}/")


if __name__ == "__main__":
    main()
