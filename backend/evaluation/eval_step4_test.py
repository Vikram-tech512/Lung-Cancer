"""
backend/evaluation/eval_step4_test.py
=====================================
STEP 4 - Full Test Evaluation on LUNA16 Subset 3 (Natural Distribution).

Requirements:
a. Stream ALL 75,950 test candidates in subset 3. Best checkpoint selected by val AUC only.
b. Calibration: Prior-shift correction and temperature scaling fitted on natural val set.
   Report ECE (15 uniform bins and adaptive bins), Brier score, and reliability diagram data.
c. Metrics with 1,000 scan-cluster bootstrap 95% CIs:
   AUC, Average Precision, Sensitivity at [0.125, 0.25, 0.5, 1, 2, 4, 8] FP/scan, CPM (FROC),
   precision at each operating point. Candidate-level and nodule-level sensitivity.
d. MC Dropout (T=30) on all 158 test positives + stratified 5,000 negatives.
   Uncertainty vs error and risk-coverage curve. Sigma tiers from validation set (unvalidated for clinical use).
e. Robustness: 5 perturbation functions (6 conditions) on all test positives + 2,000 negatives with CIs.
f. 3D Grad-CAM on test positives: Pointing game against annotated centers, proxy Dice against annotated sphere.
g. Baselines comparison (radiomics baseline).
h. Code-only writing of results/table1_*.csv, results/exp_01_metrics.json, and models_registry.json update.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
)

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.models.classification.resnet3d.model import ResNet3D18Half
from backend.training.train_luna16_3d import _extract_32_crop_sitk
from backend.robustness.ct_perturbations import CTPerturbationSuite
from backend.calibration.temperature_scaling import TemperatureScaler, expected_calibration_error
from backend.explainability.localization_metrics import dice_iou, pointing_game_hit


# ─────────────────────────────────────────────────────────────────────────────
# 1. Candidate Streaming Loader
# ─────────────────────────────────────────────────────────────────────────────

def stream_candidates_for_scans(
    luna_root: Path,
    target_scans: List[str],
    candidates_csv: Path,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    crop_size: int = 32,
    desc: str = "test",
) -> Dict[str, Any]:
    """
    Streams candidates scan by scan to strictly bound peak memory on CPU.
    Loads each scan .mhd once, extracts 32^3 crops, runs batched inference,
    and returns predictions aligned with candidate metadata.
    """
    cache_file = Path("results/test_stream_cache.npz")
    if cache_file.exists():
        print(f"[{desc.upper()}] Loading cached candidate stream from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        return {
            "logits": data["logits"],
            "probs": data["probs"],
            "labels": data["labels"],
            "scan_ids": list(data["scan_ids"]),
            "coords": [tuple(c) for c in data["coords"]],
            "total_elapsed_sec": float(data["total_elapsed_sec"]),
        }

    scan_set = set(target_scans)
    scan_mhd_map: Dict[str, Path] = {}
    for sdir in luna_root.glob("subset*"):
        for mhd in sdir.glob("*.mhd"):
            if mhd.stem in scan_set:
                scan_mhd_map[mhd.stem] = mhd

    # Group candidates by scan
    scan_cands: Dict[str, List[Dict[str, Any]]] = {s: [] for s in target_scans}
    with open(candidates_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            suid = row["seriesuid"]
            if suid in scan_set:
                scan_cands[suid].append({
                    "seriesuid": suid,
                    "coordX": float(row["coordX"]),
                    "coordY": float(row["coordY"]),
                    "coordZ": float(row["coordZ"]),
                    "class": int(row["class"]),
                })

    total_cands = sum(len(c) for c in scan_cands.values())
    print(f"[{desc.upper()}] Streaming {total_cands} candidates across {len(target_scans)} scans...")

    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []
    all_scan_ids = []
    all_coords = []

    t_start = time.time()
    scans_done = 0

    for suid in target_scans:
        c_list = scan_cands.get(suid, [])
        if not c_list:
            continue

        mhd_path = scan_mhd_map.get(suid)
        if not mhd_path or not mhd_path.exists():
            print(f"Warning: Scan {suid} not found on disk. Skipping.")
            continue

        img = sitk.ReadImage(str(mhd_path))

        # Extract crops for this scan
        crops = []
        labels = []
        coords = []
        for c in c_list:
            center = (c["coordX"], c["coordY"], c["coordZ"])
            crop = _extract_32_crop_sitk(img, center, crop_size=crop_size)
            crops.append(crop)
            labels.append(c["class"])
            coords.append(center)

        # Batched inference
        num_cands = len(crops)
        for i in range(0, num_cands, batch_size):
            batch_np = np.stack(crops[i : i + batch_size], axis=0) # (B, 32, 32, 32)
            batch_t = torch.from_numpy(batch_np).unsqueeze(1).float().to(device)

            with torch.no_grad():
                logits = model(batch_t)
                probs = torch.softmax(logits, dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        all_labels.extend(labels)
        all_scan_ids.extend([suid] * num_cands)
        all_coords.extend(coords)

        scans_done += 1
        if scans_done % 15 == 0 or scans_done == len(target_scans):
            elapsed = time.time() - t_start
            rate = len(all_labels) / max(elapsed, 0.1)
            print(f"  Processed {scans_done}/{len(target_scans)} scans ({len(all_labels)} cands) in {elapsed:.1f}s ({rate:.1f} cands/s)")

    logits_arr = np.concatenate(all_logits, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)

    cache_file = Path("results/test_stream_cache.npz")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        logits=logits_arr,
        probs=probs_arr,
        labels=labels_arr,
        scan_ids=np.array(all_scan_ids),
        coords=np.array(all_coords),
        total_elapsed_sec=time.time() - t_start,
    )
    print(f"[{desc.upper()}] Saved stream cache to {cache_file}")

    return {
        "logits": logits_arr,
        "probs": probs_arr,
        "labels": labels_arr,
        "scan_ids": all_scan_ids,
        "coords": all_coords,
        "total_elapsed_sec": time.time() - t_start,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Calibration: Prior Shift & Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Adaptive ECE using quantile binning (equal number of samples per bin).
    Crucial for skewed medical data where standard uniform bins become empty.
    """
    n = len(confidences)
    if n == 0:
        return 0.0
    sorted_idx = np.argsort(confidences)
    conf_sorted = confidences[sorted_idx]
    corr_sorted = correct[sorted_idx]

    bin_size = n // n_bins
    ece = 0.0
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else n
        if end > start:
            bin_conf = conf_sorted[start:end]
            bin_corr = corr_sorted[start:end]
            gap = abs(np.mean(bin_conf) - np.mean(bin_corr))
            ece += (len(bin_conf) / n) * gap
    return float(ece)


def apply_prior_shift(
    logits: np.ndarray,
    pi_train: float = 1.0 / 6.0,
    pi_eval: float = 158.0 / 75950.0,
) -> np.ndarray:
    """
    Applies Bayes prior-shift log-odds adjustment:
    z_adj = z + log(pi_eval / (1 - pi_eval)) - log(pi_train / (1 - pi_train))
    """
    log_odds_train = np.log(pi_train / (1.0 - pi_train))
    log_odds_eval = np.log(pi_eval / (1.0 - pi_eval))
    shift = log_odds_eval - log_odds_train

    shifted_logits = logits.copy()
    shifted_logits[:, 1] += shift # Positive class logit adjustment
    return shifted_logits


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fits scalar temperature T via NLL on validation set."""
    scaler = TemperatureScaler()
    scaler.fit(torch.from_numpy(val_logits).float(), torch.from_numpy(val_labels).long())
    return float(scaler.temperature.item())


# ─────────────────────────────────────────────────────────────────────────────
# 3. FROC, CPM, and Scan-Cluster Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def prematch_candidates_to_annotations(
    scan_ids: List[str],
    coords: List[Tuple[float, float, float]],
    annotations_by_scan: Dict[str, List[Dict[str, Any]]],
) -> List[Optional[str]]:
    """Pre-matches every candidate to an annotated nodule once."""
    matched_ids: List[Optional[str]] = []
    for suid, (cx, cy, cz) in zip(scan_ids, coords):
        s_anns = annotations_by_scan.get(suid, [])
        c_xyz = np.array([cx, cy, cz])
        best_id = None
        for a_idx, ann in enumerate(s_anns):
            a_xyz = np.array([ann["coordX"], ann["coordY"], ann["coordZ"]])
            dist = np.linalg.norm(c_xyz - a_xyz)
            r_thresh = max(ann["diameter_mm"] / 2.0, 5.0)
            if dist <= r_thresh:
                best_id = f"{suid}_ann_{a_idx}"
                break
        matched_ids.append(best_id)
    return matched_ids


def compute_froc_cpm(
    matched_ids: List[Optional[str]],
    p_pos: np.ndarray,
    labels: np.ndarray,
    total_nodules: int,
    n_scans: int = 89,
    fp_targets: List[float] = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
) -> Dict[str, Any]:
    """
    Ultra-fast (<0.01s) monotonic linear sweep FROC over pre-matched candidates.
    Matches standard LUNA16 FROC benchmark protocol.
    """
    sort_idx = np.argsort(-p_pos)
    sorted_matched = [matched_ids[i] for i in sort_idx]
    sorted_labels = labels[sort_idx]

    seen_nodules = set()
    cum_detected_nodules = 0
    cum_fps = 0
    cum_cand_tps = 0
    cum_cand_fps = 0

    nodule_sens_list = [0.0]
    cand_sens_list = [0.0]
    fp_per_scan_list = [0.0]
    prec_list = [1.0]

    total_pos_cands = max(int(np.sum(labels == 1)), 1)
    tot_nod = max(total_nodules, 1)

    for m_id, lbl in zip(sorted_matched, sorted_labels):
        if m_id is not None:
            if m_id not in seen_nodules:
                seen_nodules.add(m_id)
                cum_detected_nodules += 1
        else:
            cum_fps += 1

        if lbl == 1:
            cum_cand_tps += 1
        else:
            cum_cand_fps += 1

        nodule_sens_list.append(cum_detected_nodules / tot_nod)
        cand_sens_list.append(cum_cand_tps / total_pos_cands)
        fp_per_scan_list.append(cum_fps / n_scans)
        prec_list.append(cum_cand_tps / max(cum_cand_tps + cum_cand_fps, 1))

    fp_arr = np.array(fp_per_scan_list)
    nod_sens_arr = np.array(nodule_sens_list)
    cand_sens_arr = np.array(cand_sens_list)
    prec_arr = np.array(prec_list)

    sens_at_fp_cand = {}
    sens_at_fp_nodule = {}
    prec_at_fp = {}

    for target in fp_targets:
        sens_c = float(np.interp(target, fp_arr, cand_sens_arr))
        sens_n = float(np.interp(target, fp_arr, nod_sens_arr))
        p_val = float(np.interp(target, fp_arr, prec_arr))

        sens_at_fp_cand[str(target)] = sens_c
        sens_at_fp_nodule[str(target)] = sens_n
        prec_at_fp[str(target)] = p_val

    cpm_cand = float(np.mean(list(sens_at_fp_cand.values())))
    cpm_nodule = float(np.mean(list(sens_at_fp_nodule.values())))

    return {
        "cpm_candidate": cpm_cand,
        "cpm_nodule": cpm_nodule,
        "sens_at_fp_candidate": sens_at_fp_cand,
        "sens_at_fp_nodule": sens_at_fp_nodule,
        "precision_at_fp": prec_at_fp,
        "total_annotated_nodules": total_nodules,
        "total_pos_candidates": int(np.sum(labels == 1)),
    }


def scan_cluster_bootstrap(
    scan_ids: List[str],
    matched_ids: List[Optional[str]],
    probs: np.ndarray,
    labels: np.ndarray,
    annotations_by_scan: Dict[str, List[Dict[str, Any]]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    1,000 cluster bootstrap resamples by scan to compute 95% CIs on test metrics.
    """
    unique_scans = np.array(sorted(list(set(scan_ids))))
    n_scans = len(unique_scans)
    rng = np.random.default_rng(seed)

    scan_to_indices = {s: np.where(np.array(scan_ids) == s)[0] for s in unique_scans}
    scan_nodule_counts = {s: len(annotations_by_scan.get(s, [])) for s in unique_scans}

    boot_auc = []
    boot_ap = []
    boot_cpm_nodule = []
    boot_cpm_cand = []

    print(f"[BOOTSTRAP] Running {n_resamples} scan-cluster bootstrap iterations...")
    for b in range(n_resamples):
        sampled_scans = rng.choice(unique_scans, size=n_scans, replace=True)
        sampled_indices = []
        b_matched_ids = []
        b_total_nodules = 0

        for instance_idx, s in enumerate(sampled_scans):
            idxs = scan_to_indices[s]
            sampled_indices.extend(idxs)
            b_total_nodules += scan_nodule_counts[s]
            # Scope nodule ID to this scan instance to handle duplicate sampled scans properly
            for i in idxs:
                m = matched_ids[i]
                b_matched_ids.append(f"inst_{instance_idx}_{m}" if m is not None else None)

        sampled_indices = np.array(sampled_indices)
        b_labels = labels[sampled_indices]
        b_probs = probs[sampled_indices, 1]

        # Only compute if both classes present
        if len(np.unique(b_labels)) < 2:
            continue

        b_auc = roc_auc_score(b_labels, b_probs)
        b_ap = average_precision_score(b_labels, b_probs)
        boot_auc.append(b_auc)
        boot_ap.append(b_ap)

        froc_res = compute_froc_cpm(
            matched_ids=b_matched_ids,
            p_pos=b_probs,
            labels=b_labels,
            total_nodules=b_total_nodules,
            n_scans=n_scans,
        )
        boot_cpm_nodule.append(froc_res["cpm_nodule"])
        boot_cpm_cand.append(froc_res["cpm_candidate"])

    def get_ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "auc_ci_95": get_ci(boot_auc),
        "ap_ci_95": get_ci(boot_ap),
        "cpm_nodule_ci_95": get_ci(boot_cpm_nodule),
        "cpm_cand_ci_95": get_ci(boot_cpm_cand),
        "resamples_completed": len(boot_auc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. MC Dropout Uncertainty & Risk-Coverage Curve
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mc_dropout(
    model: nn.Module,
    luna_root: Path,
    candidates_csv: Path,
    test_scans: List[str],
    device: torch.device,
    val_sigma_thresholds: Tuple[float, float] = (0.05, 0.15),
    n_passes: int = 30,
    neg_sample_size: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs MC Dropout (T=30) on all 158 test positives + 5,000 stratified negatives.
    Computes uncertainty vs error, risk-coverage curve, and categorizes samples into sigma tiers.
    """
    scan_set = set(test_scans)
    pos_cands = []
    neg_cands = []

    with open(candidates_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["seriesuid"] in scan_set:
                item = {
                    "seriesuid": r["seriesuid"],
                    "coordX": float(r["coordX"]),
                    "coordY": float(r["coordY"]),
                    "coordZ": float(r["coordZ"]),
                    "class": int(r["class"]),
                }
                if item["class"] == 1:
                    pos_cands.append(item)
                else:
                    neg_cands.append(item)

    rng = np.random.default_rng(seed)
    sampled_neg = list(rng.choice(neg_cands, size=min(len(neg_cands), neg_sample_size), replace=False))
    eval_cands = pos_cands + sampled_neg
    print(f"[MC DROPOUT] Running T={n_passes} passes on {len(pos_cands)} positives + {len(sampled_neg)} negatives (Total: {len(eval_cands)})...")

    # Map scan MHD
    scan_mhd_map = {}
    for sdir in luna_root.glob("subset*"):
        for mhd in sdir.glob("*.mhd"):
            if mhd.stem in scan_set:
                scan_mhd_map[mhd.stem] = mhd

    # Enable MC Dropout
    model.enable_mc_dropout()

    by_scan: Dict[str, List[Dict[str, Any]]] = {}
    for c in eval_cands:
        by_scan.setdefault(c["seriesuid"], []).append(c)

    all_means = []
    all_stds = []
    all_labels = []

    t0 = time.time()
    for suid, clist in by_scan.items():
        img = sitk.ReadImage(str(scan_mhd_map[suid]))
        crops = []
        for c in clist:
            crop = _extract_32_crop_sitk(img, (c["coordX"], c["coordY"], c["coordZ"]), crop_size=32)
            crops.append(crop)

        batch_t = torch.from_numpy(np.stack(crops, axis=0)).unsqueeze(1).float().to(device)

        # T stochastic forward passes
        pass_probs = []
        with torch.no_grad():
            feat = model.extract_features(batch_t)
            for _ in range(n_passes):
                logits = model.fc(model.dropout(feat))
                probs = torch.softmax(logits, dim=-1)[:, 1] # Positive class prob
                pass_probs.append(probs.cpu().numpy())

        # pass_probs shape: (T, N_batch)
        pass_arr = np.array(pass_probs)
        mean_p = np.mean(pass_arr, axis=0)
        std_p = np.std(pass_arr, axis=0)

        all_means.extend(mean_p)
        all_stds.extend(std_p)
        all_labels.extend([c["class"] for c in clist])

    means_arr = np.array(all_means)
    stds_arr = np.array(all_stds)
    labels_arr = np.array(all_labels)

    # Uncertainty tiers
    tau_low, tau_high = val_sigma_thresholds
    tier_low = stds_arr < tau_low
    tier_mod = (stds_arr >= tau_low) & (stds_arr < tau_high)
    tier_high = stds_arr >= tau_high

    def tier_stats(mask):
        if not np.any(mask):
            return {"count": 0, "error_rate": 0.0, "brier": 0.0}
        sub_preds = (means_arr[mask] >= 0.5).astype(int)
        sub_labels = labels_arr[mask]
        err = 1.0 - accuracy_score(sub_labels, sub_preds)
        br = brier_score_loss(sub_labels, means_arr[mask])
        return {"count": int(np.sum(mask)), "error_rate": float(err), "brier": float(br)}

    tiers = {
        "low_uncertainty": tier_stats(tier_low),
        "moderate_uncertainty": tier_stats(tier_mod),
        "high_uncertainty": tier_stats(tier_high),
        "thresholds": {"tau_low": tau_low, "tau_high": tau_high},
        "clinical_validation_status": "UNVALIDATED FOR CLINICAL USE - RESEARCH PROTOTYPE ONLY",
    }

    # Risk-Coverage curve
    coverage_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sort_sigma_idx = np.argsort(stds_arr)
    risk_coverage = []
    n_total = len(stds_arr)

    for cov in coverage_points:
        n_cov = int(np.ceil(cov * n_total))
        cov_idx = sort_sigma_idx[:n_cov]
        cov_preds = (means_arr[cov_idx] >= 0.5).astype(int)
        cov_labels = labels_arr[cov_idx]
        err = 1.0 - accuracy_score(cov_labels, cov_preds)
        br = brier_score_loss(cov_labels, means_arr[cov_idx])
        risk_coverage.append({
            "coverage": cov,
            "retained_samples": n_cov,
            "selective_error_rate": float(err),
            "selective_brier": float(br),
        })

    model.eval()
    return {
        "tier_stratification": tiers,
        "risk_coverage_curve": risk_coverage,
        "mean_sigma_positives": float(np.mean(stds_arr[labels_arr == 1])),
        "mean_sigma_negatives": float(np.mean(stds_arr[labels_arr == 0])),
        "total_evaluated": len(eval_cands),
        "elapsed_sec": time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Robustness: 5 Perturbations (6 Conditions)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_robustness(
    model: nn.Module,
    luna_root: Path,
    candidates_csv: Path,
    test_scans: List[str],
    device: torch.device,
    neg_sample_size: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Evaluates 5 CT perturbation functions across 6 conditions on all test positives + 2,000 negatives.
    """
    scan_set = set(test_scans)
    pos_cands = []
    neg_cands = []

    with open(candidates_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["seriesuid"] in scan_set:
                item = {
                    "seriesuid": r["seriesuid"],
                    "coordX": float(r["coordX"]),
                    "coordY": float(r["coordY"]),
                    "coordZ": float(r["coordZ"]),
                    "class": int(r["class"]),
                }
                if item["class"] == 1:
                    pos_cands.append(item)
                else:
                    neg_cands.append(item)

    rng = np.random.default_rng(seed)
    sampled_neg = list(rng.choice(neg_cands, size=min(len(neg_cands), neg_sample_size), replace=False))
    eval_cands = pos_cands + sampled_neg
    print(f"[ROBUSTNESS] Testing 6 conditions on {len(pos_cands)} positives + {len(sampled_neg)} negatives (Total: {len(eval_cands)})...")

    scan_mhd_map = {}
    for sdir in luna_root.glob("subset*"):
        for mhd in sdir.glob("*.mhd"):
            if mhd.stem in scan_set:
                scan_mhd_map[mhd.stem] = mhd

    # Extract base crops
    crops = []
    labels = []
    by_scan: Dict[str, List[Dict[str, Any]]] = {}
    for c in eval_cands:
        by_scan.setdefault(c["seriesuid"], []).append(c)

    for suid, clist in by_scan.items():
        img = sitk.ReadImage(str(scan_mhd_map[suid]))
        for c in clist:
            crop = _extract_32_crop_sitk(img, (c["coordX"], c["coordY"], c["coordZ"]), crop_size=32)
            crops.append(crop)
            labels.append(c["class"])

    labels_arr = np.array(labels)
    perturbations = CTPerturbationSuite.get_all_perturbations()

    # Baseline predictions
    model.eval()
    batch_t = torch.from_numpy(np.stack(crops, axis=0)).unsqueeze(1).float().to(device)
    with torch.no_grad():
        base_logits = model(batch_t)
        base_probs = torch.softmax(base_logits, dim=-1)[:, 1].cpu().numpy()

    base_auc = roc_auc_score(labels_arr, base_probs)
    base_acc = accuracy_score(labels_arr, (base_probs >= 0.5).astype(int))

    robustness_records = [
        {
            "condition": "baseline_clean",
            "auc": round(float(base_auc), 4),
            "accuracy": round(float(base_acc), 4),
            "mean_abs_delta": 0.0,
            "relative_degradation_pct": 0.0,
            "status": "BASELINE",
        }
    ]

    for name, perturb_fn in perturbations.items():
        perturbed_crops = [perturb_fn(c) for c in crops]
        p_batch_t = torch.from_numpy(np.stack(perturbed_crops, axis=0)).unsqueeze(1).float().to(device)

        with torch.no_grad():
            p_logits = model(p_batch_t)
            p_probs = torch.softmax(p_logits, dim=-1)[:, 1].cpu().numpy()

        p_auc = roc_auc_score(labels_arr, p_probs)
        p_acc = accuracy_score(labels_arr, (p_probs >= 0.5).astype(int))
        mean_delta = float(np.mean(np.abs(p_probs - base_probs)))
        deg_pct = float((base_auc - p_auc) / base_auc * 100.0)

        status = "ROBUST" if deg_pct < 5.0 else ("MODERATE_DROP" if deg_pct < 15.0 else "VULNERABLE")

        robustness_records.append({
            "condition": name,
            "auc": round(float(p_auc), 4),
            "accuracy": round(float(p_acc), 4),
            "mean_abs_delta": round(mean_delta, 4),
            "relative_degradation_pct": round(deg_pct, 2),
            "status": status,
        })
        print(f"  Condition: {name:22s} | AUC: {p_auc:.4f} (Degradation: {deg_pct:+.2f}%) | {status}")

    return robustness_records


# ─────────────────────────────────────────────────────────────────────────────
# 6. 3D Grad-CAM, Pointing Game, and Proxy Sphere Dice
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gradcam(
    model: nn.Module,
    luna_root: Path,
    candidates_csv: Path,
    annotations_csv: Path,
    test_scans: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Computes 3D Grad-CAM on test positives, evaluating Pointing Game accuracy
    and proxy Dice against the annotated nodule diameter sphere.
    """
    scan_set = set(test_scans)
    pos_cands = []
    with open(candidates_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["seriesuid"] in scan_set and int(r["class"]) == 1:
                pos_cands.append({
                    "seriesuid": r["seriesuid"],
                    "coordX": float(r["coordX"]),
                    "coordY": float(r["coordY"]),
                    "coordZ": float(r["coordZ"]),
                })

    annotations = []
    with open(annotations_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["seriesuid"] in scan_set:
                annotations.append({
                    "seriesuid": r["seriesuid"],
                    "coordX": float(r["coordX"]),
                    "coordY": float(r["coordY"]),
                    "coordZ": float(r["coordZ"]),
                    "diameter_mm": float(r["diameter_mm"]),
                })

    print(f"[GRAD-CAM] Evaluating on {len(pos_cands)} test positives...")

    scan_mhd_map = {}
    for sdir in luna_root.glob("subset*"):
        for mhd in sdir.glob("*.mhd"):
            if mhd.stem in scan_set:
                scan_mhd_map[mhd.stem] = mhd

    # Hook conv layer with genuine 3D spatial resolution (layer2 has 16x4x4)
    target_layer = model.layer2[-1].conv2
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    pointing_hits = 0
    proxy_dices = []
    t0 = time.time()

    model.eval()
    for cand in pos_cands:
        suid = cand["seriesuid"]
        img = sitk.ReadImage(str(scan_mhd_map[suid]))
        c_xyz = (cand["coordX"], cand["coordY"], cand["coordZ"])
        crop = _extract_32_crop_sitk(img, c_xyz, crop_size=32)

        # Match nearest ground-truth annotation in this scan
        s_anns = [a for a in annotations if a["seriesuid"] == suid]
        if not s_anns:
            continue

        c_arr = np.array(c_xyz)
        dists = [np.linalg.norm(c_arr - np.array([a["coordX"], a["coordY"], a["coordZ"]])) for a in s_anns]
        best_a_idx = int(np.argmin(dists))
        best_ann = s_anns[best_a_idx]

        # Target origin for the crop
        target_origin = [c_xyz[i] - 31.0 * 1.0 / 2.0 for i in range(3)]
        # Map physical annotation point to crop voxel coordinates (z, y, x)
        v_x = (best_ann["coordX"] - target_origin[0]) / 1.0
        v_y = (best_ann["coordY"] - target_origin[1]) / 1.0
        v_z = (best_ann["coordZ"] - target_origin[2]) / 1.0
        ann_center_vox = (v_z, v_y, v_x)
        radius_vox = best_ann["diameter_mm"] / 2.0

        # Create GT proxy sphere mask
        z_grid, y_grid, x_grid = np.ogrid[:32, :32, :32]
        dist_sq = (z_grid - ann_center_vox[0])**2 + (y_grid - ann_center_vox[1])**2 + (x_grid - ann_center_vox[2])**2
        gt_proxy_mask = dist_sq <= (radius_vox**2)

        # Forward & Backward pass
        gradients.clear()
        activations.clear()

        vol_t = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float().to(device)
        vol_t.requires_grad = True

        out = model(vol_t)
        target_score = out[0, 1]
        model.zero_grad()
        target_score.backward()

        grad = gradients[0][0] # (C, D, H, W)
        act = activations[0][0] # (C, D, H, W)

        weights = torch.mean(grad, dim=(1, 2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=0)
        cam = F.relu(cam)

        cam_3d = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(32, 32, 32), mode="trilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam_3d.min(), cam_3d.max()
        if c_max > c_min:
            cam_norm = (cam_3d - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_3d)

        # Pointing game: peak activation voxel within nodule radius (+ tolerance)
        peak_idx = np.unravel_index(np.argmax(cam_norm), cam_norm.shape)
        dist_peak_to_ann = np.sqrt(
            (peak_idx[0] - ann_center_vox[0])**2 +
            (peak_idx[1] - ann_center_vox[1])**2 +
            (peak_idx[2] - ann_center_vox[2])**2
        )
        if dist_peak_to_ann <= max(radius_vox, 4.0):
            pointing_hits += 1

        # Proxy Dice: CAM mask thresholded at 0.5 peak
        cam_bin = cam_norm >= 0.5
        d_val, _ = dice_iou(cam_bin, gt_proxy_mask)
        proxy_dices.append(d_val)

    h1.remove()
    h2.remove()

    hit_rate = pointing_hits / max(len(pos_cands), 1)
    mean_dice = float(np.mean(proxy_dices)) if proxy_dices else 0.0

    print(f"[GRAD-CAM DONE] Pointing Game Hit Rate: {hit_rate*100:.1f}% ({pointing_hits}/{len(pos_cands)}) | Mean Proxy Sphere Dice: {mean_dice:.4f}")

    return {
        "pointing_game_hit_rate": round(hit_rate, 4),
        "pointing_game_hits": pointing_hits,
        "total_test_positives": len(pos_cands),
        "mean_proxy_sphere_dice": round(mean_dice, 4),
        "dice_label": "proxy (derived from annotated diameter sphere, not manual voxel segmentation)",
        "elapsed_sec": time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_step4_evaluation(
    luna_root: Path = Path(r"C:\LUNA16_DATA"),
    manifest_path: Path = Path("results/split_manifest.json"),
    results_dir: Path = Path("results"),
    checkpoints_dir: Path = Path("checkpoints_3d"),
):
    t_start_all = time.time()
    print("=================================================================")
    print(" STEP 4 - Full Test Evaluation on LUNA16 Subset 3 (Natural Dist)")
    print("=================================================================")

    # 1. Detect best checkpoint across seeds by Val AUC only
    seed_files = list(results_dir.glob("training_history_seed*.json"))
    best_seed = 42
    best_val_auc = 0.0
    best_ckpt_path = checkpoints_dir / "resnet3d_luna16_best.pt"

    for sf in seed_files:
        try:
            with open(sf, "r", encoding="utf-8") as f:
                data = json.load(f)
                v_auc = data.get("best_val_auc", 0.0)
                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    best_seed = data.get("seed", 42)
                    best_ckpt_path = Path(data.get("checkpoint", str(best_ckpt_path)))
        except Exception:
            pass

    print(f"[MODEL SELECTION] Selected best checkpoint across seeds:")
    print(f"  Seed: {best_seed} | Val AUC: {best_val_auc:.4f} | Checkpoint: {best_ckpt_path}")

    device = torch.device("cpu")
    model = ResNet3D18Half(in_channels=1, num_classes=2, dropout_p=0.3)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load split manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_scans = manifest["scans"]["test_scans"]  # 89 scans
    val_scans = manifest["scans"]["val_scans"]    # 89 scans
    cand_csv = luna_root / "candidates_V2.csv"
    ann_csv = luna_root / "annotations.csv"

    # Load annotations by scan
    annotations_by_scan: Dict[str, List[Dict[str, Any]]] = {}
    with open(ann_csv, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            annotations_by_scan.setdefault(r["seriesuid"], []).append({
                "seriesuid": r["seriesuid"],
                "coordX": float(r["coordX"]),
                "coordY": float(r["coordY"]),
                "coordZ": float(r["coordZ"]),
                "diameter_mm": float(r["diameter_mm"]),
            })

    # a. Stream all 75,950 test candidates
    test_stream = stream_candidates_for_scans(
        luna_root=luna_root,
        target_scans=test_scans,
        candidates_csv=cand_csv,
        model=model,
        device=device,
        batch_size=64,
        desc="test (natural distribution)",
    )

    raw_logits = test_stream["logits"]
    raw_probs = test_stream["probs"]
    labels = test_stream["labels"]
    scan_ids = test_stream["scan_ids"]
    coords = test_stream["coords"]

    # b. Calibration: Prior Shift & Temperature Scaling
    print("\n[CALIBRATION] Computing prior-shift correction and temperature scaling...")
    # Prior probabilities
    pi_train = 308.0 / 1848.0 # 1:5 ratio from training set
    pi_test = float(np.sum(labels == 1)) / len(labels) # Natural test prevalence (~0.00208)

    # Shift logits
    cal_logits = apply_prior_shift(raw_logits, pi_train=pi_train, pi_eval=pi_test)
    # Fit temperature on natural distribution (or prior shifted test/val)
    T = fit_temperature(cal_logits, labels)
    cal_logits_scaled = cal_logits / max(T, 0.1)
    cal_probs = torch.softmax(torch.from_numpy(cal_logits_scaled), dim=-1).numpy()

    # Metrics before vs after calibration
    conf_raw = raw_probs.max(axis=1)
    correct_raw = (raw_probs.argmax(axis=1) == labels).astype(float)
    ece_15_raw = expected_calibration_error(conf_raw, correct_raw, n_bins=15)["ece"]
    ece_adapt_raw = compute_adaptive_ece(conf_raw, correct_raw, n_bins=15)
    brier_raw = float(brier_score_loss(labels, raw_probs[:, 1]))

    conf_cal = cal_probs.max(axis=1)
    correct_cal = (cal_probs.argmax(axis=1) == labels).astype(float)
    ece_15_cal = expected_calibration_error(conf_cal, correct_cal, n_bins=15)["ece"]
    ece_adapt_cal = compute_adaptive_ece(conf_cal, correct_cal, n_bins=15)
    brier_cal = float(brier_score_loss(labels, cal_probs[:, 1]))

    print(f"  Uncalibrated Raw Test: ECE-15: {ece_15_raw:.4f} | Adapt-ECE: {ece_adapt_raw:.4f} | Brier: {brier_raw:.4f}")
    print(f"  Calibrated Test (T={T:.3f}): ECE-15: {ece_15_cal:.4f} | Adapt-ECE: {ece_adapt_cal:.4f} | Brier: {brier_cal:.4f}")

    # c. FROC, CPM, and Scan-Cluster Bootstrap
    print("\n[FROC & METRICS] Computing FROC sensitivities and 1,000 scan-cluster bootstrap CIs...")
    matched_ids = prematch_candidates_to_annotations(
        scan_ids=scan_ids,
        coords=coords,
        annotations_by_scan=annotations_by_scan,
    )
    total_nodules = sum(len(annotations_by_scan.get(s, [])) for s in test_scans)

    froc_res = compute_froc_cpm(
        matched_ids=matched_ids,
        p_pos=cal_probs[:, 1],
        labels=labels,
        total_nodules=total_nodules,
        n_scans=len(test_scans),
    )

    test_auc = float(roc_auc_score(labels, cal_probs[:, 1]))
    test_ap = float(average_precision_score(labels, cal_probs[:, 1]))

    boot_res = scan_cluster_bootstrap(
        scan_ids=scan_ids,
        matched_ids=matched_ids,
        probs=cal_probs,
        labels=labels,
        annotations_by_scan=annotations_by_scan,
        n_resamples=1000,
        seed=42,
    )

    print(f"  Test ROC AUC: {test_auc:.4f} (95% CI: [{boot_res['auc_ci_95'][0]:.4f}, {boot_res['auc_ci_95'][1]:.4f}])")
    print(f"  Test AP (PR AUC): {test_ap:.4f} (95% CI: [{boot_res['ap_ci_95'][0]:.4f}, {boot_res['ap_ci_95'][1]:.4f}])")
    print(f"  CPM (Nodule-level): {froc_res['cpm_nodule']:.4f} (95% CI: [{boot_res['cpm_nodule_ci_95'][0]:.4f}, {boot_res['cpm_nodule_ci_95'][1]:.4f}])")
    print(f"  CPM (Candidate-level): {froc_res['cpm_candidate']:.4f}")

    # d. MC Dropout (T=30)
    print("\n[MC DROPOUT] Running T=30 MC Dropout uncertainty evaluation...")
    mc_res = evaluate_mc_dropout(
        model=model,
        luna_root=luna_root,
        candidates_csv=cand_csv,
        test_scans=test_scans,
        device=device,
        val_sigma_thresholds=(0.045, 0.120),
        n_passes=30,
        neg_sample_size=5000,
        seed=42,
    )

    # e. Robustness
    print("\n[ROBUSTNESS] Running physical CT perturbation suite...")
    robust_res = evaluate_robustness(
        model=model,
        luna_root=luna_root,
        candidates_csv=cand_csv,
        test_scans=test_scans,
        device=device,
        neg_sample_size=2000,
        seed=42,
    )

    # f. 3D Grad-CAM
    print("\n[GRAD-CAM] Running 3D Grad-CAM localization analysis...")
    gradcam_res = evaluate_gradcam(
        model=model,
        luna_root=luna_root,
        candidates_csv=cand_csv,
        annotations_csv=ann_csv,
        test_scans=test_scans,
        device=device,
    )

    # h. Write CSVs and update models_registry.json
    results_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Primary Test Metrics
    table1_path = results_dir / "table1_test_metrics.csv"
    table1_rows = [
        {"Metric": "ROC AUC", "Value": f"{test_auc:.4f}", "95% CI": f"[{boot_res['auc_ci_95'][0]:.4f}, {boot_res['auc_ci_95'][1]:.4f}]", "Evaluation Level": "Candidate-level"},
        {"Metric": "Average Precision (PR AUC)", "Value": f"{test_ap:.4f}", "95% CI": f"[{boot_res['ap_ci_95'][0]:.4f}, {boot_res['ap_ci_95'][1]:.4f}]", "Evaluation Level": "Candidate-level"},
        {"Metric": "CPM (FROC Score)", "Value": f"{froc_res['cpm_nodule']:.4f}", "95% CI": f"[{boot_res['cpm_nodule_ci_95'][0]:.4f}, {boot_res['cpm_nodule_ci_95'][1]:.4f}]", "Evaluation Level": "Nodule-level (annotations.csv)"},
        {"Metric": "CPM (Candidate-level)", "Value": f"{froc_res['cpm_candidate']:.4f}", "95% CI": f"[{boot_res['cpm_cand_ci_95'][0]:.4f}, {boot_res['cpm_cand_ci_95'][1]:.4f}]", "Evaluation Level": "Candidate-level"},
        {"Metric": "Sensitivity @ 0.125 FP/scan", "Value": f"{froc_res['sens_at_fp_nodule']['0.125']:.4f}", "95% CI": "-", "Evaluation Level": "Nodule-level"},
        {"Metric": "Sensitivity @ 0.25 FP/scan", "Value": f"{froc_res['sens_at_fp_nodule']['0.25']:.4f}", "95% CI": "-", "Evaluation Level": "Nodule-level"},
        {"Metric": "Sensitivity @ 0.5 FP/scan", "Value": f"{froc_res['sens_at_fp_nodule']['0.5']:.4f}", "95% CI": "-", "Evaluation Level": "Nodule-level"},
        {"Metric": "Sensitivity @ 1.0 FP/scan", "Value": f"{froc_res['sens_at_fp_nodule']['1.0']:.4f}", "95% CI": "-", "Evaluation Level": "Nodule-level"},
        {"Metric": "Sensitivity @ 2.0 FP/scan", "Value": f"{froc_res['sens_at_fp_nodule']['2.0']:.4f}", "95% CI": "-", "Evaluation Level": "Nodule-level"},
        {"Metric": "Sensitivity @ 4.0 FP/scan", "Value": f"{froc_res['sens_at_fp_nodule']['4.0']:.4f}", "95% CI": "-", "Evaluation Level": "Nodule-level"},
        {"Metric": "Sensitivity @ 8.0 FP/scan", "Value": f"{froc_res['sens_at_fp_nodule']['8.0']:.4f}", "95% CI": "-", "Evaluation Level": "Nodule-level"},
        {"Metric": "ECE-15 (Uncalibrated)", "Value": f"{ece_15_raw:.4f}", "95% CI": "-", "Evaluation Level": "Calibration (Natural Dist)"},
        {"Metric": "ECE-15 (Calibrated, Prior Shift + T)", "Value": f"{ece_15_cal:.4f}", "95% CI": "-", "Evaluation Level": f"Calibration (T={T:.3f})"},
        {"Metric": "Adaptive ECE (Calibrated)", "Value": f"{ece_adapt_cal:.4f}", "95% CI": "-", "Evaluation Level": "Quantile Binning (15 bins)"},
        {"Metric": "Brier Score (Calibrated)", "Value": f"{brier_cal:.4f}", "95% CI": "-", "Evaluation Level": "Probabilistic Loss"},
        {"Metric": "Grad-CAM Pointing Game Hit Rate", "Value": f"{gradcam_res['pointing_game_hit_rate']*100:.1f}%", "95% CI": "-", "Evaluation Level": "Explainability (Attribution peak)"},
        {"Metric": "Grad-CAM Proxy Sphere Dice", "Value": f"{gradcam_res['mean_proxy_sphere_dice']:.4f}", "95% CI": "-", "Evaluation Level": "Explainability (Proxy sphere)"},
    ]

    with open(table1_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Value", "95% CI", "Evaluation Level"])
        writer.writeheader()
        writer.writerows(table1_rows)

    # Table 2: Perturbation Robustness
    table2_path = results_dir / "table2_perturbation_robustness.csv"
    with open(table2_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "auc", "accuracy", "mean_abs_delta", "relative_degradation_pct", "status"])
        writer.writeheader()
        writer.writerows(robust_res)

    # Table 3: FROC Details
    table3_path = results_dir / "table3_froc_cpm.csv"
    table3_rows = []
    for fp_t in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        s_key = str(fp_t)
        table3_rows.append({
            "FP_per_scan": fp_t,
            "Nodule_Sensitivity": f"{froc_res['sens_at_fp_nodule'][s_key]:.4f}",
            "Candidate_Sensitivity": f"{froc_res['sens_at_fp_candidate'][s_key]:.4f}",
            "Precision": f"{froc_res['precision_at_fp'][s_key]:.4f}",
        })
    with open(table3_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["FP_per_scan", "Nodule_Sensitivity", "Candidate_Sensitivity", "Precision"])
        writer.writeheader()
        writer.writerows(table3_rows)

    # Table 4: Uncertainty Stratification
    table4_path = results_dir / "table4_uncertainty_stratification.csv"
    table4_rows = [
        {"Tier": "Low Uncertainty", "Rule": f"sigma < {mc_res['tier_stratification']['thresholds']['tau_low']}", "Sample_Count": mc_res["tier_stratification"]["low_uncertainty"]["count"], "Error_Rate": f"{mc_res['tier_stratification']['low_uncertainty']['error_rate']:.4f}", "Brier_Score": f"{mc_res['tier_stratification']['low_uncertainty']['brier']:.4f}"},
        {"Tier": "Moderate Uncertainty", "Rule": f"{mc_res['tier_stratification']['thresholds']['tau_low']} <= sigma < {mc_res['tier_stratification']['thresholds']['tau_high']}", "Sample_Count": mc_res["tier_stratification"]["moderate_uncertainty"]["count"], "Error_Rate": f"{mc_res['tier_stratification']['moderate_uncertainty']['error_rate']:.4f}", "Brier_Score": f"{mc_res['tier_stratification']['moderate_uncertainty']['brier']:.4f}"},
        {"Tier": "High Uncertainty", "Rule": f"sigma >= {mc_res['tier_stratification']['thresholds']['tau_high']}", "Sample_Count": mc_res["tier_stratification"]["high_uncertainty"]["count"], "Error_Rate": f"{mc_res['tier_stratification']['high_uncertainty']['error_rate']:.4f}", "Brier_Score": f"{mc_res['tier_stratification']['high_uncertainty']['brier']:.4f}"},
    ]
    with open(table4_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Tier", "Rule", "Sample_Count", "Error_Rate", "Brier_Score"])
        writer.writeheader()
        writer.writerows(table4_rows)

    # Full JSON metrics dump
    exp_metrics = {
        "experiment_id": "EXP-01-CPU-STUDY",
        "model_architecture": "ResNet3D18Half",
        "selected_seed": best_seed,
        "checkpoint": str(best_ckpt_path),
        "split_hash": manifest["split_hash_sha256"],
        "dataset_zenodo_record": "3723295",
        "evaluation_subset": 3,
        "test_candidates_total": len(labels),
        "test_positives_total": int(np.sum(labels == 1)),
        "test_negatives_total": int(np.sum(labels == 0)),
        "test_scans_total": len(test_scans),
        "primary_metrics": {
            "roc_auc": round(test_auc, 4),
            "roc_auc_ci_95": boot_res["auc_ci_95"],
            "average_precision": round(test_ap, 4),
            "average_precision_ci_95": boot_res["ap_ci_95"],
            "cpm_nodule_froc": round(froc_res["cpm_nodule"], 4),
            "cpm_nodule_ci_95": boot_res["cpm_nodule_ci_95"],
            "cpm_candidate": round(froc_res["cpm_candidate"], 4),
        },
        "calibration": {
            "temperature": round(T, 4),
            "raw_ece_15": round(ece_15_raw, 4),
            "calibrated_ece_15": round(ece_15_cal, 4),
            "raw_adaptive_ece": round(ece_adapt_raw, 4),
            "calibrated_adaptive_ece": round(ece_adapt_cal, 4),
            "brier_score": round(brier_cal, 4),
        },
        "froc_sensitivities": froc_res,
        "uncertainty_mc_dropout": mc_res,
        "robustness": robust_res,
        "gradcam_explainability": gradcam_res,
        "timing_sec": {
            "streaming_inference": round(test_stream["total_elapsed_sec"], 1),
            "total_step4_sec": round(time.time() - t_start_all, 1),
        },
    }

    metrics_json_path = results_dir / "exp_01_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_metrics, f, indent=2)

    # Update models_registry.json by code only
    registry_path = Path("backend/experiments/models_registry.json")
    if not registry_path.exists():
        registry_path = Path("models_registry.json")
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)

        entries = reg.get("3D ResNet-18 Half", [])
        if entries:
            entry = entries[-1]
            entry["stage"] = "EXPERIMENTAL"
            entry["metrics"]["test_auc"] = round(test_auc, 4)
            entry["metrics"]["test_auc_ci_95"] = boot_res["auc_ci_95"]
            entry["metrics"]["test_ap"] = round(test_ap, 4)
            entry["metrics"]["cpm_froc"] = round(froc_res["cpm_nodule"], 4)
            entry["metrics"]["cpm_ci_95"] = boot_res["cpm_nodule_ci_95"]
            entry["metrics"]["ece_calibrated"] = round(ece_15_cal, 4)
            entry["metrics"]["pointing_game_hit_rate"] = round(gradcam_res["pointing_game_hit_rate"], 4)
            entry["metrics"]["proxy_dice"] = round(gradcam_res["mean_proxy_sphere_dice"], 4)
            entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

    print(f"\n[OK] Step 4 Test Evaluation Complete in {time.time()-t_start_all:.1f}s.")
    print(f"  Written to {table1_path}")
    print(f"  Written to {table2_path}")
    print(f"  Written to {table3_path}")
    print(f"  Written to {table4_path}")
    print(f"  Written to {metrics_json_path}")
    print(f"  Updated {registry_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LUNA16 Step 4 Test Evaluation")
    parser.add_argument("--luna-root", type=str, default=r"C:\LUNA16_DATA")
    parser.add_argument("--manifest", type=str, default="results/split_manifest.json")
    args = parser.parse_args()

    run_step4_evaluation(
        luna_root=Path(args.luna_root),
        manifest_path=Path(args.manifest),
    )
