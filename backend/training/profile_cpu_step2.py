"""
backend/training/profile_cpu_step2.py
======================================
CPU Training Profiler for 3D Lung Nodule Classification (Step 2).

System Specs:
- Windows CPU-only (16 GB RAM total, 12 GB allocation limit, 512 GB disk).
- Target architectures: 3D ResNet-18 and Half-Width 3D ResNet-18 (8.3M params).
- Input: 32x32x32 voxel crops at 1.0 mm isotropic spacing, [-1000, 400] HU windowed.
- Dataset: 20-scan subset from LUNA16 official Train partition (subsets 0, 1).

Objectives:
1. Verify 32^3 crop extraction pipeline and HU windowing.
2. Profile memory footprint across pipeline stages (assert peak < 12 GB).
3. Benchmark batch size feasibility (B=8 vs B=16) and model throughput (crops/sec).
4. Run 1-epoch dry-run training loop with real loss updates.
5. Project per-epoch training time for the full 178-scan Train partition.
6. Write results to results/cpu_profile_step2.json.
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
from torch.utils.data import DataLoader, Dataset

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.models.classification.resnet3d.model import BasicBlock3d, ResNet3D18



# ─────────────────────────────────────────────────────────────────────────────
#  Architecture Definitions
# ─────────────────────────────────────────────────────────────────────────────

class ResNet3D18Half(nn.Module):
    """
    Half-width 3D ResNet-18 variant designed for CPU compute budgets.
    Channels: 32 -> 64 -> 128 -> 256 (8.29M params vs 33.15M params for standard).
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Fast 32^3 Subvolume Crop Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_32_crop(
    image: sitk.Image,
    center_xyz: Tuple[float, float, float],
    crop_size: int = 32,
    target_spacing: float = 1.0,
    window_min: float = -1000.0,
    window_max: float = 400.0,
) -> np.ndarray:
    """
    Directly resample a 32^3 subvolume around center_xyz in world coordinates.
    Applies [-1000, 400] HU windowing normalized to [0, 1].
    """
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
    arr = sitk.GetArrayFromImage(crop_img).astype(np.float32)  # (D, H, W)

    # HU windowing [-1000, 400] -> [0.0, 1.0]
    clipped = np.clip(arr, window_min, window_max)
    norm = (clipped - window_min) / (window_max - window_min)
    return norm


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset Class for Profiling
# ─────────────────────────────────────────────────────────────────────────────

class LUNA16ProfileDataset(Dataset):
    """Dataset holding pre-extracted 32^3 crops for profiling."""

    def __init__(self, crops: List[np.ndarray], labels: List[int]):
        self.crops = [torch.from_numpy(c).unsqueeze(0).float() for c in crops]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.crops[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Main Profiling Routine
# ─────────────────────────────────────────────────────────────────────────────

def get_current_ram_mb() -> float:
    """Return resident set size (RSS) memory in MB for current process."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def run_cpu_profile(
    luna_root: Path = Path(r"C:\LUNA16_DATA"),
    manifest_path: Path = Path("results/split_manifest.json"),
    n_scans: int = 20,
    neg_to_pos_ratio: int = 10,
    seed: int = 42,
) -> Dict:
    print("=================================================================")
    print(" Step 2: CPU Training Profiler (3D ResNet-18 & Half-Width)")
    print("=================================================================")
    
    baseline_ram_mb = get_current_ram_mb()
    print(f"[RAM] Baseline memory: {baseline_ram_mb:.1f} MB")

    # 1. Load Split Manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_scans = manifest["scans"]["train_scans"][:n_scans]
    scan_set = set(train_scans)
    print(f"[DATA] Selected {len(train_scans)} scans from Train set.")

    # 2. Locate MHD files
    scan_mhd_map = {}
    for subset_id in (0, 1):
        sdir = luna_root / f"subset{subset_id}"
        for mhd in sdir.glob("*.mhd"):
            if mhd.stem in scan_set:
                scan_mhd_map[mhd.stem] = mhd

    assert len(scan_mhd_map) == n_scans, f"Expected {n_scans} MHD scans, found {len(scan_mhd_map)}"
    print(f"[OK] Located all {n_scans} scan files on disk.")

    # 3. Parse candidates_V2.csv
    cand_csv = luna_root / "candidates_V2.csv"
    pos_candidates = []
    neg_candidates = []

    with open(cand_csv, "r", newline="", encoding="utf-8") as f:
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
                    pos_candidates.append(entry)
                else:
                    neg_candidates.append(entry)

    # Balance negatives
    np.random.seed(seed)
    n_pos = len(pos_candidates)
    n_neg = min(len(neg_candidates), n_pos * neg_to_pos_ratio)
    sampled_indices = np.random.choice(len(neg_candidates), size=n_neg, replace=False)
    sampled_neg = [neg_candidates[i] for i in sampled_indices]

    profile_candidates = pos_candidates + sampled_neg
    np.random.shuffle(profile_candidates)

    print(f"[CANDIDATES] Cohort candidates: {n_pos} positives + {n_neg} negatives = {len(profile_candidates)} total.")

    # 4. Extract 32^3 Crops & Measure Extraction Throughput
    print(f"\n--- Extracting 32^3 isotropic crops (HU [-1000, 400] -> [0, 1]) ---")
    t0_extract = time.time()
    crops = []
    labels = []

    # Group by scan to minimize disk reads
    by_scan: Dict[str, List] = {}
    for c in profile_candidates:
        by_scan.setdefault(c["seriesuid"], []).append(c)

    crops_extracted = 0
    ram_after_extraction = baseline_ram_mb

    for suid, c_list in by_scan.items():
        mhd_path = scan_mhd_map[suid]
        img = sitk.ReadImage(str(mhd_path))
        for cand in c_list:
            coord = (cand["coord_x"], cand["coord_y"], cand["coord_z"])
            crop_32 = extract_32_crop(img, coord, crop_size=32)
            crops.append(crop_32)
            labels.append(cand["class"])
            crops_extracted += 1

        cur_ram = get_current_ram_mb()
        if cur_ram > ram_after_extraction:
            ram_after_extraction = cur_ram

    dt_extract = time.time() - t0_extract
    extract_rate = crops_extracted / dt_extract
    print(f"[OK] Extracted {crops_extracted} crops in {dt_extract:.2f}s ({extract_rate:.1f} crops/sec)")
    print(f"[RAM] Peak memory after crop extraction: {ram_after_extraction:.1f} MB (Delta: +{ram_after_extraction - baseline_ram_mb:.1f} MB)")

    # 5. Model Architecture Comparison
    print(f"\n--- Model Architecture Benchmarking ---")
    dataset = LUNA16ProfileDataset(crops, labels)
    
    models = {
        "ResNet3D-18-Standard": ResNet3D18(in_channels=1, num_classes=2),
        "ResNet3D-18-Half": ResNet3D18Half(in_channels=1, num_classes=2),
    }

    arch_benchmarks = {}

    for name, model in models.items():
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        bench_loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Warmup
        x_w, y_w = next(iter(bench_loader))
        out_w = model(x_w)
        loss_w = crit(out_w, y_w)
        loss_w.backward()
        opt.step()
        opt.zero_grad()

        # Timed benchmark: 10 batches
        t0_bench = time.time()
        n_bench_samples = 0
        for i, (bx, by) in enumerate(bench_loader):
            if i >= 10:
                break
            t_fwd_0 = time.time()
            out = model(bx)
            loss = crit(out, by)
            t_fwd = time.time() - t_fwd_0

            t_bwd_0 = time.time()
            loss.backward()
            opt.step()
            opt.zero_grad()
            t_bwd = time.time() - t_bwd_0

            n_bench_samples += bx.size(0)

        dt_bench = time.time() - t0_bench
        throughput = n_bench_samples / dt_bench
        ram_now = get_current_ram_mb()

        arch_benchmarks[name] = {
            "parameters": n_params,
            "throughput_crops_per_sec": round(throughput, 1),
            "step_time_ms_b8": round((dt_bench / 10) * 1000, 1),
            "memory_mb": round(ram_now, 1),
        }
        print(f"  {name:22s} | Params: {n_params:>10,d} | Throughput: {throughput:>5.1f} crops/s | Step (B=8): {arch_benchmarks[name]['step_time_ms_b8']} ms")

    # 6. Batch Size Scaling (B=8 vs B=16) on Selected Model (ResNet3D-18-Half)
    print(f"\n--- Batch Size Scaling (ResNet3D-18-Half) ---")
    batch_scaling = {}
    selected_model = ResNet3D18Half(in_channels=1, num_classes=2)

    for bsz in [8, 16]:
        loader = DataLoader(dataset, batch_size=bsz, shuffle=True)
        opt = torch.optim.AdamW(selected_model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        t0_bsz = time.time()
        samples_done = 0
        for i, (bx, by) in enumerate(loader):
            if i >= 8:
                break
            out = selected_model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
            opt.zero_grad()
            samples_done += bx.size(0)

        dt_bsz = time.time() - t0_bsz
        rate = samples_done / dt_bsz
        ram_now = get_current_ram_mb()
        batch_scaling[f"B={bsz}"] = {
            "batch_size": bsz,
            "throughput_crops_per_sec": round(rate, 1),
            "latency_per_batch_ms": round((dt_bsz / 8) * 1000, 1),
            "memory_mb": round(ram_now, 1),
        }
        print(f"  Batch size B={bsz:2d} | Throughput: {rate:>5.1f} crops/s | Latency/batch: {batch_scaling[f'B={bsz}']['latency_per_batch_ms']} ms | Memory: {ram_now:.1f} MB")

    # 7. 1-Epoch Dry Run on 20 Scans Cohort
    print(f"\n--- 1-Epoch Dry Run on 20-Scan Cohort ({len(dataset)} crops) ---")
    train_model = ResNet3D18Half(in_channels=1, num_classes=2)
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Balanced class weights
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr, minlength=2)
    weights = torch.tensor([len(labels_arr) / (2.0 * class_counts[0]), len(labels_arr) / (2.0 * class_counts[1])], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    t0_epoch = time.time()
    epoch_losses = []
    all_preds = []
    all_targets = []

    peak_ram_mb = get_current_ram_mb()

    train_model.train()
    for batch_idx, (volumes, target_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = train_model(volumes)
        loss = criterion(logits, target_labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_targets.extend(target_labels.tolist())

        cur_ram = get_current_ram_mb()
        if cur_ram > peak_ram_mb:
            peak_ram_mb = cur_ram

    dt_epoch = time.time() - t0_epoch
    avg_loss = float(np.mean(epoch_losses))
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    epoch_throughput = len(dataset) / dt_epoch

    print(f"[DRY-RUN DONE] 1 Epoch completed in {dt_epoch:.2f}s ({epoch_throughput:.1f} crops/sec)")
    print(f"               Initial Loss: {epoch_losses[0]:.4f} -> Final Batch Loss: {epoch_losses[-1]:.4f}")
    print(f"               Average Loss: {avg_loss:.4f} | Training Accuracy: {accuracy*100:.1f}%")
    print(f"[RAM] Peak memory observed during training: {peak_ram_mb:.1f} MB ({peak_ram_mb / 1024:.2f} GB)")

    # Assert memory < 12 GB limit
    assert peak_ram_mb < 12288.0, f"OOM Hazard: Peak memory ({peak_ram_mb:.1f} MB) exceeded 12 GB allocation limit!"
    print(f"[ASSERT] Memory usage {peak_ram_mb:.1f} MB is strictly within the 12 GB CPU limit (Headroom: {12288.0 - peak_ram_mb:.1f} MB free).")

    # 8. Projection for Full Train Set (178 Scans)
    # LUNA16 subsets 0, 1: 308 positives
    # At 1:10 ratio: 308 + 3,080 = 3,388 crops
    # At 1:5 ratio: 308 + 1,540 = 1,848 crops
    full_crops_10x = 308 + 308 * 10  # 3,388
    full_crops_5x = 308 + 308 * 5    # 1,848

    projected_sec_10x = full_crops_10x / epoch_throughput
    projected_sec_5x = full_crops_5x / epoch_throughput

    projections = {
        "full_train_scans": 178,
        "full_train_positives": 308,
        "ratio_10x": {
            "total_crops_per_epoch": full_crops_10x,
            "projected_epoch_time_sec": round(projected_sec_10x, 1),
            "projected_epoch_time_min": round(projected_sec_10x / 60.0, 2),
            "projected_10_epochs_min": round((projected_sec_10x * 10) / 60.0, 1),
        },
        "ratio_5x": {
            "total_crops_per_epoch": full_crops_5x,
            "projected_epoch_time_sec": round(projected_sec_5x, 1),
            "projected_epoch_time_min": round(projected_sec_5x / 60.0, 2),
            "projected_10_epochs_min": round((projected_sec_5x * 10) / 60.0, 1),
        },
    }

    print(f"\n--- Full Training (178 Scans) Projections ---")
    print(f"  Ratio 1:10 ({full_crops_10x} crops): {projections['ratio_10x']['projected_epoch_time_sec']}s/epoch ({projections['ratio_10x']['projected_epoch_time_min']} min) -> 10 epochs: {projections['ratio_10x']['projected_10_epochs_min']} min")
    print(f"  Ratio 1:5  ({full_crops_5x} crops): {projections['ratio_5x']['projected_epoch_time_sec']}s/epoch ({projections['ratio_5x']['projected_epoch_time_min']} min) -> 10 epochs: {projections['ratio_5x']['projected_10_epochs_min']} min")

    # 9. Save Results Manifest
    results = {
        "step": 2,
        "title": "CPU Training Profile (3D ResNet-18)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "hardware": {
            "device": "cpu",
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_limit_gb": 12.0,
            "peak_memory_mb": round(peak_ram_mb, 1),
            "peak_memory_gb": round(peak_ram_mb / 1024.0, 3),
            "oom_safe": True,
        },
        "dry_run_cohort": {
            "scans_count": n_scans,
            "positives": n_pos,
            "negatives": n_neg,
            "total_crops": len(dataset),
            "crop_dimensions": [32, 32, 32],
            "spacing_mm": [1.0, 1.0, 1.0],
            "hu_window": [-1000.0, 400.0],
            "extraction_throughput_crops_per_sec": round(extract_rate, 1),
        },
        "model_benchmarks": arch_benchmarks,
        "batch_size_scaling": batch_scaling,
        "dry_run_metrics": {
            "model": "ResNet3D-18-Half",
            "batch_size": 16,
            "epoch_duration_sec": round(dt_epoch, 2),
            "throughput_crops_per_sec": round(epoch_throughput, 1),
            "initial_loss": round(epoch_losses[0], 4),
            "final_loss": round(epoch_losses[-1], 4),
            "average_loss": round(avg_loss, 4),
            "accuracy": round(accuracy, 4),
        },
        "full_dataset_projections": projections,
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "cpu_profile_step2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results written to {out_file}")
    print("=================================================================")
    return results


if __name__ == "__main__":
    run_cpu_profile()
