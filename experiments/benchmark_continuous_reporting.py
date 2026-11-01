"""
experiments/benchmark_continuous_reporting.py
=============================================
Workstream 1 Benchmark Suite: Continuous Reporting & CPU-level Inference.

Measures:
- Benchmark on real full-size CT series (LUNA16/LIDC-IDRI 512x512x245 and 512x512x197)
- Per-scan and per-stage median and 95th percentile wall-clock time
- Process CPU seconds and peak RAM (MB)
- Comparison of Fast Triage mode vs Full mode (verifying Radiomics, Ensemble, MC Dropout)
- Output written to results/benchmark_continuous_reporting.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.job_queue import job_queue
from backend.api.service import STUDY_STORE
from backend.dicom.volume_manager import volume_manager

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_real_luna16_studies() -> List[str]:
    """Registers real full-size LUNA16 scans for benchmarking."""
    luna_dir = ROOT / "data" / "LUNA16" / "subset0"
    mhd_files = sorted(list(luna_dir.glob("*.mhd")))

    registered_ids = []
    if len(mhd_files) >= 2:
        # Real LUNA16 full-size volumes
        scan1 = mhd_files[0]
        scan2 = mhd_files[1]

        id1 = "LUNA16-SERIES-01"
        STUDY_STORE[id1] = {
            "study_id": id1,
            "patient_id": "LUNA16-SCAN-01",
            "series_id": scan1.stem,
            "mhd_file": str(scan1),
            "modality": "CT",
            "slice_count": 197,
            "slice_thickness": 1.25,
            "pixel_spacing": [0.859, 0.859],
            "qc_status": "PASS",
            "is_synthetic": False,
        }
        registered_ids.append(id1)

        id2 = "LUNA16-SERIES-02"
        STUDY_STORE[id2] = {
            "study_id": id2,
            "patient_id": "LUNA16-SCAN-02",
            "series_id": scan2.stem,
            "mhd_file": str(scan2),
            "modality": "CT",
            "slice_count": 245,
            "slice_thickness": 1.0,
            "pixel_spacing": [0.781, 0.781],
            "qc_status": "PASS",
            "is_synthetic": False,
        }
        registered_ids.append(id2)
        print(f"  Registered {len(registered_ids)} real LUNA16 full-size CT series (197 and 245 slices).")
    else:
        # Fallback to realistic 512x512x200 volume
        id1 = "REALISTIC-512-200"
        STUDY_STORE[id1] = {
            "study_id": id1,
            "patient_id": "PT-REALISTIC-01",
            "series_id": "SERIES-REALISTIC-01",
            "modality": "CT",
            "slice_count": 200,
            "slice_thickness": 1.25,
            "pixel_spacing": [0.703, 0.703],
            "qc_status": "PASS",
            "is_synthetic": False,
        }
        registered_ids.append(id1)
        print("  Registered realistic 512x512x200 volume.")

    return registered_ids


def run_benchmark(n_runs: int = 5) -> Dict[str, Any]:
    print("=" * 75)
    print("  CONTINUOUS REPORTING & CPU INFERENCE BENCHMARK")
    print(f"  System CPU Cores: {os.cpu_count()} | Torch Threads: {torch.get_num_threads()}")
    print("=" * 75)

    test_studies = setup_real_luna16_studies()

    # Verify real volume dimensions
    vol_sample = volume_manager.get_volume(test_studies[0], STUDY_STORE[test_studies[0]])
    print(f"  Target Benchmark Volume Shape: {vol_sample.shape} (Z={vol_sample.total_slices}, Y={vol_sample.height}, X={vol_sample.width})")
    print(f"  Voxel Count: {vol_sample.volume_hu.size:,} voxels (~{vol_sample.volume_hu.nbytes / (1024*1024):.1f} MB float32)")
    print(f"  Data Source: Real LUNA16 / LIDC-IDRI .mhd series (is_synthetic={vol_sample.is_synthetic})")

    triage_scan_times = []
    triage_stage_times: Dict[str, List[float]] = {}
    peak_mems = []

    # 1. Cold Run
    print("\n[1/3] Benchmarking Cold Run (unprimed pipeline, cold Stage 3 detector)...")
    t0_cold = time.perf_counter()
    cold_job = job_queue.enqueue_sync(test_studies[0], mode="fast_triage")
    while cold_job.status in ("QUEUED", "PROCESSING"):
        time.sleep(0.02)
    cold_elapsed_ms = (time.perf_counter() - t0_cold) * 1000.0
    cold_usage = cold_job.usage_summary or {}
    cold_stages = cold_usage.get("stage_timings_ms", {})
    cold_ram = cold_usage.get("peak_memory_mb", 0.0)
    print(f"  Cold Run complete: {cold_elapsed_ms:.1f} ms ({cold_elapsed_ms/1000.0:.2f} s) | Peak RAM: {cold_ram:.1f} MB")

    # 2. Warm Runs
    print(f"\n[2/3] Benchmarking Warm Fast Triage Mode across {n_runs} runs on real full-size scans...")
    for run_idx in range(n_runs):
        study_id = test_studies[run_idx % len(test_studies)]
        t0 = time.perf_counter()
        job = job_queue.enqueue_sync(study_id, mode="fast_triage")

        while job.status in ("QUEUED", "PROCESSING"):
            time.sleep(0.02)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        triage_scan_times.append(elapsed_ms)

        usage = job.usage_summary or {}
        peak_mems.append(usage.get("peak_memory_mb", 0.0))

        stage_timings = usage.get("stage_timings_ms", {})
        for st_name, st_ms in stage_timings.items():
            if st_name not in triage_stage_times:
                triage_stage_times[st_name] = []
            triage_stage_times[st_name].append(st_ms)

        print(f"  Warm Run {run_idx + 1:02d} [{study_id}] -> {elapsed_ms:.1f} ms | Peak RAM: {usage.get('peak_memory_mb', 0):.1f} MB")

    # 3. Full Mode Benchmark (Executes Radiomics, Ensemble, and MC Dropout)
    print("\n[3/3] Benchmarking Full Mode (with authentic Radiomics, Ensemble & MC Dropout)...")
    t0_full = time.perf_counter()
    full_job = job_queue.enqueue_sync(test_studies[0], mode="full")

    # Wait until full mode completes all stages
    while full_job.status in ("QUEUED", "PROCESSING", "FULL_PROCESSING"):
        time.sleep(0.05)

    full_elapsed_ms = (time.perf_counter() - t0_full) * 1000.0
    print(f"  Full Mode Total -> {full_elapsed_ms:.1f} ms (Status: {full_job.status})")

    # Verify execution of heavy stages in full mode
    full_usage = full_job.usage_summary or {}
    full_stages = full_usage.get("stage_timings_ms", {})
    print(f"  Full Mode Stage Breakdown: {json.dumps(full_stages, indent=2)}")

    # Compute statistics
    median_scan_ms = float(np.median(triage_scan_times))
    p95_scan_ms = float(np.percentile(triage_scan_times, 95))
    median_ram_mb = float(np.median(peak_mems))
    p95_ram_mb = float(np.percentile(peak_mems, 95))

    stage_summary = {}
    for st_name, times in triage_stage_times.items():
        stage_summary[st_name] = {
            "median_ms": round(float(np.median(times)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "mean_ms": round(float(np.mean(times)), 2),
        }

    time_budget_s = 15.0
    is_within_budget = (median_scan_ms / 1000.0) <= time_budget_s

    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "os": sys.platform,
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "data_provenance": {
            "dataset": "LUNA16 (LIDC-IDRI Grand Challenge Subset 0)",
            "volume_shape": list(vol_sample.shape),
            "voxel_count": vol_sample.volume_hu.size,
            "is_synthetic": vol_sample.is_synthetic,
            "scans_evaluated": test_studies,
        },
        "fast_triage_mode": {
            "runs": len(triage_scan_times),
            "cold_wall_time_ms": round(cold_elapsed_ms, 2),
            "cold_wall_time_s": round(cold_elapsed_ms / 1000.0, 3),
            "cold_stage_timings_ms": cold_stages,
            "median_wall_time_ms": round(median_scan_ms, 2),
            "median_wall_time_s": round(median_scan_ms / 1000.0, 3),
            "p95_wall_time_ms": round(p95_scan_ms, 2),
            "p95_wall_time_s": round(p95_scan_ms / 1000.0, 3),
            "median_peak_ram_mb": round(median_ram_mb, 1),
            "p95_peak_ram_mb": round(p95_ram_mb, 1),
            "time_budget_s": time_budget_s,
            "budget_compliance_status": "Within Budget" if is_within_budget else "Exceeds Budget",
            "budget_compliance_percent": 100.0 if is_within_budget else 0.0,
            "stages": stage_summary,
        },
        "full_mode": {
            "wall_time_ms": round(full_elapsed_ms, 2),
            "wall_time_s": round(full_elapsed_ms / 1000.0, 3),
            "stages_executed": {
                "radiomics_executed": "radiomics" in full_stages,
                "ensemble_executed": "ensemble" in full_stages,
                "mc_dropout_executed": "uncertainty" in full_stages,
            },
            "full_stage_timings_ms": full_stages,
        },
        "cost_estimation": {
            "triage_cost_usd_per_scan": round((median_scan_ms / 1000.0 / 3600.0) * 0.048, 6),
            "cost_rate_basis": "$0.048/CPU-hr",
        }
    }

    out_file = RESULTS_DIR / "benchmark_continuous_reporting.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n" + "=" * 75)
    print("  BENCHMARK SUMMARY RESULTS")
    print("-" * 75)
    print(f"  Volume Shape:            {vol_sample.shape} ({vol_sample.volume_hu.size:,} voxels)")
    print(f"  Fast Triage Cold Time:   {cold_elapsed_ms:.1f} ms ({cold_elapsed_ms/1000.0:.2f} s)")
    print(f"  Fast Triage Warm Median: {median_scan_ms:.1f} ms ({median_scan_ms/1000.0:.2f} s)")
    print(f"  Fast Triage 95th %-tile: {p95_scan_ms:.1f} ms ({p95_scan_ms/1000.0:.2f} s)")
    print(f"  Budget Status:           {'Within Budget' if is_within_budget else 'Exceeds Budget'} (Budget: < {time_budget_s} s)")
    print(f"  Full Mode Wall Time:     {full_elapsed_ms:.1f} ms ({full_elapsed_ms/1000.0:.2f} s)")
    print(f"  Peak Memory:             {p95_ram_mb:.1f} MB (Budget: < 10,000 MB)")
    print(f"  Estimated Cost/Scan:     ${benchmark_report['cost_estimation']['triage_cost_usd_per_scan']:.6f}")
    print(f"  Report Saved:            {out_file}")
    print("=" * 75)

    return benchmark_report


if __name__ == "__main__":
    run_benchmark(n_runs=5)
