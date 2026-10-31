"""
backend/monitoring/metering.py
===============================
Per-Scan Compute Metering, Stage Timers, Resource Profiling & Cost Tracking.

Measures:
- Wall time per stage and total end-to-end latency
- Process CPU-seconds consumed
- Peak RAM consumption in MB (via psutil / resource)
- Total input voxels (D × H × W)
- Number of candidate crops evaluated
- Swin3D / ViT patch tokens processed
- Estimated compute cost ($) via configurable CPU/GPU hourly pricing
- LLM token accounting (or explicitly "no LLM tokens used")
- Visible time-budget adherence and SLA violations
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from backend.monitoring import get_logger, MetricsRegistry

log = get_logger(__name__)

# Configurable cost rates ($ per hour)
DEFAULT_CPU_COST_PER_HOUR = float(os.getenv("COMPUTE_COST_PER_CPU_HOUR", "0.048"))  # e.g., standard 4-vCPU rate
DEFAULT_GPU_COST_PER_HOUR = float(os.getenv("COMPUTE_COST_PER_GPU_HOUR", "0.520"))  # e.g., T4/A10G instance
DEFAULT_TRIAGE_BUDGET_SECONDS = float(os.getenv("TRIAGE_TIME_BUDGET_SECONDS", "15.0"))


@dataclass
class ScanUsageSummary:
    study_id: str
    mode: str  # "fast_triage" | "full"
    total_wall_time_ms: float
    total_cpu_seconds: float
    peak_memory_mb: float
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    input_voxels: int = 0
    candidate_crops_processed: int = 0
    swin_patch_tokens_processed: int = 0
    compute_cost_usd: float = 0.0
    cost_basis: str = ""
    llm_tokens_used: str = "no LLM tokens used"
    exceeded_time_budget: bool = False
    time_budget_ms: float = 15000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_id": self.study_id,
            "mode": self.mode,
            "total_wall_time_ms": round(self.total_wall_time_ms, 2),
            "total_wall_time_s": round(self.total_wall_time_ms / 1000.0, 3),
            "total_cpu_seconds": round(self.total_cpu_seconds, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "stage_timings_ms": {k: round(v, 2) for k, v in self.stage_timings_ms.items()},
            "input_voxels": self.input_voxels,
            "candidate_crops_processed": self.candidate_crops_processed,
            "swin_patch_tokens_processed": self.swin_patch_tokens_processed,
            "compute_cost_usd": round(self.compute_cost_usd, 5),
            "cost_basis": self.cost_basis,
            "llm_tokens_used": self.llm_tokens_used,
            "exceeded_time_budget": self.exceeded_time_budget,
            "time_budget_ms": round(self.time_budget_ms, 1),
            "time_budget_s": round(self.time_budget_ms / 1000.0, 1),
        }


class ScanMeter:
    """Context tracker for profiling a single study analysis run."""

    def __init__(
        self,
        study_id: str,
        mode: str = "fast_triage",
        time_budget_s: Optional[float] = None,
        cpu_cost_per_hour: float = DEFAULT_CPU_COST_PER_HOUR,
    ) -> None:
        self.study_id = study_id
        self.mode = mode
        self.time_budget_ms = (time_budget_s or DEFAULT_TRIAGE_BUDGET_SECONDS) * 1000.0
        self.cpu_cost_per_hour = cpu_cost_per_hour

        self._start_wall = time.perf_counter()
        self._start_cpu = time.process_time()

        self._stage_timings: Dict[str, float] = {}
        self._current_stage_name: Optional[str] = None
        self._current_stage_start: float = 0.0

        self.input_voxels = 0
        self.candidate_crops = 0
        self.swin_tokens = 0
        self.llm_accounting = "no LLM tokens used"

        self._initial_mem_mb = self._get_memory_mb()
        self._peak_mem_mb = self._initial_mem_mb

    def _get_memory_mb(self) -> float:
        if _HAS_PSUTIL:
            try:
                proc = psutil.Process()
                return proc.memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                pass
        return 0.0

    def start_stage(self, stage_name: str) -> None:
        """Begin measuring a pipeline stage."""
        if self._current_stage_name:
            self.end_stage(self._current_stage_name)
        self._current_stage_name = stage_name
        self._current_stage_start = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        """Finish measuring a pipeline stage and return its duration in ms."""
        now = time.perf_counter()
        elapsed_ms = (now - self._current_stage_start) * 1000.0
        self._stage_timings[stage_name] = elapsed_ms
        self._current_stage_name = None

        # Track peak memory
        current_mem = self._get_memory_mb()
        if current_mem > self._peak_mem_mb:
            self._peak_mem_mb = current_mem

        return elapsed_ms

    def record_input_voxels(self, count: int) -> None:
        self.input_voxels = max(self.input_voxels, int(count))

    def record_candidate_crops(self, count: int) -> None:
        self.candidate_crops += int(count)

    def record_swin_tokens(self, count: int) -> None:
        self.swin_tokens += int(count)

    def record_llm_usage(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        cost = (prompt_tokens * 0.0000015) + (completion_tokens * 0.000002)
        self.llm_accounting = f"{model}: {prompt_tokens} prompt + {completion_tokens} completion tokens (${cost:.5f})"

    def finalize(self) -> ScanUsageSummary:
        """Produce the final usage summary."""
        if self._current_stage_name:
            self.end_stage(self._current_stage_name)

        total_wall_ms = (time.perf_counter() - self._start_wall) * 1000.0
        total_cpu_s = max(0.001, time.process_time() - self._start_cpu)

        # Update peak memory check
        current_mem = self._get_memory_mb()
        peak_mb = max(self._peak_mem_mb, current_mem)

        # Compute cost estimation
        # CPU compute: (total_cpu_s / 3600.0) * cost_per_hour
        cpu_hours = total_cpu_s / 3600.0
        compute_cost = cpu_hours * self.cpu_cost_per_hour

        exceeded = total_wall_ms > self.time_budget_ms
        basis = f"${self.cpu_cost_per_hour:.3f}/CPU-hr ({total_cpu_s:.2f}s CPU consumed)"

        summary = ScanUsageSummary(
            study_id=self.study_id,
            mode=self.mode,
            total_wall_time_ms=total_wall_ms,
            total_cpu_seconds=total_cpu_s,
            peak_memory_mb=peak_mb,
            stage_timings_ms=self._stage_timings,
            input_voxels=self.input_voxels,
            candidate_crops_processed=self.candidate_crops,
            swin_patch_tokens_processed=self.swin_tokens,
            compute_cost_usd=compute_cost,
            cost_basis=basis,
            llm_tokens_used=self.llm_accounting,
            exceeded_time_budget=exceeded,
            time_budget_ms=self.time_budget_ms,
        )

        log.info(
            "scan_metering_finalized",
            study_id=self.study_id,
            mode=self.mode,
            wall_ms=round(total_wall_ms, 1),
            cpu_s=round(total_cpu_s, 2),
            cost_usd=round(compute_cost, 5),
            exceeded_budget=exceeded,
        )

        return summary

