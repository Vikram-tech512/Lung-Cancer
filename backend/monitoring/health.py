"""
backend/monitoring/health.py
==============================
Liveness & readiness probe utilities.

FastAPI integration
-------------------
Mount at /health/live and /health/ready (already done in api/service.py).

HealthChecker.run_all() returns a dict suitable for direct JSON response.
"""

from __future__ import annotations

import asyncio
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import get_logger

log = get_logger(__name__)


class HealthChecker:
    """
    Aggregates health checks from all subsystems.

    Usage
    -----
    >>> checker = HealthChecker()
    >>> report = await checker.run_all()
    >>> # {"status": "healthy", "checks": {...}, "uptime_s": 42.1}
    """

    _START_TIME: float = time.monotonic()

    def __init__(self) -> None:
        self._checks: list[tuple[str, Any]] = []
        self._register_defaults()

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def _register_defaults(self) -> None:
        self._checks = [
            ("disk_space",    self._check_disk_space),
            ("database",      self._check_database),
            ("checkpoints",   self._check_checkpoints),
            ("gpu",           self._check_gpu),
            ("mlflow",        self._check_mlflow),
        ]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    @staticmethod
    async def _check_disk_space() -> Dict[str, Any]:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024 ** 3)
        status = "ok" if free_gb > 2.0 else "warning"
        return {"status": status, "free_gb": round(free_gb, 2)}

    @staticmethod
    async def _check_database() -> Dict[str, Any]:
        db_url = os.getenv("DATABASE_URL", "sqlite:///./lung_platform.db")
        try:
            db_path = db_url.replace("sqlite:///", "").lstrip(".")
            db_file = Path(db_path)
            if db_file.exists() or "sqlite" not in db_url:
                return {"status": "ok", "engine": db_url.split(":")[0]}
            return {"status": "warning", "detail": "DB file missing — will be created on first request"}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    @staticmethod
    async def _check_checkpoints() -> Dict[str, Any]:
        ckpt_dir = Path(os.getenv("CHECKPOINT_DIR", "./checkpoints"))
        files = list(ckpt_dir.glob("**/*.pt")) + list(ckpt_dir.glob("**/*.pth"))
        return {
            "status": "ok" if files else "warning",
            "checkpoint_count": len(files),
            "checkpoint_dir": str(ckpt_dir),
        }

    @staticmethod
    async def _check_gpu() -> Dict[str, Any]:
        try:
            import torch
            if torch.cuda.is_available():
                gpus = [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_gb": round(
                            torch.cuda.get_device_properties(i).total_memory / 1e9, 2
                        ),
                    }
                    for i in range(torch.cuda.device_count())
                ]
                return {"status": "ok", "gpus": gpus}
            return {"status": "cpu_only", "gpus": []}
        except ImportError:
            return {"status": "torch_not_installed"}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    @staticmethod
    async def _check_mlflow() -> Dict[str, Any]:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./experiments_log")
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            return {"status": "ok", "experiment_count": len(experiments)}
        except ImportError:
            return {"status": "mlflow_not_installed"}
        except Exception as exc:
            return {"status": "warning", "detail": str(exc)}

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------
    async def run_all(self, timeout_s: float = 5.0) -> Dict[str, Any]:
        """
        Execute all health checks concurrently with *timeout_s* deadline.
        Returns a structured dict:
        {
            "status": "healthy" | "degraded" | "unhealthy",
            "uptime_s": float,
            "platform": str,
            "checks": { "<name>": {...} }
        }
        """
        uptime = time.monotonic() - HealthChecker._START_TIME

        async def _run_one(name: str, fn) -> tuple[str, dict]:
            try:
                result = await asyncio.wait_for(fn(), timeout=timeout_s)
                return name, result
            except asyncio.TimeoutError:
                return name, {"status": "timeout"}
            except Exception as exc:
                return name, {"status": "error", "detail": str(exc)}

        tasks = [_run_one(name, fn) for name, fn in self._checks]
        results = await asyncio.gather(*tasks)
        checks = dict(results)

        # Derive overall status
        statuses = {v.get("status", "unknown") for v in checks.values()}
        if "error" in statuses or "unhealthy" in statuses:
            overall = "unhealthy"
        elif "warning" in statuses or "timeout" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        report = {
            "status": overall,
            "uptime_s": round(uptime, 2),
            "platform": platform.platform(),
            "checks": checks,
        }
        log.info("health_check_completed", status=overall, uptime_s=report["uptime_s"])
        return report

    # ------------------------------------------------------------------
    # Liveness (lightweight — never runs slow checks)
    # ------------------------------------------------------------------
    async def liveness(self) -> Dict[str, Any]:
        return {
            "status": "alive",
            "uptime_s": round(time.monotonic() - HealthChecker._START_TIME, 2),
        }
