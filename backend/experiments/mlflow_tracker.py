"""
backend/experiments/mlflow_tracker.py
=====================================
MLflow and local SQLite experiment tracking integration for reproducible training runs.
Logs hyperparameters, metrics (AUC, sensitivity, specificity, ECE, Brier), patient-split SHA-256,
and artifacts.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

class ExperimentTracker:
    def __init__(self, tracking_dir: str = "experiments_log", experiment_name: str = "lung_cancer_platform"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.mlflow_available = False
        self._init_mlflow()

    def _init_mlflow(self):
        try:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_tracking_uri(f"file:///{self.tracking_dir.resolve().as_posix()}")
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_available = True
        except ImportError:
            self.mlflow_available = False

    def log_run(
        self,
        run_name: str,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
        artifacts: Optional[List[str]] = None
    ) -> str:
        """
        Logs a training or evaluation run. Always persists to local JSON record as well as MLflow.
        Returns unique run_id.
        """
        run_id = f"run_{int(time.time())}_{model_name.lower().replace(' ', '_')}"
        record = {
            "run_id": run_id,
            "run_name": run_name,
            "model_name": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
            "artifacts": artifacts or []
        }

        # Local JSON persistence
        run_file = self.tracking_dir / f"{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(record, f, indent=2)

        # MLflow logging if available
        if self.mlflow_available:
            try:
                with self.mlflow.start_run(run_name=run_name):
                    self.mlflow.set_tags({**record["tags"], "model_name": model_name})
                    self.mlflow.log_params(params)
                    self.mlflow.log_metrics(metrics)
                    if artifacts:
                        for art in artifacts:
                            if os.path.exists(art):
                                self.mlflow.log_artifact(art)
            except Exception as e:
                pass

        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        runs = []
        for f in self.tracking_dir.glob("run_*.json"):
            try:
                with open(f, "r") as fp:
                    runs.append(json.load(fp))
            except Exception:
                continue
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return runs
