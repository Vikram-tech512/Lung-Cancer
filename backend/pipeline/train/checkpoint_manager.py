"""
backend/pipeline/train/checkpoint_manager.py
===============================================
Versioned checkpoint save / load manager.

Features
--------
* Saves full training state: model weights, optimizer state, epoch, best metric
* Maintains a best-model checkpoint  (model_name_best.pt)
* Maintains a rolling history of last N checkpoints
* Integrates with MLflow model registry for production model logging
* SHA-256 integrity hash stored alongside each checkpoint
* Structured logging on every save/load

Usage
-----
>>> mgr = CheckpointManager(checkpoint_dir=Path("checkpoints"), model_name="ResNet3D-50")
>>> mgr.save(model, optimizer, epoch=10, metric_val=0.92)
>>> start_epoch = mgr.load(model, optimizer)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from backend.monitoring import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger(__name__)


class CheckpointManager:
    """
    Versioned model checkpoint save / restore with integrity hashing.

    Parameters
    ----------
    checkpoint_dir : directory to store checkpoints
    model_name     : prefix for checkpoint filenames
    keep_last_n    : rolling window of saved checkpoints (best is always kept)
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        model_name: str = "model",
        keep_last_n: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name     = model_name.replace(" ", "_")
        self.keep_last_n    = keep_last_n
        self.best_metric    = float("-inf")
        self._saved:  list[Path] = []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(
        self,
        model:     "torch.nn.Module",
        optimizer: "torch.optim.Optimizer",
        epoch:     int,
        metric_val: float = 0.0,
        extra:     Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a versioned checkpoint.

        Automatically updates the *_best.pt symlink when metric_val improves.
        Returns the path of the saved file.
        """
        import torch

        state = {
            "epoch":             epoch,
            "model_name":        self.model_name,
            "metric_val":        metric_val,
            "model_state_dict":  model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if extra:
            state.update(extra)

        fname = self.checkpoint_dir / f"{self.model_name}_epoch{epoch:04d}.pt"
        torch.save(state, fname)
        self._saved.append(fname)

        # Compute SHA-256 and write sidecar
        sha = self._sha256(fname)
        meta = {"epoch": epoch, "metric_val": metric_val, "sha256": sha}
        fname.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        # Update best
        if metric_val > self.best_metric:
            self.best_metric = metric_val
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(state, best_path)
            log.info(
                "checkpoint_best_saved",
                model=self.model_name,
                epoch=epoch,
                metric=round(metric_val, 4),
                path=str(best_path),
            )
            self._register_mlflow(best_path, epoch, metric_val)

        # Rolling cleanup
        self._cleanup()

        log.info(
            "checkpoint_saved",
            model=self.model_name,
            epoch=epoch,
            metric=round(metric_val, 4),
            path=str(fname),
        )
        return fname

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(
        self,
        model:     "torch.nn.Module",
        optimizer: Optional["torch.optim.Optimizer"] = None,
        path:      Optional[Path] = None,
        strict:    bool = False,
    ) -> int:
        """
        Load checkpoint from *path* (or auto-detect best checkpoint).

        Returns the starting epoch (epoch + 1).
        """
        import torch

        if path is None:
            path = self.checkpoint_dir / f"{self.model_name}_best.pt"

        if not Path(path).exists():
            log.warning("checkpoint_not_found", path=str(path))
            return 0

        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=strict)
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        epoch = int(state.get("epoch", 0))
        metric = float(state.get("metric_val", 0.0))
        self.best_metric = metric

        log.info(
            "checkpoint_loaded",
            model=self.model_name,
            epoch=epoch,
            metric=round(metric, 4),
            path=str(path),
        )
        return epoch + 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cleanup(self) -> None:
        """Remove oldest checkpoints beyond keep_last_n."""
        while len(self._saved) > self.keep_last_n:
            old = self._saved.pop(0)
            try:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
                log.info("checkpoint_pruned", path=str(old))
            except Exception as exc:
                log.warning("checkpoint_prune_failed", path=str(old), error=str(exc))

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _register_mlflow(self, path: Path, epoch: int, metric: float) -> None:
        try:
            import mlflow
            mlflow.log_artifact(str(path))
            mlflow.log_metric("best_val_metric", metric, step=epoch)
        except Exception:
            pass
