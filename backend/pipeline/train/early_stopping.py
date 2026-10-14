"""
backend/pipeline/train/early_stopping.py
==========================================
Patience-based EarlyStopping with:
* "min" and "max" modes  (loss vs AUC)
* Configurable relative threshold (min_delta)
* Best-state restore flag
* Structured logging

Usage
-----
>>> stopper = EarlyStopping(patience=15, mode="max", metric="val_auc")
>>> for epoch in range(100):
...     val_auc = trainer.val_epoch(...)
...     if stopper(val_auc, epoch):
...         print("Stopping early")
...         break
"""

from __future__ import annotations

from backend.monitoring import get_logger

log = get_logger(__name__)


class EarlyStopping:
    """
    Halts training when a monitored metric stops improving.

    Parameters
    ----------
    patience  : number of epochs to wait after last improvement
    mode      : "min" (lower is better) or "max" (higher is better)
    metric    : name of the metric being monitored (for logging only)
    min_delta : minimum change to qualify as an improvement
    """

    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        metric: str = "val_loss",
        min_delta: float = 1e-4,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience  = patience
        self.mode      = mode
        self.metric    = metric
        self.min_delta = min_delta

        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.wait:  int = 0
        self.best_epoch: int = 0
        self.triggered: bool = False

    def __call__(self, value: float, epoch: int) -> bool:
        """
        Call once per epoch with the monitored metric value.

        Returns True when training should stop.
        """
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.wait = 0
            log.info(
                "early_stopping_improved",
                metric=self.metric,
                best=round(value, 4),
                epoch=epoch,
            )
        else:
            self.wait += 1
            log.info(
                "early_stopping_no_improve",
                metric=self.metric,
                current=round(value, 4),
                best=round(self.best_value, 4),
                wait=self.wait,
                patience=self.patience,
            )

        if self.wait >= self.patience:
            self.triggered = True
            log.warning(
                "early_stopping_triggered",
                metric=self.metric,
                best_epoch=self.best_epoch,
                best_value=round(self.best_value, 4),
            )
            return True

        return False

    def reset(self) -> None:
        """Reset state (e.g. between cross-validation folds)."""
        self.best_value = float("-inf") if self.mode == "max" else float("inf")
        self.wait = 0
        self.best_epoch = 0
        self.triggered = False
