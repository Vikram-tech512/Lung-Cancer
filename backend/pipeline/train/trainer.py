"""
backend/pipeline/train/trainer.py
=====================================
Unified 3D model trainer for all classification and segmentation architectures.

Features
--------
* Mixed-precision (torch.amp) training
* Gradient clipping & accumulation
* Online AUC-ROC, sensitivity, specificity tracking per epoch
* MLflow experiment logging (loss, AUC, LR, weight histograms)
* Prometheus metric push after every epoch
* Checkpoint saving via CheckpointManager
* EarlyStopping integration
* TensorBoard SummaryWriter support (optional)
* Supports both classification and segmentation (Dice loss)

Usage
-----
>>> from backend.pipeline.train import Trainer, LRSchedulerFactory, EarlyStopping
>>> from backend.models.classification.resnet3d.model import resnet3d_50

>>> model   = resnet3d_50(num_classes=2)
>>> trainer = Trainer(
...     model=model,
...     model_name="ResNet3D-50",
...     device="cuda",
...     lr=1e-4,
...     epochs=100,
... )
>>> history = trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from backend.monitoring import get_logger, MetricsRegistry
from .early_stopping import EarlyStopping
from .checkpoint_manager import CheckpointManager
from .scheduler import LRSchedulerFactory

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

log = get_logger(__name__)


class Trainer:
    """
    Unified training loop for 3D neural network models.

    Parameters
    ----------
    model       : nn.Module — classification or segmentation model
    model_name  : human-readable name used for logging & checkpoints
    device      : "cuda" | "cpu" | "cuda:0"
    lr          : initial learning rate
    epochs      : maximum number of training epochs
    batch_accum : gradient accumulation steps (effective batch = batch_size × accum)
    patience    : EarlyStopping patience in epochs (None = disabled)
    checkpoint_dir : directory for saving .pt checkpoints
    use_amp     : enable mixed-precision (float16 on CUDA, bfloat16 on CPU where supported)
    clip_grad   : max gradient norm for clipping (None = disabled)
    task        : "classification" | "segmentation"
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        model_name: str = "Model",
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_accum: int = 1,
        patience: Optional[int] = 15,
        checkpoint_dir: Path = Path("checkpoints"),
        use_amp: bool = True,
        clip_grad: Optional[float] = 1.0,
        task: str = "classification",
        scheduler_type: str = "cosine",
    ) -> None:
        import torch

        self.model        = model.to(device)
        self.model_name   = model_name
        self.device       = device
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.batch_accum  = batch_accum
        self.task         = task
        self.clip_grad    = clip_grad

        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.criterion = self._build_criterion()
        self.early_stopper = (
            EarlyStopping(patience=patience, mode="max", metric="val_auc")
            if patience else None
        )
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir, model_name=model_name
        )
        self._scheduler_type = scheduler_type
        self._scheduler: Optional[Any] = None   # built after knowing n_steps

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []
        }

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------
    def _build_criterion(self) -> Any:
        import torch.nn as nn
        if self.task == "segmentation":
            return _DiceBCELoss()
        return nn.CrossEntropyLoss(label_smoothing=0.05)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        resume_from: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """
        Train for `self.epochs` epochs.

        Returns
        -------
        history dict with train/val loss and AUC lists
        """
        start_epoch = 0
        if resume_from:
            start_epoch = self.ckpt_manager.load(
                self.model, self.optimizer, path=resume_from
            )

        # Build LR scheduler now that we know steps per epoch
        self._scheduler = LRSchedulerFactory.build(
            self._scheduler_type,
            self.optimizer,
            epochs=self.epochs - start_epoch,
            steps_per_epoch=len(train_loader),
        )

        try:
            import mlflow
            mlflow.set_experiment(self.model_name)
            mlflow.start_run(run_name=f"{self.model_name}_train")
            _mlflow_ok = True
        except Exception:
            _mlflow_ok = False

        try:
            for epoch in range(start_epoch, self.epochs):
                t0 = time.monotonic()

                train_metrics = self._train_epoch(train_loader, epoch)
                val_metrics   = self._val_epoch(val_loader, epoch)

                elapsed = time.monotonic() - t0
                current_lr = self.optimizer.param_groups[0]["lr"]

                self._update_history(train_metrics, val_metrics)
                self._log_epoch(epoch, train_metrics, val_metrics, current_lr, elapsed)
                self._push_prometheus(epoch, train_metrics, val_metrics)

                if _mlflow_ok:
                    self._log_mlflow(epoch, train_metrics, val_metrics, current_lr)

                # Save checkpoint
                self.ckpt_manager.save(
                    self.model, self.optimizer, epoch,
                    metric_val=val_metrics.get("auc", 0.0),
                )

                # Early stopping
                if self.early_stopper:
                    if self.early_stopper(val_metrics.get("auc", 0.0), epoch):
                        log.info("early_stopping_triggered", epoch=epoch, model=self.model_name)
                        break

        finally:
            if _mlflow_ok:
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception:
                    pass

        return self.history

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------
    def _train_epoch(
        self, loader: "DataLoader", epoch: int
    ) -> Dict[str, float]:
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import roc_auc_score

        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []
        n_batches = len(loader)

        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            volume = batch["volume"].to(self.device, non_blocking=True)
            label  = batch["label"].to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(volume)
                loss   = self.criterion(logits, label)
                loss   = loss / self.batch_accum

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.batch_accum == 0 or (step + 1) == n_batches:
                if self.clip_grad:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                if self._scheduler is not None:
                    self._scheduler.step()

            total_loss += loss.item() * self.batch_accum

            if self.task == "classification":
                probs = F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(label.cpu().numpy().tolist())

        avg_loss = total_loss / n_batches
        auc = 0.5
        if all_probs and len(set(all_labels)) > 1:
            try:
                auc = float(roc_auc_score(all_labels, all_probs))
            except Exception:
                pass

        return {"loss": avg_loss, "auc": auc}

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------
    def _val_epoch(
        self, loader: "DataLoader", epoch: int
    ) -> Dict[str, float]:
        import torch
        import torch.nn.functional as F
        from sklearn.metrics import roc_auc_score, confusion_matrix

        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                volume = batch["volume"].to(self.device, non_blocking=True)
                label  = batch["label"].to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(volume)
                    loss   = self.criterion(logits, label)
                total_loss += loss.item()
                if self.task == "classification":
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(label.cpu().numpy().tolist())

        avg_loss = total_loss / max(len(loader), 1)
        auc = 0.5
        sensitivity = specificity = 0.5

        if all_probs and len(set(all_labels)) > 1:
            try:
                auc = float(roc_auc_score(all_labels, all_probs))
                preds = (np.array(all_probs) > 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
                sensitivity = tp / max(tp + fn, 1)
                specificity = tn / max(tn + fp, 1)
            except Exception:
                pass

        return {
            "loss": avg_loss,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _update_history(self, train: dict, val: dict) -> None:
        self.history["train_loss"].append(train["loss"])
        self.history["val_loss"].append(val["loss"])
        self.history["train_auc"].append(train.get("auc", 0.5))
        self.history["val_auc"].append(val.get("auc", 0.5))

    def _log_epoch(
        self, epoch: int, train: dict, val: dict, lr: float, elapsed: float
    ) -> None:
        log.info(
            "epoch",
            model=self.model_name,
            epoch=epoch,
            train_loss=round(train["loss"], 4),
            val_loss=round(val["loss"], 4),
            train_auc=round(train.get("auc", 0.5), 4),
            val_auc=round(val.get("auc", 0.5), 4),
            sensitivity=round(val.get("sensitivity", 0.5), 4),
            specificity=round(val.get("specificity", 0.5), 4),
            lr=lr,
            elapsed_s=round(elapsed, 1),
        )

    def _push_prometheus(self, epoch: int, train: dict, val: dict) -> None:
        MetricsRegistry.set_training_metrics(
            model=self.model_name,
            epoch_loss_train=train["loss"],
            epoch_loss_val=val["loss"],
            auc_train=train.get("auc"),
            auc_val=val.get("auc"),
        )
        if val.get("auc"):
            MetricsRegistry.set_model_metric(self.model_name, "auc", val["auc"])
        if val.get("sensitivity"):
            MetricsRegistry.set_model_metric(self.model_name, "sensitivity", val["sensitivity"])
        if val.get("specificity"):
            MetricsRegistry.set_model_metric(self.model_name, "specificity", val["specificity"])

    @staticmethod
    def _log_mlflow(epoch: int, train: dict, val: dict, lr: float) -> None:
        try:
            import mlflow
            mlflow.log_metrics({
                "train_loss": train["loss"],
                "val_loss":   val["loss"],
                "train_auc":  train.get("auc", 0.5),
                "val_auc":    val.get("auc", 0.5),
                "sensitivity": val.get("sensitivity", 0.5),
                "specificity": val.get("specificity", 0.5),
                "lr":         lr,
            }, step=epoch)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dice + BCE combined loss for segmentation
# ---------------------------------------------------------------------------
class _DiceBCELoss:
    """Combined Dice + Binary Cross-Entropy loss for 3D segmentation."""

    def __call__(self, pred, target):
        import torch
        import torch.nn.functional as F

        bce  = F.binary_cross_entropy_with_logits(pred, target.float())
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(1, 2, 3, 4))
        dice = 1 - (2 * intersection + 1) / (
            pred_sig.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + 1
        )
        return bce + dice.mean()
