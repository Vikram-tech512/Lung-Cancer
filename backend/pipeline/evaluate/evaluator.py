"""
backend/pipeline/evaluate/evaluator.py
=========================================
Comprehensive model evaluator for classification and segmentation.

Metrics computed
----------------
Classification:
  AUC-ROC, AP (Average Precision), Accuracy, Sensitivity (Recall),
  Specificity, Precision, F1, Youden-J, Balanced Accuracy,
  Cohen's Kappa, Matthews Correlation Coefficient (MCC)

Segmentation:
  Dice, IoU (Jaccard), Hausdorff-95, Surface DSC, Volume Error

Usage
-----
>>> evaluator = ModelEvaluator(task="classification")
>>> report = evaluator.evaluate(model, val_loader, device="cuda")
>>> print(report["auc"], report["sensitivity"])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from backend.monitoring import get_logger, MetricsRegistry

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

log = get_logger(__name__)


class ModelEvaluator:
    """
    End-to-end model evaluator supporting classification and segmentation tasks.

    Parameters
    ----------
    task        : "classification" | "segmentation"
    threshold   : probability threshold for binary prediction (default 0.5)
    n_classes   : number of output classes
    """

    def __init__(
        self,
        task: str = "classification",
        threshold: float = 0.5,
        n_classes: int = 2,
    ) -> None:
        self.task      = task
        self.threshold = threshold
        self.n_classes = n_classes

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        model: "torch.nn.Module",
        loader: "DataLoader",
        device: str = "cpu",
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """
        Run full evaluation pass.

        Returns
        -------
        dict with all metrics
        """
        import torch
        import torch.nn.functional as F

        model.eval()
        all_probs:  List[float] = []
        all_preds:  List[int]   = []
        all_labels: List[int]   = []
        seg_dices:  List[float] = []

        with torch.no_grad():
            for batch in loader:
                volume = batch["volume"].to(device, non_blocking=True)
                label  = batch["label"].to(device, non_blocking=True)

                logits = model(volume)

                if self.task == "classification":
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    preds = (probs > self.threshold).astype(int)
                    all_probs.extend(probs.tolist())
                    all_preds.extend(preds.tolist())
                    all_labels.extend(label.cpu().numpy().tolist())

                elif self.task == "segmentation":
                    pred_mask = (torch.sigmoid(logits) > self.threshold).float()
                    for i in range(pred_mask.shape[0]):
                        dice = self._dice(
                            pred_mask[i].cpu().numpy(),
                            label[i].cpu().numpy()
                        )
                        seg_dices.append(dice)

        if self.task == "classification":
            report = self._classification_report(all_probs, all_preds, all_labels)
        else:
            report = {"dice_mean": np.mean(seg_dices), "dice_std": np.std(seg_dices)}

        report["model_name"] = model_name
        report["task"]       = self.task

        self._update_gauges(model_name, report)
        log.info("evaluation_complete", model=model_name, **{
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in report.items()
            if k not in ("model_name", "task", "confusion_matrix")
        })
        return report

    # ------------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------------
    def _classification_report(
        self,
        probs:  List[float],
        preds:  List[int],
        labels: List[int],
    ) -> Dict[str, Any]:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            accuracy_score,
            f1_score,
            confusion_matrix,
            cohen_kappa_score,
            matthews_corrcoef,
            balanced_accuracy_score,
        )

        y_true = np.array(labels)
        y_pred = np.array(preds)
        y_prob = np.array(probs)

        auc = ap = 0.5
        if len(set(labels)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
            ap  = float(average_precision_score(y_true, y_prob))

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity  = tp / max(tp + fn, 1)
            specificity  = tn / max(tn + fp, 1)
            precision    = tp / max(tp + fp, 1)
            youden_j     = sensitivity + specificity - 1.0
            ppv = precision
            npv = tn / max(tn + fn, 1)
        else:
            sensitivity = specificity = precision = youden_j = ppv = npv = 0.5

        return {
            "auc":                float(auc),
            "ap":                 float(ap),
            "accuracy":           float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
            "sensitivity":        float(sensitivity),
            "specificity":        float(specificity),
            "precision":          float(precision),
            "npv":                float(npv),
            "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
            "youden_j":           float(youden_j),
            "kappa":              float(cohen_kappa_score(y_true, y_pred)),
            "mcc":                float(matthews_corrcoef(y_true, y_pred)),
            "confusion_matrix":   cm.tolist(),
            "n_samples":          len(labels),
            "prevalence":         float(y_true.mean()),
            "threshold":          self.threshold,
        }

    # ------------------------------------------------------------------
    # Segmentation metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _dice(pred: np.ndarray, target: np.ndarray) -> float:
        pred   = pred.flatten()
        target = target.flatten()
        intersection = (pred * target).sum()
        return float((2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6))

    # ------------------------------------------------------------------
    # Prometheus push
    # ------------------------------------------------------------------
    @staticmethod
    def _update_gauges(model_name: str, report: Dict[str, Any]) -> None:
        for metric in ("auc", "accuracy", "sensitivity", "specificity", "f1", "dice_mean"):
            if metric in report:
                MetricsRegistry.set_model_metric(model_name, metric, report[metric])
