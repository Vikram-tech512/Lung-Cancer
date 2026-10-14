"""
backend/pipeline/train/scheduler.py
=======================================
Learning Rate Scheduler Factory.

Supported schedulers
--------------------
* cosine       — CosineAnnealingLR with warm-restarts
* step         — StepLR (decay by gamma every N epochs)
* onecycle     — OneCycleLR (super-convergence)
* plateau      — ReduceLROnPlateau (val-loss driven)
* warmup_cosine— Linear warmup then cosine decay (Transformer standard)
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class LRSchedulerFactory:
    """Factory for building PyTorch LR schedulers."""

    @staticmethod
    def build(
        scheduler_type: str,
        optimizer: "torch.optim.Optimizer",
        epochs: int = 100,
        steps_per_epoch: int = 100,
        warmup_epochs: int = 5,
        min_lr: float = 1e-7,
        step_size: int = 30,
        gamma: float = 0.1,
        max_lr_factor: float = 10.0,
    ) -> Optional[Any]:
        """
        Build and return a scheduler.

        Parameters
        ----------
        scheduler_type   : one of "cosine", "step", "onecycle", "plateau", "warmup_cosine"
        optimizer        : the optimizer to attach the scheduler to
        epochs           : total number of epochs
        steps_per_epoch  : number of optimizer steps per epoch
        warmup_epochs    : linear warmup length for "warmup_cosine"
        min_lr           : minimum learning rate
        step_size        : StepLR step size in epochs
        gamma            : StepLR decay multiplier
        max_lr_factor    : OneCycleLR peak = base_lr × factor
        """
        try:
            import torch.optim.lr_scheduler as sched
        except ImportError:
            return None

        base_lr = optimizer.param_groups[0]["lr"]
        total_steps = epochs * steps_per_epoch

        if scheduler_type == "cosine":
            return sched.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, epochs // 3),
                T_mult=2,
                eta_min=min_lr,
            )

        if scheduler_type == "step":
            return sched.StepLR(optimizer, step_size=step_size, gamma=gamma)

        if scheduler_type == "onecycle":
            return sched.OneCycleLR(
                optimizer,
                max_lr=base_lr * max_lr_factor,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e4,
            )

        if scheduler_type == "plateau":
            return sched.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                threshold=1e-4,
                min_lr=min_lr,
            )

        if scheduler_type == "warmup_cosine":
            warmup_steps = warmup_epochs * steps_per_epoch

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
                import math
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_lr / base_lr, cosine_decay)

            return sched.LambdaLR(optimizer, lr_lambda=lr_lambda)

        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}'. "
            f"Choose from: cosine, step, onecycle, plateau, warmup_cosine"
        )
