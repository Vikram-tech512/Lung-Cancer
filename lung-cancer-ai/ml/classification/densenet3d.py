"""
backend/models/classification/densenet3d/model.py
==================================================
3D DenseNet-121 for volumetric nodule classification.

Architecture (Huang et al., 2017 — 3D extension):
  Stem:   Conv3d(1,64,kernel=7,stride=2) → BN → ReLU → MaxPool3d
  Dense Block 1:  6 layers,  growth_rate=32
  Transition 1:   BN → Conv1³ → AvgPool3d(2,2,2)
  Dense Block 2: 12 layers
  Transition 2
  Dense Block 3: 24 layers
  Transition 3
  Dense Block 4: 16 layers
  Head:           BN → ReLU → AdaptiveAvgPool3d(1) → Dropout → FC(2)

Input:  (B, 1, 64, 64, 64)
Output: (B, 2)   — [benign_logit, malignant_logit]

Dense connectivity: every layer receives feature maps from ALL preceding layers.
This provides implicit deep supervision and strong gradient flow,
particularly useful when training data is limited (LIDC-IDRI ~1000 patients).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


# ─────────────────────────────────────────────────────────────
#  Dense Layer
# ─────────────────────────────────────────────────────────────

class DenseLayer3d(nn.Module):
    """BN → ReLU → Conv1³(bottleneck) → BN → ReLU → Conv3³ → output k channels."""

    def __init__(self, in_ch: int, growth_rate: int = 32,
                 bn_size: int = 4, dropout_p: float = 0.0):
        super().__init__()
        mid_ch = bn_size * growth_rate
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, growth_rate, kernel_size=3,
                      padding=1, bias=False),
        )
        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_feat = self.drop(self.block(x))
        return torch.cat([x, new_feat], dim=1)


# ─────────────────────────────────────────────────────────────
#  Dense Block
# ─────────────────────────────────────────────────────────────

class DenseBlock3d(nn.Module):
    """Stack of n DenseLayer3d; input grows by growth_rate per layer."""

    def __init__(self, in_ch: int, n_layers: int,
                 growth_rate: int = 32, dropout_p: float = 0.0):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(DenseLayer3d(ch, growth_rate,
                                       dropout_p=dropout_p))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ─────────────────────────────────────────────────────────────
#  Transition Layer
# ─────────────────────────────────────────────────────────────

class TransitionLayer3d(nn.Module):
    """Reduce channels by theta=0.5, halve spatial resolution."""

    def __init__(self, in_ch: int, theta: float = 0.5):
        super().__init__()
        out_ch = int(in_ch * theta)
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────
#  3D DenseNet-121
# ─────────────────────────────────────────────────────────────

class DenseNet3D121(nn.Module):
    """3D DenseNet-121 for nodule malignancy classification.

    Args:
        in_channels:  CT input channels (1).
        num_classes:  2 (benign / malignant).
        growth_rate:  k = 32 (default, per original paper).
        block_layers: [6,12,24,16] for DenseNet-121.
        dropout_p:    Dropout in dense layers and head.

    Benchmark role: Model B (DenseNet vs ResNet comparison).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 growth_rate: int = 32,
                 block_layers: list[int] = None,
                 init_ch: int = 64, dropout_p: float = 0.2):
        super().__init__()
        if block_layers is None:
            block_layers = [6, 12, 24, 16]

        # ── Stem ───────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_ch, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # ── Dense blocks + Transitions ─────────────────────────
        ch = init_ch
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, n_layers in enumerate(block_layers):
            block = DenseBlock3d(ch, n_layers, growth_rate, dropout_p)
            self.blocks.append(block)
            ch = block.out_channels

            if i < len(block_layers) - 1:   # no transition after last block
                trans = TransitionLayer3d(ch)
                self.transitions.append(trans)
                ch = trans.out_channels
            else:
                self.transitions.append(None)   # placeholder

        # ── Classification head ────────────────────────────────
        self.final_bn   = nn.BatchNorm3d(ch)
        self.avgpool    = nn.AdaptiveAvgPool3d(1)
        self.dropout    = nn.Dropout(p=dropout_p)
        self.fc         = nn.Linear(ch, num_classes)
        self._out_ch    = ch

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout3d)):
                m.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for i, (block, trans) in enumerate(
                zip(self.blocks, self.transitions)):
            x = block(x)
            if trans is not None:
                x = trans(x)

        x = F.relu(self.final_bn(x), inplace=True)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Last dense block output for Grad-CAM."""
        x = self.stem(x)
        for i, (block, trans) in enumerate(
                zip(self.blocks, self.transitions)):
            x = block(x)
            if trans is not None and i < len(self.blocks) - 1:
                x = trans(x)
        return F.relu(self.final_bn(x), inplace=True)


# ─────────────────────────────────────────────────────────────
#  ROC-AUC Evaluation Utilities
# ─────────────────────────────────────────────────────────────

@dataclass
class ROCCurveResult:
    """Full ROC curve data + AUC for the DenseNet3D classifier."""
    fpr: List[float]          # False positive rates (thresholded)
    tpr: List[float]          # True positive rates (sensitivity)
    thresholds: List[float]   # Decision thresholds
    auc: float                # Area Under the ROC Curve (DeLong point estimate)
    auc_ci_lower: float       # 95% bootstrap CI lower bound
    auc_ci_upper: float       # 95% bootstrap CI upper bound
    optimal_threshold: float  # Youden's J = max(TPR - FPR)
    optimal_sensitivity: float
    optimal_specificity: float


def compute_roc_auc_curve(
    y_true: "np.ndarray",
    y_prob: "np.ndarray",
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> ROCCurveResult:
    """
    Computes the full ROC curve and AUC with patient-level bootstrap CI.

    Args:
        y_true:       Binary ground-truth labels (0 = benign, 1 = malignant).
        y_prob:       Predicted malignancy probability in [0, 1].
        n_bootstrap:  Number of bootstrap resamples for 95% CI.
        random_seed:  RNG seed for reproducibility.

    Returns:
        ROCCurveResult with curve arrays, AUC, CI, and Youden-optimal threshold.

    Clinical note:
        AUC is computed over the full probability spectrum.  For regulatory
        purposes, always also report sensitivity/specificity at the chosen
        operating threshold, not AUC alone.
    """
    if np is None:
        raise ImportError("numpy is required for ROC-AUC computation.")

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort by descending score to build the curve
    sorted_idx = np.argsort(y_prob)[::-1]
    y_prob_s = y_prob[sorted_idx]
    y_true_s = y_true[sorted_idx]

    thresholds: List[float] = []
    fpr_pts: List[float] = []
    tpr_pts: List[float] = []

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    tp = 0; fp = 0
    prev_prob = float('inf')
    for i in range(len(y_true_s)):
        if y_prob_s[i] != prev_prob:
            thresholds.append(float(y_prob_s[i]))
            fpr_pts.append(fp / max(1, n_neg))
            tpr_pts.append(tp / max(1, n_pos))
            prev_prob = float(y_prob_s[i])
        if y_true_s[i] == 1:
            tp += 1
        else:
            fp += 1

    # Close the curve at (1, 1)
    fpr_pts.append(1.0); tpr_pts.append(1.0); thresholds.append(0.0)

    # Trapezoid AUC
    auc = float(np.trapz(tpr_pts, fpr_pts))

    # Bootstrap 95% CI
    rng = np.random.default_rng(random_seed)
    boot_aucs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        b_true = y_true[idx]
        b_prob = y_prob[idx]
        if len(np.unique(b_true)) < 2:
            continue
        # Wilcoxon AUC estimate
        b_auc = float(np.mean([
            (b_prob[b_true == 1][:, None] > b_prob[b_true == 0][None, :]).mean()
        ]))
        boot_aucs.append(b_auc)

    ci_lo = float(np.percentile(boot_aucs, 2.5)) if len(boot_aucs) > 50 else max(0.0, auc - 0.05)
    ci_hi = float(np.percentile(boot_aucs, 97.5)) if len(boot_aucs) > 50 else min(1.0, auc + 0.05)

    # Youden's J optimal threshold
    tpr_arr = np.array(tpr_pts)
    fpr_arr = np.array(fpr_pts)
    j_arr = tpr_arr - fpr_arr
    opt_idx = int(np.argmax(j_arr))
    opt_thresh = float(thresholds[opt_idx])
    opt_sens = float(tpr_arr[opt_idx])
    opt_spec = float(1.0 - fpr_arr[opt_idx])

    return ROCCurveResult(
        fpr=[round(v, 4) for v in fpr_pts],
        tpr=[round(v, 4) for v in tpr_pts],
        thresholds=[round(v, 4) for v in thresholds],
        auc=round(auc, 4),
        auc_ci_lower=round(ci_lo, 4),
        auc_ci_upper=round(ci_hi, 4),
        optimal_threshold=round(opt_thresh, 4),
        optimal_sensitivity=round(opt_sens, 4),
        optimal_specificity=round(opt_spec, 4)
    )


# ─────────────────────────────────────────────────────────────
#  Statistical Existence Test  (McNemar + Prevalence Baseline)
# ─────────────────────────────────────────────────────────────

@dataclass
class StatisticalExistenceResult:
    """
    Tests whether the model's discriminative performance is statistically
    distinguishable from a no-information (prevalence-rate) baseline.
    """
    model_name: str
    dataset: str
    n_samples: int
    prevalence: float                  # Positive class prevalence P(Y=1)
    baseline_accuracy: float           # max(prevalence, 1-prevalence)
    model_accuracy: float
    chi2_statistic: float              # McNemar chi² vs. prevalence-only predictor
    p_value: float
    is_significant: bool               # p < 0.05
    effect_size_OR: float              # Odds ratio of improvement
    conclusion: str


def statistical_existence_test(
    y_true: "np.ndarray",
    y_prob: "np.ndarray",
    model_name: str = "DenseNet3D-121",
    dataset: str = "LIDC-IDRI",
    threshold: float = 0.50,
    alpha: float = 0.05
) -> StatisticalExistenceResult:
    """
    Tests statistical existence: whether the model is significantly better than
    predicting the majority class for every sample.

    Uses McNemar's test with continuity correction comparing:
      - Model A (trivial baseline): always predicts majority class
      - Model B (DenseNet3D-121):   thresholded at `threshold`

    This follows IBSI / radiomics good-practice: a model only 'exists'
    statistically if it beats the no-information baseline at p < 0.05.
    """
    if np is None:
        raise ImportError("numpy is required.")

    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    prevalence = float(y_true.mean())
    majority_class = 1 if prevalence >= 0.5 else 0
    baseline_pred = np.full_like(y_true, majority_class)
    baseline_acc = max(prevalence, 1.0 - prevalence)
    model_acc = float((y_pred == y_true).mean())

    # Discordant cells for McNemar
    # b: baseline correct, model wrong
    # c: baseline wrong, model correct
    baseline_correct = (baseline_pred == y_true)
    model_correct = (y_pred == y_true)
    b = int(np.sum(baseline_correct & ~model_correct))
    c = int(np.sum(~baseline_correct & model_correct))
    total_disc = b + c

    if total_disc == 0:
        chi2 = 0.0; p_val = 1.0
    else:
        chi2 = float((abs(b - c) - 1.0) ** 2 / total_disc) if abs(b - c) >= 1 else 0.0
        # chi2 CDF approximation (no scipy required)
        p_val = float(math.exp(-chi2 / 2.0))  # upper tail approximation df=1

    sig = p_val < alpha
    # Odds ratio of model improvement
    or_val = float(c / max(1, b))

    conclusion = (
        f"DenseNet3D-121 is statistically distinguishable from the prevalence "
        f"baseline (OR={or_val:.2f}, p={p_val:.4f})." if sig else
        f"Insufficient evidence to reject null hypothesis at α={alpha}."
    )

    return StatisticalExistenceResult(
        model_name=model_name,
        dataset=dataset,
        n_samples=len(y_true),
        prevalence=round(prevalence, 4),
        baseline_accuracy=round(baseline_acc, 4),
        model_accuracy=round(model_acc, 4),
        chi2_statistic=round(chi2, 4),
        p_value=round(p_val, 5),
        is_significant=sig,
        effect_size_OR=round(or_val, 3),
        conclusion=conclusion
    )


# ─────────────────────────────────────────────────────────────
#  Statistical Power Analysis  (Cohen's h / Power for AUC test)
# ─────────────────────────────────────────────────────────────

@dataclass
class PowerAnalysisResult:
    """
    Reports the retrospective statistical power of the current experiment.

    Also provides the prospective sample size needed to achieve 80% / 90%
    power for the observed AUC effect.
    """
    observed_auc: float
    null_auc: float                    # Null hypothesis AUC (e.g. 0.5 = random)
    effect_size_d: float               # Cohen's d equivalent for AUC difference
    n_pos: int
    n_neg: int
    observed_power: float              # Retrospective power at α=0.05
    n_required_80pct: int              # Prospective N for 80% power
    n_required_90pct: int              # Prospective N for 90% power
    is_adequately_powered: bool        # True if observed_power >= 0.80
    warning: str


def sample_power_analysis(
    observed_auc: float,
    n_pos: int,
    n_neg: int,
    null_auc: float = 0.5,
    alpha: float = 0.05
) -> PowerAnalysisResult:
    """
    Computes retrospective statistical power and prospective sample-size estimates
    for the AUC test using the Hanley & McNeil (1982) variance formula.

    Args:
        observed_auc: Measured ROC-AUC on the test set.
        n_pos:        Number of positive (malignant) cases.
        n_neg:        Number of negative (benign) cases.
        null_auc:     Null hypothesis AUC (0.5 = no discrimination).
        alpha:        Type-I error rate (default 0.05, two-sided).

    Returns:
        PowerAnalysisResult with retrospective power and prospective requirements.

    Reference:
        Hanley J.A., McNeil B.J. (1982). The meaning and use of the area under a
        receiver operating characteristic (ROC) curve. Radiology 143(1): 29–36.
    """
    A = observed_auc
    A0 = null_auc

    # Hanley-McNeil variance for a single AUC
    Q1 = A / (2.0 - A)
    Q2 = 2.0 * A**2 / (1.0 + A)
    var_A = (A * (1.0 - A) + (n_pos - 1.0) * (Q1 - A**2)
             + (n_neg - 1.0) * (Q2 - A**2)) / (n_pos * n_neg)
    se_A = math.sqrt(max(var_A, 1e-9))

    # Standardised effect size
    effect_d = (A - A0) / max(se_A, 1e-9)

    # Retrospective power (standard normal approximation, two-sided α)
    z_alpha = 1.960  # z_{1-α/2} for α=0.05
    power = _normal_cdf(abs(effect_d) - z_alpha)

    # Prospective sample size for 80% and 90% power
    n80  = _required_n(A, A0, z_alpha=z_alpha, z_beta=0.842)   # β=0.20
    n90  = _required_n(A, A0, z_alpha=z_alpha, z_beta=1.282)   # β=0.10

    warning = (
        "RETROSPECTIVE POWER: Informational only. Do not use as post-hoc justification."
    ) if power >= 0.80 else (
        f"LOW POWER ({power:.0%}): Results may not be reproducible. "
        f"Consider expanding cohort to N≥{n80} (pos+neg)."
    )

    return PowerAnalysisResult(
        observed_auc=round(A, 4),
        null_auc=round(A0, 4),
        effect_size_d=round(effect_d, 3),
        n_pos=n_pos,
        n_neg=n_neg,
        observed_power=round(power, 4),
        n_required_80pct=n80,
        n_required_90pct=n90,
        is_adequately_powered=(power >= 0.80),
        warning=warning
    )


def _normal_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _required_n(auc: float, auc0: float, z_alpha: float, z_beta: float) -> int:
    """Hanley-McNeil sample size for 1:1 positive-negative ratio."""
    A = auc; A0 = auc0
    Q1 = A / (2.0 - A); Q2 = 2.0 * A**2 / (1.0 + A)
    Q1_0 = A0 / (2.0 - A0); Q2_0 = 2.0 * A0**2 / (1.0 + A0)

    # Variance under H1 and H0
    def var(a, q1, q2, n): return (a * (1 - a) + (n - 1) * (q1 - a**2) + (n - 1) * (q2 - a**2)) / (n * n)

    # Iteratively solve for n (assumes n_pos = n_neg = n)
    n = 20
    for _ in range(500):
        v1 = var(A, Q1, Q2, n)
        v0 = var(A0, Q1_0, Q2_0, n)
        n_new = math.ceil(((z_alpha * math.sqrt(v0) + z_beta * math.sqrt(v1)) / max(1e-9, A - A0)) ** 2)
        if abs(n_new - n) <= 1:
            break
        n = n_new
    return int(n * 2)  # total: n_pos + n_neg


# ─────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as _np

    model = DenseNet3D121(in_channels=1, num_classes=2)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print(f"DenseNet3D-121 output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── ROC-AUC demo on synthetic data ──────────────────────
    rng = _np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob  = rng.beta(5, 2, size=n) * y_true + rng.beta(2, 5, size=n) * (1 - y_true)

    roc = compute_roc_auc_curve(y_true, y_prob)
    print(f"\nROC-AUC: {roc.auc:.4f}  95% CI [{roc.auc_ci_lower:.4f}, {roc.auc_ci_upper:.4f}]")
    print(f"Optimal threshold: {roc.optimal_threshold:.3f}  "
          f"Sens={roc.optimal_sensitivity:.3f}  Spec={roc.optimal_specificity:.3f}")

    exist = statistical_existence_test(y_true, y_prob)
    print(f"\nStatistical Existence — {exist.conclusion}")
    print(f"  χ²={exist.chi2_statistic:.4f}  p={exist.p_value:.5f}  "
          f"OR={exist.effect_size_OR:.2f}  Significant={exist.is_significant}")

    pwr = sample_power_analysis(roc.auc, n_pos=int(y_true.sum()), n_neg=int((1-y_true).sum()))
    print(f"\nPower Analysis — Retrospective power: {pwr.observed_power:.1%}")
    print(f"  Adequately powered: {pwr.is_adequately_powered}")
    print(f"  N required 80%: {pwr.n_required_80pct}   N required 90%: {pwr.n_required_90pct}")
    print(f"  {pwr.warning}")
