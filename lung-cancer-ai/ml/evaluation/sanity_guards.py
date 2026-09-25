"""
ml/evaluation/sanity_guards.py
==============================
Sanity Guard Addendum implementation for the Master Execution pipeline.

Guards:
  1. Mandatory Run-Mode Label (Guard 1):
     Explicit, impossible-to-miss banner distinguishing SYNTHETIC/SMOKE-TEST runs
     from REAL DATA runs with ground truth tiers.
  2. Implausible-Metric Circuit Breaker (Guard 2):
     Hard-failure check after every metric computation (zero-width CIs, perfect AUC
     on real data, small-n abstention claims) that halts rather than publishing clean tables.
  3. Change-Gate Model Versioning (Guard 3):
     Recalibration yields STATUS = "VALIDATED_CANDIDATE", never a live version string,
     until Part U change-gate and Part Q approval items pass. Synthetic runs use
     non-deployment candidate tags ("smoketest-candidate-v0").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GUARD 1: Mandatory Run-Mode Label
# ---------------------------------------------------------------------------

class RunMode(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    REAL = "REAL"


def format_run_mode_banner(
    run_mode: RunMode | str = RunMode.SYNTHETIC,
    dataset_name: Optional[str] = None,
    ground_truth_tier: Optional[str] = None
) -> str:
    """
    Renders the mandatory impossible-to-miss run-mode declaration banner.
    No stage or report is permitted to output without this header.
    """
    banner_bar = "=" * 80
    mode_str = str(run_mode).upper()
    
    if "REAL" in mode_str:
        ds_name = dataset_name or "UNSPECIFIED_COHORT"
        tier_str = ground_truth_tier or "TIER_UNVERIFIED"
        content = f"  RUN MODE: REAL DATA — {ds_name} — {tier_str}"
    else:
        content = "  RUN MODE: SYNTHETIC / SMOKE-TEST DATA — NOT A CLINICAL OR RESEARCH RESULT"
        
    return f"\n{banner_bar}\n{content}\n{banner_bar}\n"


# ---------------------------------------------------------------------------
# GUARD 2: Implausible-Metric Circuit Breaker
# ---------------------------------------------------------------------------

class MetricsSanityError(ValueError):
    """Raised when evaluation metrics violate fundamental empirical sanity constraints."""
    def __init__(self, warnings: List[str]):
        self.warnings = warnings
        msg = (
            "CIRCUIT BREAKER TRIGGERED (Guard 2): Metric report generation blocked due to "
            f"implausible/degenerate values:\n  - " + "\n  - ".join(warnings)
        )
        super().__init__(msg)


def sanity_check_metrics(metrics: Dict[str, Any], n_cases: int, run_mode: str = "SYNTHETIC") -> None:
    """
    Automated circuit breaker that runs after every metrics computation.
    Fails loudly and blocks report generation rather than logging and continuing.
    """
    warnings = []
    mode = str(run_mode).upper()

    # Rule 1: Zero-width confidence interval on non-infinite cohorts
    ci_width = metrics.get("ci_width", None)
    if ci_width is not None and ci_width == 0.0 and n_cases < 500:
        warnings.append(
            f"CI width is exactly zero on a small sample (n={n_cases}) — check for "
            "degenerate/synthetic labels or a data leakage bug."
        )

    # Rule 2: Perfect AUC (1.0000) on real data is a leakage signal
    auc = metrics.get("auc", None)
    if auc is not None and auc >= 0.9999 and "REAL" in mode:
        warnings.append(
            "Perfect AUC (1.0000) on real data is a leakage/overfitting signal until "
            "proven otherwise — do not report without a documented leakage audit."
        )

    # Rule 3: Abstention precision claimed as 100% on <10 cases
    abs_prec = metrics.get("abstention_precision", None)
    n_abstained = metrics.get("n_abstained", 0)
    if abs_prec is not None and abs_prec >= 0.9999 and n_abstained < 10:
        warnings.append(
            f"Abstention precision of 100% on <10 abstained cases (n={n_abstained}) is "
            "not a statistically meaningful claim — report n, not just %."
        )

    # Rule 4: Exactly identical conditional and unconditional metrics when cases were abstained
    cond_auc = metrics.get("conditional_auc", None)
    uncond_auc = metrics.get("unconditional_auc", None)
    if cond_auc is not None and uncond_auc is not None and n_abstained > 5:
        if abs(cond_auc - uncond_auc) < 1e-9 and metrics.get("abstention_rate_pct", 0) > 10.0:
            warnings.append(
                "Conditional and unconditional AUC are identical despite >10% abstention rate — "
                "verify that abstained cases are genuinely removed from conditional subset."
            )

    if warnings:
        raise MetricsSanityError(warnings)


# ---------------------------------------------------------------------------
# GUARD 3: Change-Gate Model Versioning
# ---------------------------------------------------------------------------

class CandidateGateStatus(str, Enum):
    VALIDATED_CANDIDATE = "VALIDATED_CANDIDATE"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED_FOR_DEPLOYMENT = "APPROVED_FOR_DEPLOYMENT"
    REJECTED = "REJECTED"


@dataclass
class ChangeGateChecklist:
    candidate_id: str
    unit_tests_passed: bool = False
    leakage_audit_passed: bool = False
    internal_val_passed: bool = False
    calibration_val_passed: bool = False
    robustness_audit_passed: bool = False
    ood_detection_passed: bool = False
    explanation_reliability_passed: bool = False
    abstention_eval_passed: bool = False
    regression_tests_passed: bool = False
    clinical_lead_approval_signed: bool = False

    @property
    def all_checks_passed(self) -> bool:
        return (
            self.unit_tests_passed and
            self.leakage_audit_passed and
            self.internal_val_passed and
            self.calibration_val_passed and
            self.robustness_audit_passed and
            self.ood_detection_passed and
            self.explanation_reliability_passed and
            self.abstention_eval_passed and
            self.regression_tests_passed and
            self.clinical_lead_approval_signed
        )


def evaluate_candidate_gate(
    checklist: ChangeGateChecklist,
    is_synthetic_run: bool = True
) -> Tuple[CandidateGateStatus, Optional[str], str]:
    """
    Evaluates whether a candidate model can be minted as a production version.
    
    Guard 3 Rule:
    - Never produce live version strings on synthetic runs (uses 'smoketest-candidate-v0').
    - Requires all 10 gate items and clinical lead sign-off before minting 'vX.Y.Z-calibrated'.
    """
    if is_synthetic_run:
        return (
            CandidateGateStatus.VALIDATED_CANDIDATE,
            None,
            "smoketest-candidate-v0 (Synthetic run — not eligible for production versioning)"
        )

    if not checklist.all_checks_passed:
        missing = [k for k, v in checklist.__dict__.items() if v is False]
        return (
            CandidateGateStatus.PENDING_APPROVAL,
            None,
            f"Candidate gated. Pending requirements: {', '.join(missing)}"
        )

    # All gates passed on real validation cohort
    production_version_tag = f"v2.1.0-calibrated"
    return (
        CandidateGateStatus.APPROVED_FOR_DEPLOYMENT,
        production_version_tag,
        f"All 10 Part U change-gates and Part Q approvals verified. Minted {production_version_tag}."
    )
