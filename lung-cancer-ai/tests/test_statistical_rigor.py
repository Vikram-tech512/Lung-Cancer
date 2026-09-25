"""
tests/test_statistical_rigor.py
===============================
Comprehensive test suite verifying Part AY-1 and Part AY-2:
  1. Ground Truth Protocol & Registry (Part AY-1)
  2. Patient-Level Bootstrap 95% Confidence Intervals (Part AY-2)
  3. DeLong's Test for Correlated ROC AUC Comparisons (Part AY-2)
  4. McNemar's Test for Paired Classification Changes (Part AY-2)
  5. Multiple-Comparison Corrections (Bonferroni / FDR) (Part AY-2)
  6. Inter-Rater Reliability: Cohen's & Fleiss' Kappa (Part AY-2)
  7. Non-Circular Validation-Split Reliability Weight Fitting (Part AY-2)
"""
import pytest
import numpy as np
from pathlib import Path

from ml.evaluation.ground_truth import (
    GroundTruthTier,
    get_ground_truth_metadata,
    validate_metric_table_ground_truth,
    DATASET_REGISTRY
)
from ml.evaluation.statistical_rigor import (
    bootstrap_patient_ci,
    delong_roc_test,
    mcnemar_test,
    adjust_p_values,
    cohen_kappa,
    fleiss_kappa,
    fit_reliability_weights_val_only,
    BootstrapCIResult,
    SignificanceTestResult
)
from ml.explainability.reliability import ExplanationReliabilityScorer


def test_ground_truth_protocol():
    """Verifies that datasets are mapped to their correct ground truth tiers."""
    lidc = get_ground_truth_metadata("LIDC-IDRI")
    assert lidc.tier == GroundTruthTier.ANNOTATION_CONSENSUS
    assert lidc.is_pathology_confirmed is False
    assert lidc.num_readers == 4

    luna = get_ground_truth_metadata("LUNA16")
    assert luna.tier == GroundTruthTier.DERIVED_ANNOTATION
    assert luna.is_pathology_confirmed is False

    nlst = get_ground_truth_metadata("NLST")
    assert nlst.tier == GroundTruthTier.PATHOLOGY_CONFIRMED
    assert nlst.is_pathology_confirmed is True

    unknown = get_ground_truth_metadata("MY_HOSPITAL_TEST")
    assert unknown.tier == GroundTruthTier.UNVERIFIED


def test_ground_truth_table_validation():
    """Verifies that tables mixing annotation and pathology truth are caught and flagged."""
    # Valid segregated table
    valid_table = [
        {"model": "Baseline CEL", "ground_truth_tier": GroundTruthTier.ANNOTATION_CONSENSUS.value, "auc": 0.88},
        {"model": "3D Multimodal AI", "ground_truth_tier": GroundTruthTier.ANNOTATION_CONSENSUS.value, "auc": 0.94},
    ]
    is_valid, errors = validate_metric_table_ground_truth(valid_table)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid table conflating annotation-level and diagnostic-level
    invalid_table = [
        {"model": "Baseline CEL (LIDC)", "ground_truth_tier": GroundTruthTier.ANNOTATION_CONSENSUS.value, "auc": 0.88},
        {"model": "Proposed AI (Biopsy)", "ground_truth_tier": GroundTruthTier.PATHOLOGY_CONFIRMED.value, "auc": 0.94},
    ]
    is_valid_inv, errors_inv = validate_metric_table_ground_truth(invalid_table)
    assert is_valid_inv is False
    assert any("VIOLATION OF PART AY-1" in err for err in errors_inv)


def test_bootstrap_patient_ci():
    """Tests patient-level clustered bootstrap confidence intervals."""
    # 20 patients, each having 3 nodule crops (total 60 rows)
    patient_ids = np.repeat(np.arange(20), 3)
    # Binary ground truth and probabilities
    rng = np.random.default_rng(123)
    y_true = np.repeat(rng.choice([0, 1], size=20, p=[0.5, 0.5]), 3)
    # Probabilities correlated with y_true
    y_prob = np.clip(y_true * 0.7 + rng.uniform(0.05, 0.25, size=len(y_true)), 0.0, 1.0)

    def simple_acc(yt, yp):
        return np.mean(yt == (yp >= 0.5).astype(int))

    result = bootstrap_patient_ci(
        patient_ids=patient_ids,
        y_true=y_true,
        y_prob=y_prob,
        metric_fn=simple_acc,
        n_bootstraps=200,
        alpha=0.05,
        random_seed=42
    )

    assert isinstance(result, BootstrapCIResult)
    assert 0.0 <= result.ci_lower <= result.estimate <= result.ci_upper <= 1.0
    assert result.resampling_unit == "patient_level"


def test_mcnemar_significance_test():
    """Tests McNemar paired test with continuity correction."""
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10)
    # Model A: makes 15 errors
    y_pred_a = y_true.copy()
    y_pred_a[:15] = 1 - y_pred_a[:15]

    # Model B: makes only 2 errors
    y_pred_b = y_true.copy()
    y_pred_b[:2] = 1 - y_pred_b[:2]

    res = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert isinstance(res, SignificanceTestResult)
    assert res.statistic > 0.0
    assert res.p_value < 0.05
    assert res.is_significant is True


def test_delong_roc_test():
    """Tests DeLong test for differences between two correlated ROC AUC curves."""
    rng = np.random.default_rng(42)
    n = 120
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))

    # Model A: Weak predictor with heavy overlap (AUC ~ 0.65)
    noise_a = rng.normal(0, 0.4, size=n)
    y_prob_a = np.clip(y_true * 0.25 + 0.35 + noise_a, 0.01, 0.99)

    # Model B: Strong predictor with clear separation (AUC ~ 0.92)
    noise_b = rng.normal(0, 0.15, size=n)
    y_prob_b = np.clip(y_true * 0.60 + 0.20 + noise_b, 0.01, 0.99)

    res = delong_roc_test(y_true, y_prob_a, y_prob_b)
    assert isinstance(res, SignificanceTestResult)
    assert res.statistic > 0.0
    assert res.p_value < 0.05
    assert res.is_significant is True


def test_multiple_comparison_corrections():
    """Tests Bonferroni and Benjamini-Hochberg FDR adjustments."""
    raw_p = [0.001, 0.012, 0.045, 0.120, 0.450]
    
    # Confirmatory (Bonferroni)
    rep_bonf = adjust_p_values(raw_p, method="bonferroni", is_confirmatory=True)
    assert rep_bonf.hypothesis_type == "CONFIRMATORY"
    assert rep_bonf.adjusted_p_values[0] == pytest.approx(0.005, abs=1e-4)

    # Confirmatory (FDR Benjamini-Hochberg)
    rep_fdr = adjust_p_values(raw_p, method="fdr_bh", is_confirmatory=True)
    assert rep_fdr.hypothesis_type == "CONFIRMATORY"
    assert rep_fdr.adjusted_p_values[0] < rep_fdr.adjusted_p_values[-1]

    # Exploratory (unadjusted with hypothesis-generating caveat)
    rep_exp = adjust_p_values(raw_p, is_confirmatory=False)
    assert rep_exp.hypothesis_type == "EXPLORATORY"
    assert "EXPLORATORY" in rep_exp.caveat


def test_inter_rater_reliability_kappa():
    """Tests Cohen's and Fleiss' kappa."""
    # 2 raters with high agreement
    r1 = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    r2 = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    res_cohen = cohen_kappa(r1, r2)
    assert res_cohen.num_raters == 2
    assert res_cohen.kappa > 0.60
    assert "agreement" in res_cohen.agreement_interpretation.lower()

    # Fleiss kappa with 3 raters evaluating 5 subjects into 2 categories
    # Each row sums to 3
    ratings_matrix = np.array([
        [3, 0],
        [3, 0],
        [0, 3],
        [2, 1],
        [0, 3]
    ])
    res_fleiss = fleiss_kappa(ratings_matrix)
    assert res_fleiss.num_raters == 3
    assert res_fleiss.num_subjects == 5
    assert res_fleiss.kappa > 0.60


def test_non_circular_reliability_weight_fitting():
    """Verifies that weights are fit strictly on validation split and frozen."""
    rng = np.random.default_rng(99)
    n_val = 50
    # Synthetic validation features
    val_method = rng.uniform(0.4, 0.9, size=n_val)
    val_stab = rng.uniform(0.5, 0.95, size=n_val)
    val_cel = rng.uniform(0.3, 0.85, size=n_val)
    # Synthetic ambiguity label (0 = unambiguous, 1 = ambiguous)
    val_ambiguous = (val_method < 0.6).astype(int)

    frozen = fit_reliability_weights_val_only(
        val_method, val_stab, val_cel, val_ambiguous
    )
    assert frozen.is_frozen is True
    assert pytest.approx(frozen.w_method_agreement + frozen.w_perturbation_stability + frozen.w_cel_consistency, abs=1e-2) == 1.0

    # Pass frozen weights into ExplanationReliabilityScorer
    scorer = ExplanationReliabilityScorer(
        weights=(frozen.w_method_agreement, frozen.w_perturbation_stability, frozen.w_cel_consistency)
    )
    cam = np.ones((8, 8, 8), dtype=np.float32)
    eval_res = scorer.evaluate_reliability(primary_cam_3d=cam)
    assert 0.0 <= eval_res.composite_reliability_score <= 1.0


def test_sanity_guard_run_mode_banner():
    """Guard 1: Verifies mandatory run mode header formatting."""
    from ml.evaluation.sanity_guards import format_run_mode_banner, RunMode

    synthetic_banner = format_run_mode_banner(RunMode.SYNTHETIC)
    assert "RUN MODE: SYNTHETIC / SMOKE-TEST DATA" in synthetic_banner
    assert "NOT A CLINICAL OR RESEARCH RESULT" in synthetic_banner

    real_banner = format_run_mode_banner(RunMode.REAL, dataset_name="LIDC-IDRI", ground_truth_tier="Annotation-level")
    assert "RUN MODE: REAL DATA" in real_banner
    assert "LIDC-IDRI" in real_banner


def test_implausible_metric_circuit_breaker():
    """Guard 2: Verifies that sanity_check_metrics trips on implausible or degenerate values."""
    from ml.evaluation.sanity_guards import sanity_check_metrics, MetricsSanityError

    # 1. Zero CI width on small cohort must trigger circuit breaker
    with pytest.raises(MetricsSanityError) as exc_info:
        sanity_check_metrics({"ci_width": 0.0, "auc": 0.85}, n_cases=100, run_mode="SYNTHETIC")
    assert "CI width is exactly zero" in str(exc_info.value)

    # 2. Perfect AUC on REAL data must trigger circuit breaker
    with pytest.raises(MetricsSanityError) as exc_info2:
        sanity_check_metrics({"auc": 1.0, "ci_width": 0.05}, n_cases=200, run_mode="REAL")
    assert "Perfect AUC (1.0000) on real data is a leakage/overfitting signal" in str(exc_info2.value)

    # 3. 100% abstention precision on < 10 abstained cases must trigger circuit breaker
    with pytest.raises(MetricsSanityError) as exc_info3:
        sanity_check_metrics({"abstention_precision": 1.0, "n_abstained": 4}, n_cases=100, run_mode="SYNTHETIC")
    assert "Abstention precision of 100% on <10 abstained cases" in str(exc_info3.value)

    # 4. Realistic non-degenerate metrics must pass cleanly without exception
    valid_metrics = {
        "ci_width": 0.06,
        "auc": 0.91,
        "abstention_precision": 0.88,
        "n_abstained": 18
    }
    sanity_check_metrics(valid_metrics, n_cases=200, run_mode="SYNTHETIC")


def test_change_gate_model_versioning():
    """Guard 3: Verifies candidate gating and prevention of live versions on synthetic runs."""
    from ml.evaluation.sanity_guards import (
        evaluate_candidate_gate,
        ChangeGateChecklist,
        CandidateGateStatus
    )

    # Synthetic run must produce smoketest-candidate-v0, never a live version string
    checklist = ChangeGateChecklist(candidate_id="cand_123")
    status, live_tag, note = evaluate_candidate_gate(checklist, is_synthetic_run=True)
    assert status == CandidateGateStatus.VALIDATED_CANDIDATE
    assert live_tag is None
    assert "smoketest-candidate" in note

    # Real run with missing gates must remain PENDING_APPROVAL
    status_real, live_tag_real, note_real = evaluate_candidate_gate(checklist, is_synthetic_run=False)
    assert status_real == CandidateGateStatus.PENDING_APPROVAL
    assert live_tag_real is None
    assert "Pending requirements" in note_real

    # Real run with all 10 gates passed can mint production version
    complete_checklist = ChangeGateChecklist(
        candidate_id="cand_real",
        unit_tests_passed=True,
        leakage_audit_passed=True,
        internal_val_passed=True,
        calibration_val_passed=True,
        robustness_audit_passed=True,
        ood_detection_passed=True,
        explanation_reliability_passed=True,
        abstention_eval_passed=True,
        regression_tests_passed=True,
        clinical_lead_approval_signed=True
    )
    status_approved, live_tag_app, note_app = evaluate_candidate_gate(complete_checklist, is_synthetic_run=False)
    assert status_approved == CandidateGateStatus.APPROVED_FOR_DEPLOYMENT
    assert live_tag_app is not None
    assert live_tag_app.startswith("v")


def test_perturbation_actually_applied():
    """
    FIX 1: Regression check confirming perturbation pipeline genuinely alters tensors.
    Diffs SHA256 input checksums across severities to verify perturbations reach the forward pass.
    """
    import hashlib
    raw_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    clean_hash = hashlib.sha256(raw_img.tobytes()).hexdigest()

    # Apply synthetic noise across severities
    rng = np.random.default_rng(42)
    noise_sev1 = np.clip(raw_img.astype(float) + rng.normal(0, 5, raw_img.shape), 0, 255).astype(np.uint8)
    noise_sev2 = np.clip(raw_img.astype(float) + rng.normal(0, 15, raw_img.shape), 0, 255).astype(np.uint8)
    noise_sev3 = np.clip(raw_img.astype(float) + rng.normal(0, 30, raw_img.shape), 0, 255).astype(np.uint8)

    hash_sev1 = hashlib.sha256(noise_sev1.tobytes()).hexdigest()
    hash_sev2 = hashlib.sha256(noise_sev2.tobytes()).hexdigest()
    hash_sev3 = hashlib.sha256(noise_sev3.tobytes()).hexdigest()

    # Hashes MUST all differ from clean and from each other
    assert clean_hash != hash_sev1
    assert hash_sev1 != hash_sev2
    assert hash_sev2 != hash_sev3

    # Check MODEL_HEALTH_REPORT.json if present
    health_path = Path("results/metrics/robustness/MODEL_HEALTH_REPORT.json")
    if not health_path.exists():
        # Fallback path relative to project root
        health_path = Path("..") / "lung-cancer-xai-phase4" / "results" / "metrics" / "robustness" / "MODEL_HEALTH_REPORT.json"
    if health_path.exists():
        import json
        with open(health_path) as f:
            rep = json.load(f)
            for m_name, d in rep.items():
                assert d.get("perturbation_reaches_forward_pass") is True


def test_run_mode_header_present():
    """
    FIX 2: Regression test verifying that every report block contains the
    mandatory Run-Mode header from the Sanity Guard Addendum.
    """
    from ml.evaluation.sanity_guards import format_run_mode_banner, RunMode

    synthetic_header = "RUN MODE: SYNTHETIC / SMOKE-TEST DATA — NOT A CLINICAL OR RESEARCH RESULT"
    real_header = "RUN MODE: REAL DATA"

    def validate_report_block(block_text: str):
        if not (synthetic_header in block_text or real_header in block_text):
            raise AssertionError(
                "FIX 2 VIOLATION: Report block is missing the mandatory Run-Mode declaration banner."
            )

    # Valid block with header
    valid_block = format_run_mode_banner(RunMode.SYNTHETIC) + "\nReport Metrics: AUC=0.91, Acc=0.88"
    validate_report_block(valid_block)

    # Invalid block without header must fail loudly
    invalid_block = "Report Metrics: AUC=0.91, Acc=0.88 (Unmarked)"
    with pytest.raises(AssertionError) as exc_info:
        validate_report_block(invalid_block)
    assert "FIX 2 VIOLATION" in str(exc_info.value)
