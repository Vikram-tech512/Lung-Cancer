"""
tests/test_differentiation_features.py
======================================
Tests verifying the 4 Research Differentiation Contributions:
  1. Explanation Reliability Score & Multi-Method Agreement
  2. Abstention as Headline Metric & Abstention Precision
  3. Disagreement Feedback Loop & Temperature Recalibration
  4. Post-Deployment Surveillance & Drift API
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from ml.explainability.reliability import ExplanationReliabilityScorer
from ml.evaluation.metrics import compute_abstention_metrics
from ml.feedback.disagreement_analyzer import DisagreementAnalyzer
from backend.main import app

client = TestClient(app)


def test_explanation_reliability_scorer():
    scorer = ExplanationReliabilityScorer()
    cam1 = np.zeros((16, 16, 16), dtype=np.float32)
    cam1[6:10, 6:10, 6:10] = 1.0

    cam2 = np.zeros((16, 16, 16), dtype=np.float32)
    cam2[6:10, 6:10, 6:10] = 0.95

    res = scorer.evaluate_reliability(
        primary_cam_3d=cam1,
        aux_campp_3d=cam2,
        ig_3d=cam2,
        cel_reference_sim=0.90
    )
    assert 0.0 <= res.composite_reliability_score <= 1.0
    assert res.reliability_category in ("HIGH", "MODERATE", "LOW")
    assert isinstance(res.forces_abstention, bool)


def test_abstention_as_headline_metric():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.48, 0.2, 0.52, 0.05, 0.95])
    # Case index 5 and 7 are near 0.5 (borderline / abstained)
    abstained = np.array([False, False, False, False, False, True, False, True, False, False])
    ambiguity_gt = np.array([False, False, False, False, False, True, False, True, False, False])

    metrics = compute_abstention_metrics(y_true, y_prob, abstained, ambiguity_gt)
    assert metrics.total_cases == 10
    assert metrics.abstained_cases == 2
    assert metrics.abstention_rate_pct == 20.0
    assert metrics.abstention_precision_pct == 100.0
    assert metrics.conditional_accuracy >= metrics.unconditional_accuracy


def test_disagreement_analyzer_and_recalibration():
    analyzer = DisagreementAnalyzer()
    reviews = [
        {"action": "ACCEPT", "ai_verdict": "MALIGNANT", "radiologist_verdict": "MALIGNANT"},
        {"action": "REJECT", "ai_verdict": "MALIGNANT", "radiologist_verdict": "BENIGN"},
        {"action": "MODIFY", "ai_verdict": "MALIGNANT", "radiologist_verdict": "MALIGNANT"},
        {"action": "MARK_FP", "ai_verdict": "MALIGNANT", "radiologist_verdict": "BENIGN"},
    ]
    report = analyzer.analyze_reviews(reviews)
    assert report.overall_rejection_rate_pct > 0.0
    assert len(report.top_divergent_clusters) > 0

    # Temperature optimization (Guard 3: returns VALIDATED_CANDIDATE, not live version)
    probs = np.array([0.85, 0.45, 0.92, 0.30])
    verdicts = np.array(["MALIGNANT", "BENIGN", "MALIGNANT", "BENIGN"])
    recal = analyzer.optimize_temperature_recalibration(probs, verdicts, prior_temperature=1.25)
    assert recal.optimized_temperature > 0.0
    assert recal.status == "VALIDATED_CANDIDATE"
    assert recal.is_live_deployment_version is False
    assert "candidate" in recal.candidate_version_id.lower()


def test_drift_surveillance_endpoints():
    # Test GET /api/drift/surveillance
    r = client.get("/api/drift/surveillance")
    assert r.status_code == 200
    data = r.json()
    assert "overall_rejection_rate_pct" in data
    assert "top_divergent_clusters" in data
    assert "divergence_trend" in data

    # Test POST /api/drift/recalibrate (Guard 3: returns candidate status)
    r_recal = client.post("/api/drift/recalibrate", json={"version_tag": "v2.0.2-test"})
    assert r_recal.status_code == 200
    recal_data = r_recal.json()
    assert recal_data["status"] == "VALIDATED_CANDIDATE"
    assert recal_data["is_live_deployment_version"] is False
    assert "candidate" in recal_data["candidate_version_id"].lower()
    assert "optimized_temperature" in recal_data
