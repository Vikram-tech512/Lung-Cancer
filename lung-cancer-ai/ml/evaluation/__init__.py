"""
ml/evaluation/__init__.py
"""
from ml.evaluation.metrics import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    compute_ece,
    compute_abstention_metrics,
    ClassificationMetrics,
    SegmentationMetrics,
    AbstentionEvaluationMetrics,
)
from ml.evaluation.radiologist_study import (
    RadiologistStudyEvaluator,
    ReaderStudyResult,
    ReaderStudySummary,
)

from ml.evaluation.ground_truth import (
    GroundTruthTier,
    DatasetMetadata,
    DATASET_REGISTRY,
    get_ground_truth_metadata,
    validate_metric_table_ground_truth,
)
from ml.evaluation.statistical_rigor import (
    bootstrap_patient_ci,
    mcnemar_test,
    delong_roc_test,
    adjust_p_values,
    cohen_kappa,
    fleiss_kappa,
    fit_reliability_weights_val_only,
    BootstrapCIResult,
    SignificanceTestResult,
    MultipleComparisonReport,
    InterRaterReliabilityResult,
    FrozenReliabilityWeights,
)

__all__ = [
    "compute_classification_metrics",
    "compute_segmentation_metrics",
    "compute_ece",
    "compute_abstention_metrics",
    "ClassificationMetrics",
    "SegmentationMetrics",
    "AbstentionEvaluationMetrics",
    "RadiologistStudyEvaluator",
    "ReaderStudyResult",
    "ReaderStudySummary",
    "GroundTruthTier",
    "DatasetMetadata",
    "DATASET_REGISTRY",
    "get_ground_truth_metadata",
    "validate_metric_table_ground_truth",
    "bootstrap_patient_ci",
    "mcnemar_test",
    "delong_roc_test",
    "adjust_p_values",
    "cohen_kappa",
    "fleiss_kappa",
    "fit_reliability_weights_val_only",
    "BootstrapCIResult",
    "SignificanceTestResult",
    "MultipleComparisonReport",
    "InterRaterReliabilityResult",
    "FrozenReliabilityWeights",
]
