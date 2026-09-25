"""
ai/
===
Root AI package namespace exporting core model architectures and clinical pipelines.
Maps directly to ml/ for seamless modular access.
"""
from ml.segmentation.lung_segmenter import LungSegmenter3D, LungSegmentationResult
from ml.segmentation.nodule_segmenter import NoduleSegmenter3D, NoduleSegmentationResult, NoduleMeasurements
from ml.segmentation.unet3d import UNet3D
from ml.detection.retinanet3d import RetinaNet3D
from ml.morphology.analyzer import MorphologyAnalyzer, MorphologyAssessment
from ml.risk.malignancy_model import MalignancyRiskModel, RiskEstimationResult, ClinicalVariables
from ml.explainability.cel import ContrastiveExplainabilityLoss, DifferentiableGradCAM3D, evaluate_explanation_stability
from ml.explainability.gradcam3d import GradCAM3D, GradCAMPP3D
from ml.uncertainty.mc_dropout import mc_dropout_predict, deep_ensemble_predict, UncertaintyResult
from ml.longitudinal.comparator import LongitudinalComparator, LongitudinalComparisonResult
from ml.ood.detector import OODDetector, OODCheckResult
from ml.fp_reduction.analyzer import FalsePositiveAnalyzer, FPReductionAssessment
from ml.fusion.multimodal import MultimodalFusionEngine, MultimodalPredictionResult

__all__ = [
    "LungSegmenter3D",
    "LungSegmentationResult",
    "NoduleSegmenter3D",
    "NoduleSegmentationResult",
    "NoduleMeasurements",
    "UNet3D",
    "RetinaNet3D",
    "MorphologyAnalyzer",
    "MorphologyAssessment",
    "MalignancyRiskModel",
    "RiskEstimationResult",
    "ClinicalVariables",
    "ContrastiveExplainabilityLoss",
    "DifferentiableGradCAM3D",
    "evaluate_explanation_stability",
    "GradCAM3D",
    "GradCAMPP3D",
    "mc_dropout_predict",
    "deep_ensemble_predict",
    "UncertaintyResult",
    "LongitudinalComparator",
    "LongitudinalComparisonResult",
    "OODDetector",
    "OODCheckResult",
    "FalsePositiveAnalyzer",
    "FPReductionAssessment",
    "MultimodalFusionEngine",
    "MultimodalPredictionResult",
]
