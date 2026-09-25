"""
ml/evaluation/ground_truth.py
=============================
Part AY-1 — Ground Truth Protocol.

Defines explicit ground-truth basis and confidence tiers for all datasets used
in training, evaluation, and benchmark reporting.

Rule (Part AY-1):
Every reported risk/calibration/disagreement metric MUST state which ground-truth
tier it is measured against. Never present "annotation-consensus-level accuracy"
and "pathology-confirmed accuracy" as the same kind of number in the same table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class GroundTruthTier(str, Enum):
    """
    Categorization of ground truth confidence level.
    """
    ANNOTATION_CONSENSUS = "Annotation-level (Radiologist Consensus, not pathology-confirmed)"
    DERIVED_ANNOTATION = "Derived Annotation-level (Candidate-level consensus from radiologist markings)"
    PATHOLOGY_CONFIRMED = "Diagnostic-level (Pathology/Histology-confirmed outcome)"
    LONGITUDINAL_FOLLOW_UP = "Diagnostic-level (>= 2-Year Serial CT Stability/Resolution Confirmation)"
    UNVERIFIED = "Unverified Source (Must verify labeling protocol before clinical claims)"


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata specifying the ground truth protocol for a dataset."""
    name: str
    ground_truth_basis: str
    tier: GroundTruthTier
    confidence_level: str
    num_readers: Optional[int]
    is_pathology_confirmed: bool
    caveat: str


DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    "LIDC-IDRI": DatasetMetadata(
        name="LIDC-IDRI",
        ground_truth_basis="Multi-radiologist annotation consensus (up to 4 thoracic readers)",
        tier=GroundTruthTier.ANNOTATION_CONSENSUS,
        confidence_level="Annotation-level, not diagnostic-level",
        num_readers=4,
        is_pathology_confirmed=False,
        caveat="LIDC-IDRI malignancy scores reflect radiologist perceptual suspicion (1-5 scale), NOT tissue pathology for >90% of cases."
    ),
    "LUNA16": DatasetMetadata(
        name="LUNA16",
        ground_truth_basis="Derived from LIDC-IDRI candidate annotations (>=3 of 4 radiologist agreement)",
        tier=GroundTruthTier.DERIVED_ANNOTATION,
        confidence_level="Annotation-level, not diagnostic-level",
        num_readers=4,
        is_pathology_confirmed=False,
        caveat="Nodule presence is established by consensus among readers, but histology remains unconfirmed for the vast majority."
    ),
    "IQ-OTH/NCCD": DatasetMetadata(
        name="IQ-OTH/NCCD",
        ground_truth_basis="Per dataset documentation (Check institutional pathology vs radiologist origin)",
        tier=GroundTruthTier.UNVERIFIED,
        confidence_level="Verify before use",
        num_readers=None,
        is_pathology_confirmed=False,
        caveat="Must verify whether labels represent histologic biopsy results or subjective visual categorization prior to publishing claim."
    ),
    "NLST": DatasetMetadata(
        name="NLST",
        ground_truth_basis="Screening trial outcome with biopsy/histopathology and 7-year follow-up tracking",
        tier=GroundTruthTier.PATHOLOGY_CONFIRMED,
        confidence_level="Diagnostic-level",
        num_readers=None,
        is_pathology_confirmed=True,
        caveat="Provides true clinical outcome confirmation, separating benign screen-detected nodules from confirmed lung cancers."
    ),
    "REAL_PATIENT_COHORT": DatasetMetadata(
        name="REAL_PATIENT_COHORT",
        ground_truth_basis="Biopsy/surgical resection histopathology or >=2-year serial CT volumetric stability",
        tier=GroundTruthTier.PATHOLOGY_CONFIRMED,
        confidence_level="Diagnostic-level",
        num_readers=None,
        is_pathology_confirmed=True,
        caveat="Only pathology-confirmed or multi-year-follow-up-confirmed outcomes qualify as strong diagnostic ground truth."
    ),
}


def get_ground_truth_metadata(dataset_name: str) -> DatasetMetadata:
    """Retrieves standard ground truth metadata for a given dataset."""
    key = dataset_name.strip().upper()
    for name, meta in DATASET_REGISTRY.items():
        if name.upper() == key:
            return meta
    
    # Return cautious unverified metadata if unknown
    return DatasetMetadata(
        name=dataset_name,
        ground_truth_basis="Unknown / Unregistered data source",
        tier=GroundTruthTier.UNVERIFIED,
        confidence_level="Verify before use",
        num_readers=None,
        is_pathology_confirmed=False,
        caveat="Unregistered dataset. Must formally document whether ground truth is annotation-level or diagnostic-level."
    )


def validate_metric_table_ground_truth(
    entries: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Validates that a metric comparison table does NOT conflate annotation-consensus
    accuracy with pathology-confirmed accuracy without explicit tier segregation.
    
    Returns (is_valid, error_messages).
    """
    errors = []
    tiers_present = set()
    
    for idx, entry in enumerate(entries):
        tier = entry.get("ground_truth_tier")
        if not tier:
            errors.append(f"Row {idx} ('{entry.get('model', 'Unknown')}') missing mandatory 'ground_truth_tier' tag.")
        else:
            tiers_present.add(tier)
            
    # Check if multiple tiers are mixed in the same table without explicit sub-heading
    has_annotation = any("Annotation-level" in str(t) for t in tiers_present)
    has_diagnostic = any("Diagnostic-level" in str(t) for t in tiers_present)
    
    if has_annotation and has_diagnostic:
        errors.append(
            "VIOLATION OF PART AY-1: Table conflates 'Annotation-level' and 'Diagnostic-level' ground truth. "
            "Annotation-consensus accuracy and pathology-confirmed accuracy must NEVER be presented "
            "as comparable values in the same unqualified table."
        )
        
    return (len(errors) == 0, errors)
