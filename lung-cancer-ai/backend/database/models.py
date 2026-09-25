"""
backend/database/models.py
==========================
Comprehensive 20-table relational schema for clinical CT AI research platform.
Strict foreign keys, constraints, audit trails, and indexes.
"""

from __future__ import annotations
import uuid
import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON, Enum
)
from sqlalchemy.orm import relationship
from backend.database.database import Base

def gen_uuid():
    return str(uuid.uuid4())

# 1. Users
class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="RESEARCHER", nullable=False)  # ADMIN, RESEARCHER, VIEWER
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    analysis_jobs = relationship("AnalysisJob", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

# 2. Roles
class Role(Base):
    __tablename__ = "roles"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    name = Column(String(50), unique=True, nullable=False)
    permissions = Column(JSON, default=list)

# 3. UserRoles link
class UserRole(Base):
    __tablename__ = "user_roles"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"))
    role_id = Column(String(36), ForeignKey("roles.id", ondelete="CASCADE"))

# 4. Patients (De-identified research cohort)
class Patient(Base):
    __tablename__ = "patients"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    research_patient_id = Column(String(100), unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    biological_sex = Column(String(20), nullable=True)
    smoking_pack_years = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    studies = relationship("Study", back_populates="patient", cascade="all, delete-orphan")

# 5. Studies
class Study(Base):
    __tablename__ = "studies"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    patient_id = Column(String(36), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    study_instance_uid = Column(String(128), unique=True, index=True, nullable=False)
    study_date = Column(String(32), nullable=True)
    modality = Column(String(16), default="CT")
    slice_count = Column(Integer, default=0)
    slice_thickness = Column(Float, default=1.0)
    pixel_spacing_xy = Column(JSON, default=list)
    manufacturer = Column(String(128), nullable=True)
    qc_status = Column(String(32), default="PENDING")  # PASS, WARNING, REJECT
    qc_findings = Column(JSON, default=list)
    storage_folder = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    patient = relationship("Patient", back_populates="studies")
    series = relationship("Series", back_populates="study", cascade="all, delete-orphan")
    analysis_jobs = relationship("AnalysisJob", back_populates="study")
    lesions = relationship("Lesion", back_populates="study")
    reports = relationship("Report", back_populates="study")

# 6. Series
class Series(Base):
    __tablename__ = "series"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    series_instance_uid = Column(String(128), unique=True, index=True, nullable=False)
    series_description = Column(String(255), nullable=True)
    acquisition_number = Column(Integer, default=1)
    
    study = relationship("Study", back_populates="series")
    images = relationship("Image", back_populates="series", cascade="all, delete-orphan")

# 7. Images (Slices)
class Image(Base):
    __tablename__ = "images"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    series_id = Column(String(36), ForeignKey("series.id", ondelete="CASCADE"), nullable=False)
    slice_index = Column(Integer, nullable=False)
    z_position = Column(Float, nullable=True)
    sop_instance_uid = Column(String(128), nullable=True)
    storage_path = Column(String(255), nullable=False)
    
    series = relationship("Series", back_populates="images")

# 8. Analysis Jobs
class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(32), default="QUEUED", index=True) # QUEUED, PROCESSING, COMPLETED, FAILED
    current_stage = Column(String(64), default="QUEUED")
    progress_percent = Column(Integer, default=0)
    failure_reason = Column(Text, nullable=True)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    study = relationship("Study", back_populates="analysis_jobs")
    user = relationship("User", back_populates="analysis_jobs")

# 9. Lesions (Candidate Nodule ROIs)
class Lesion(Base):
    __tablename__ = "lesions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    lesion_index = Column(Integer, default=1)
    centroid_z = Column(Float, nullable=False)
    centroid_y = Column(Float, nullable=False)
    centroid_x = Column(Float, nullable=False)
    diameter_mm = Column(Float, nullable=False)
    volume_mm3 = Column(Float, nullable=False)
    anatomical_lobe = Column(String(64), nullable=True)
    preliminary_risk = Column(String(64), nullable=True)

    study = relationship("Study", back_populates="lesions")
    segmentations = relationship("Segmentation", back_populates="lesion")
    predictions = relationship("Prediction", back_populates="lesion")
    explanations = relationship("Explanation", back_populates="lesion")
    measurements = relationship("Measurement", back_populates="lesion")
    radiomics = relationship("RadiomicFeature", back_populates="lesion")

# 10. Segmentations
class Segmentation(Base):
    __tablename__ = "segmentations"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=True)
    segmentation_type = Column(String(32), nullable=False)  # LUNG, NODULE
    mask_storage_path = Column(String(255), nullable=False)
    dice_score = Column(Float, nullable=True)
    volume_voxels = Column(Integer, default=0)
    
    lesion = relationship("Lesion", back_populates="segmentations")

# 11. Predictions (Consensus Ensemble Verdict)
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)
    consensus_verdict = Column(String(32), nullable=False)  # MALIGNANT, BENIGN, INDETERMINATE
    consensus_probability = Column(Float, nullable=False)
    ci_lower = Column(Float, default=0.0)
    ci_upper = Column(Float, default=1.0)
    disagreement_entropy = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    lesion = relationship("Lesion", back_populates="predictions")
    model_predictions = relationship("ModelPrediction", back_populates="prediction", cascade="all, delete-orphan")

# 12. Model Predictions (Per-Architecture breakdown)
class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    prediction_id = Column(String(36), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(64), nullable=False)
    raw_probability = Column(Float, nullable=False)
    calibrated_probability = Column(Float, nullable=False)
    uncertainty_score = Column(Float, default=0.0)
    uncertainty_level = Column(String(16), default="LOW")  # LOW, MODERATE, HIGH
    status = Column(String(32), default="ACTIVE")
    latency_ms = Column(Float, default=0.0)

    prediction = relationship("Prediction", back_populates="model_predictions")

# 13. Explanations (XAI)
class Explanation(Base):
    __tablename__ = "explanations"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)
    method = Column(String(64), nullable=False)  # GRAD_CAM_3D, INTEGRATED_GRADIENTS, ATTENTION_ROLLOUT
    heatmap_storage_path = Column(String(255), nullable=False)
    colormap = Column(String(32), default="jet")
    attribution_disclaimer = Column(String(255), default="MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE")
    reliability_score = Column(Float, default=0.88)
    reliability_category = Column(String(16), default="HIGH")  # HIGH, MODERATE, LOW
    forces_abstention = Column(Boolean, default=False)

    lesion = relationship("Lesion", back_populates="explanations")

# 14. Measurements (Morphometry)
class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)
    area_mm2 = Column(Float, default=0.0)
    volume_mm3 = Column(Float, default=0.0)
    effective_diameter_mm = Column(Float, default=0.0)
    sphericity = Column(Float, default=0.0)
    spiculation_index = Column(Float, default=0.0)
    compactness = Column(Float, default=0.0)
    mean_hu = Column(Float, default=0.0)
    std_hu = Column(Float, default=0.0)
    min_hu = Column(Float, default=0.0)
    max_hu = Column(Float, default=0.0)

    lesion = relationship("Lesion", back_populates="measurements")

# 15. Radiomic Features (IBSI 107)
class RadiomicFeature(Base):
    __tablename__ = "radiomics"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)
    feature_count = Column(Integer, default=107)
    feature_vector = Column(JSON, default=dict)
    ibsi_compliant = Column(Boolean, default=True)

    lesion = relationship("Lesion", back_populates="radiomics")

# 16. Experiments (MLflow lineage)
class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    experiment_name = Column(String(128), index=True, nullable=False)
    model_name = Column(String(64), nullable=False)
    dataset_name = Column(String(64), nullable=False)
    hyperparams = Column(JSON, default=dict)
    val_metrics = Column(JSON, default=dict)
    patient_split_hash = Column(String(64), nullable=False)
    mlflow_run_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# 17. AI Models Registry
class AIModel(Base):
    __tablename__ = "models"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    model_name = Column(String(64), unique=True, nullable=False)
    version = Column(String(32), default="v1.0.0")
    stage = Column(String(32), default="NOT_TRAINED")  # PRODUCTION, STAGING, NOT_TRAINED, ARCHIVED
    architecture_details = Column(Text, nullable=False)
    input_dimension = Column(String(64), nullable=False)
    weights_storage_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# 18. Datasets
class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    dataset_name = Column(String(64), unique=True, nullable=False)
    version = Column(String(32), default="v1.0")
    source_url = Column(String(255), nullable=True)
    doi = Column(String(64), nullable=True)
    license = Column(String(64), default="Creative Commons")
    patient_count = Column(Integer, default=0)
    patient_split_verified = Column(Boolean, default=True)

# 19. Reports
class Report(Base):
    __tablename__ = "reports"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    report_pdf_path = Column(String(255), nullable=False)
    summary_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    study = relationship("Study", back_populates="reports")

# 20. Audit Logs
class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(64), nullable=True)
    ip_address = Column(String(64), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")

# ─────────────────────────────────────────────────────────────
#  NEW TABLES (Phase 2 expansion)
# ─────────────────────────────────────────────────────────────

# 21. Radiologist Reviews — stores AI, radiologist, and final results separately (§25)
#     NEVER overwrites original AI output.
class RadiologistReview(Base):
    __tablename__ = "radiologist_reviews"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=True)
    reviewer_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # AI result (frozen at time of review)
    ai_verdict = Column(String(64), nullable=True)
    ai_probability = Column(Float, nullable=True)
    ai_measurements = Column(JSON, default=dict)

    # Radiologist result
    action = Column(String(32), nullable=False)  # ACCEPT, REJECT, MODIFY, ADD_FINDING, MARK_FP, MARK_MISSED
    radiologist_verdict = Column(String(64), nullable=True)
    radiologist_measurements = Column(JSON, default=dict)
    radiologist_comments = Column(Text, nullable=True)

    # Final result (after radiologist review)
    final_verdict = Column(String(64), nullable=True)
    final_measurements = Column(JSON, default=dict)

    # Model tracking
    model_version = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    study = relationship("Study")
    lesion = relationship("Lesion")
    reviewer = relationship("User")


# 22. Longitudinal Comparison — links current/previous studies with matching results (§14)
class LongitudinalComparison(Base):
    __tablename__ = "longitudinal_comparisons"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    current_study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    previous_study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)

    # Registration quality
    registration_method = Column(String(64), default="rigid")
    registration_quality = Column(Float, nullable=True)  # 0-1 score
    is_reliable = Column(Boolean, default=False)

    # Time interval
    days_between = Column(Integer, nullable=True)

    # Matched nodules (JSON array of {current_lesion_id, previous_lesion_id, match_confidence})
    matched_nodules = Column(JSON, default=list)

    # If registration failed
    failure_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    current_study = relationship("Study", foreign_keys=[current_study_id])
    previous_study = relationship("Study", foreign_keys=[previous_study_id])


# 23. OOD (Out-of-Distribution) Results — flags cases outside training distribution (§18)
class OODResult(Base):
    __tablename__ = "ood_results"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=True)

    is_ood = Column(Boolean, default=False)
    ood_score = Column(Float, default=0.0)  # Higher = more OOD
    ood_method = Column(String(64), default="mahalanobis")
    ood_reasons = Column(JSON, default=list)  # ["unusual_scanner", "extreme_spacing", ...]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    study = relationship("Study")
    lesion = relationship("Lesion")


# 24. False Positive Analysis — categorised FP tracking (§19)
class FalsePositiveAnalysis(Base):
    __tablename__ = "false_positive_analyses"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)

    is_false_positive = Column(Boolean, default=False)
    fp_category = Column(String(64), nullable=True)  # vascular, pleural, scar, motion, anatomical, other
    fp_confidence = Column(Float, default=0.0)
    vessel_proximity_score = Column(Float, default=0.0)
    pleural_proximity_score = Column(Float, default=0.0)
    anatomical_context = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    study = relationship("Study")
    lesion = relationship("Lesion")


# 25. Morphology Analysis — nodule morphological characterisation (§10)
class MorphologyAnalysis(Base):
    __tablename__ = "morphology_analyses"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)

    # Density classification
    density_type = Column(String(32), nullable=True)  # solid, part-solid, ground-glass
    density_confidence = Column(Float, default=0.0)

    # Margin characterisation
    margin_type = Column(String(32), nullable=True)  # smooth, lobulated, spiculated, irregular
    margin_confidence = Column(Float, default=0.0)

    # Special features
    is_calcified = Column(Boolean, default=False)
    is_cavitary = Column(Boolean, default=False)
    is_pleural_associated = Column(Boolean, default=False)
    is_vessel_associated = Column(Boolean, default=False)
    is_fissural = Column(Boolean, default=False)

    # Shape metrics
    sphericity = Column(Float, default=0.0)
    elongation = Column(Float, default=0.0)
    compactness = Column(Float, default=0.0)
    surface_area_mm2 = Column(Float, default=0.0)

    # Location
    anatomical_lobe = Column(String(64), nullable=True)
    lung_side = Column(String(16), nullable=True)  # LEFT, RIGHT

    # AI disclaimer: never claims histopathological subtype
    analysis_disclaimer = Column(String(255), default=(
        "Imaging characteristics only. Does not establish histopathological subtype."
    ))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    lesion = relationship("Lesion")


# 26. Risk Prediction — calibrated malignancy risk with clinical variables (§15)
class RiskPrediction(Base):
    __tablename__ = "risk_predictions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)

    # Calibrated risk estimate
    risk_score = Column(Float, nullable=False)  # 0-1 calibrated probability
    risk_category = Column(String(64), nullable=False)  # ELEVATED_MODEL_RISK, MODERATE, LOW, INDETERMINATE

    # Confidence interval
    ci_lower = Column(Float, default=0.0)
    ci_upper = Column(Float, default=1.0)

    # What inputs were actually used
    ct_features_used = Column(Boolean, default=True)
    clinical_features_used = Column(Boolean, default=False)
    growth_data_used = Column(Boolean, default=False)
    clinical_data_available = Column(JSON, default=dict)  # Which clinical vars were supplied
    missing_clinical_vars = Column(JSON, default=list)    # Which were unavailable

    # Model info
    model_name = Column(String(64), nullable=True)
    model_version = Column(String(32), nullable=True)
    calibration_method = Column(String(32), default="temperature_scaling")

    # Disclaimer
    risk_disclaimer = Column(String(255), default=(
        "Model-estimated malignancy risk. Not a pathological diagnosis. "
        "Radiologist interpretation required."
    ))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    lesion = relationship("Lesion")
    study = relationship("Study")


# 27. Uncertainty Result — per-nodule uncertainty estimation (§16)
class UncertaintyResult(Base):
    __tablename__ = "uncertainty_results"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)

    # Uncertainty metrics
    method = Column(String(32), nullable=False)  # mc_dropout, deep_ensemble, conformal
    mean_probability = Column(Float, nullable=False)
    std_probability = Column(Float, nullable=False)
    entropy = Column(Float, default=0.0)
    uncertainty_category = Column(String(16), nullable=False)  # LOW, MODERATE, HIGH
    n_passes = Column(Integer, default=30)

    # Model agreement (for ensembles)
    model_agreement = Column(String(16), default="N/A")  # HIGH, MODERATE, LOW
    disagreement_entropy = Column(Float, default=0.0)

    # Image quality factor
    image_quality = Column(String(16), default="GOOD")  # GOOD, ACCEPTABLE, POOR

    # Abstention trigger
    requires_review = Column(Boolean, default=False)
    abstention_reason = Column(Text, nullable=True)

    # Raw distribution samples (JSON array)
    distribution_samples = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    lesion = relationship("Lesion")


# 28. Lung-RADS Assessment — optional Lung-RADS-oriented reporting (§22)
class LungRADSAssessment(Base):
    __tablename__ = "lungrads_assessments"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    study_id = Column(String(36), ForeignKey("studies.id", ondelete="CASCADE"), nullable=False)
    lesion_id = Column(String(36), ForeignKey("lesions.id", ondelete="CASCADE"), nullable=False)

    # AI suggestion (never overwritten)
    ai_suggested_category = Column(String(16), nullable=True)  # 1, 2, 3, 4A, 4B, 4X, S
    ai_rationale = Column(Text, nullable=True)
    has_required_screening_info = Column(Boolean, default=False)

    # Radiologist modification
    radiologist_category = Column(String(16), nullable=True)
    radiologist_rationale = Column(Text, nullable=True)

    # Final category
    final_category = Column(String(16), nullable=True)
    finalized_by = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    finalized_at = Column(DateTime, nullable=True)

    # Disclaimer
    assessment_disclaimer = Column(String(255), default=(
        "Candidate Lung-RADS-oriented assessment. The AI has not independently "
        "established the final clinical category."
    ))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    study = relationship("Study")
    lesion = relationship("Lesion")
    finalizer = relationship("User")
