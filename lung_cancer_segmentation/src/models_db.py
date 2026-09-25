# src/models_db.py
"""SQLAlchemy ORM models for the robust and explainable lung cancer classification platform.
Tables: users, classification_requests, segmentation_requests, training_runs, audit_logs.
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from src.database import Base
import enum


def _utcnow():
    return datetime.now(timezone.utc)


def _uuid():
    return str(uuid.uuid4())


# ─── Enums ────────────────────────────────────────────
class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelArch(str, enum.Enum):
    CNN = "cnn"
    VIT = "vit"
    HYBRID = "hybrid"


class AuditAction(str, enum.Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    UPLOAD = "upload"
    CLASSIFY = "classify"
    XAI = "xai"
    ROBUSTNESS = "robustness"
    SEGMENT = "segment"
    TRAIN = "train"
    DOWNLOAD = "download"


# ─── Users ────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(50), default="doctor")       # doctor | admin | researcher
    created_at = Column(DateTime, default=_utcnow)

    classifications = relationship("ClassificationRequest", back_populates="user")
    requests = relationship("SegmentationRequest", back_populates="user")
    audits = relationship("AuditLog", back_populates="user")


# ─── Classification & Robustness Requests ─────────────
class ClassificationRequest(Base):
    __tablename__ = "classification_requests"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    architecture = Column(String(50), default="hybrid")   # cnn, vit, hybrid
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING)

    # Predictions
    predicted_class = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    probabilities_json = Column(Text, nullable=True)      # JSON { "Benign": 0.05, "Malignant": 0.91, ... }

    # Robustness analysis
    perturbation = Column(String(100), nullable=True)    # e.g. "gaussian_noise_med"
    perturbation_severity = Column(Float, nullable=True)
    robustness_degradation = Column(Float, nullable=True) # RD = original_conf - perturbed_conf
    explanation_consistency = Column(Float, nullable=True)

    # XAI artifacts (base64 PNGs)
    gradcam_base64 = Column(Text, nullable=True)
    gradcam_pp_base64 = Column(Text, nullable=True)
    attention_base64 = Column(Text, nullable=True)
    lime_base64 = Column(Text, nullable=True)
    shap_base64 = Column(Text, nullable=True)

    inference_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="classifications")


# ─── Legacy / Segmentation Requests ───────────────────
class SegmentationRequest(Base):
    __tablename__ = "segmentation_requests"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING)
    dice_score = Column(Float, nullable=True)
    iou_score = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    result_path = Column(Text, nullable=True)
    mask_base64 = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="requests")


# ─── Training Runs ────────────────────────────────────
class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(String(36), primary_key=True, default=_uuid)
    architecture = Column(String(50), default="hybrid")
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    best_accuracy = Column(Float, nullable=True)
    best_f1 = Column(Float, nullable=True)
    best_dice = Column(Float, nullable=True)
    best_iou = Column(Float, nullable=True)
    status = Column(SAEnum(TaskStatus), default=TaskStatus.PENDING)
    checkpoint_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)


# ─── Audit Log ────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    action = Column(SAEnum(AuditAction), nullable=False)
    ip_address = Column(String(45), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="audits")
