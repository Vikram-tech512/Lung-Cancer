# src/schemas.py
"""Pydantic schemas for API request/response serialisation."""
from datetime import datetime
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, EmailStr, Field


# ─── Auth ─────────────────────────────────────────────
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ─── Classification & XAI ─────────────────────────────
class ClassificationRequestOut(BaseModel):
    id: str
    filename: str
    architecture: str
    status: str
    predicted_class: Optional[str]
    confidence: Optional[float]
    probabilities: Optional[Dict[str, float]] = None
    perturbation: Optional[str] = None
    robustness_degradation: Optional[float] = None
    explanation_consistency: Optional[float] = None
    gradcam_base64: Optional[str] = None
    gradcam_pp_base64: Optional[str] = None
    attention_base64: Optional[str] = None
    lime_base64: Optional[str] = None
    shap_base64: Optional[str] = None
    inference_time_ms: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class ClassificationSubmitOut(BaseModel):
    task_id: str
    status: str = "pending"
    message: str = "Inference task queued"


class PerturbationSweepItem(BaseModel):
    name: str
    display_name: str
    predicted_class: str
    confidence: float
    robustness_degradation: float
    explanation_consistency: float
    preview_base64: Optional[str] = None
    heatmap_base64: Optional[str] = None


class RobustnessSweepResponse(BaseModel):
    original_class: str
    original_confidence: float
    mean_robustness_degradation: float
    mean_explanation_consistency: float
    perturbations: List[PerturbationSweepItem]


# ─── Legacy / Segmentation ────────────────────────────
class SegmentationRequestOut(BaseModel):
    id: str
    filename: str
    status: str
    dice_score: Optional[float]
    iou_score: Optional[float]
    inference_time_ms: Optional[float]
    mask_base64: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ─── Training ────────────────────────────────────────
class TrainRequest(BaseModel):
    architecture: str = Field("hybrid", description="cnn, vit, or hybrid")
    epochs: int = Field(30, ge=1, le=500)
    batch_size: int = Field(16, ge=1, le=128)
    learning_rate: float = Field(1e-4, gt=0)


class TrainingRunOut(BaseModel):
    id: str
    architecture: str
    epochs: int
    batch_size: int
    learning_rate: float
    best_accuracy: Optional[float]
    best_f1: Optional[float]
    best_dice: Optional[float]
    best_iou: Optional[float]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ─── Stats & Benchmark ────────────────────────────────
class ModelBenchmarkMetric(BaseModel):
    architecture: str
    accuracy: float
    precision: float
    recall_sensitivity: float
    specificity: float
    f1_score: float
    roc_auc: float
    mean_robustness_degradation: float
    mean_explanation_consistency: float
    inference_latency_ms: float


class DashboardStats(BaseModel):
    total_classifications: int
    completed_classifications: int
    avg_confidence: Optional[float]
    avg_robustness_degradation: Optional[float]
    avg_inference_ms: Optional[float]
    active_model: str
    benchmarks: List[ModelBenchmarkMetric]
    total_users: int


# ─── Claude AI ───────────────────────────────────────────────────────────────
class ClaudeReportRequest(BaseModel):
    """Request body for POST /api/claude/report"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float] = {}
    architecture: str = "hybrid"
    inference_time_ms: Optional[float] = None
    perturbation: Optional[str] = "none"
    robustness_degradation: Optional[float] = 0.0
    explanation_consistency: Optional[float] = 1.0
    patient_context: Optional[str] = None   # Free-text clinician note


class ClaudeReportResponse(BaseModel):
    """Response body for POST /api/claude/report"""
    report: str                   # Full Markdown clinical report
    model: str                    # Claude model used
    input_tokens: int
    output_tokens: int


class ClaudeXAIRequest(BaseModel):
    """Request body for POST /api/claude/explain-xai"""
    xai_type: str                 # gradcam | gradcam_pp | attention | lime | shap
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float] = {}
    architecture: str = "hybrid"


class ClaudeXAIResponse(BaseModel):
    """Response body for POST /api/claude/explain-xai"""
    explanation: str
    model: str
    input_tokens: int
    output_tokens: int


class ClaudeRobustnessRequest(BaseModel):
    """Request body for POST /api/claude/robustness"""
    original_class: str
    original_confidence: float
    mean_robustness_degradation: float
    mean_explanation_consistency: float
    perturbations: List[Dict[str, Any]] = []


class ClaudeRobustnessResponse(BaseModel):
    """Response body for POST /api/claude/robustness"""
    summary: str
    model: str
    input_tokens: int
    output_tokens: int


class ClaudeChatMessage(BaseModel):
    role: str                     # "user" | "assistant"
    content: str


class ClaudeChatRequest(BaseModel):
    """Request body for POST /api/claude/chat"""
    messages: List[ClaudeChatMessage]
    case_context: Optional[Dict[str, Any]] = None  # Current diagnosis state


class ClaudeChatResponse(BaseModel):
    """Response body for POST /api/claude/chat"""
    reply: str
    model: str
    input_tokens: int
    output_tokens: int

