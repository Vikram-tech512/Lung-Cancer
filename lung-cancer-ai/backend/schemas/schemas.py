"""
backend/schemas/schemas.py
==========================
Pydantic v2 schemas for API requests, responses, WebSocket payloads,
and system telemetry.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: str
    role: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    role: Optional[str] = "RESEARCHER"

class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool

class QCReportOut(BaseModel):
    status: str
    slice_count: int
    slice_thickness: float
    findings: List[str]
    is_acceptable: bool

class StudyOut(BaseModel):
    id: str
    patient_id: str
    study_instance_uid: str
    modality: str
    slice_count: int
    slice_thickness: float
    qc_status: str
    created_at: str

class AnalysisJobCreate(BaseModel):
    study_id: str

class AnalysisJobOut(BaseModel):
    job_id: str
    study_id: str
    status: str
    current_stage: str
    progress_percent: int
    failure_reason: Optional[str] = None

class MorphometryOut(BaseModel):
    lesion_id: str
    area_mm2: float
    volume_mm3: float
    effective_diameter_mm: float
    sphericity: float
    spiculation_index: float
    compactness: float
    mean_hu: float
    std_hu: float
    min_hu: float
    max_hu: float

class ModelBreakdownItem(BaseModel):
    model_name: str
    raw_probability: float
    calibrated_probability: float
    uncertainty_score: float
    uncertainty_level: str
    status: str
    latency_ms: float

class VerdictOut(BaseModel):
    lesion_id: str
    consensus_verdict: str
    consensus_probability: float
    ci_lower: float
    ci_upper: float
    disagreement_entropy: float
    model_breakdown: List[ModelBreakdownItem]
    morphometry: MorphometryOut
    clinical_advisory: str

class HealthStatus(BaseModel):
    status: str
    app_name: str
    public_domain: str
    version: str
    backend: str
    database: str
    redis: str
    worker: str
    storage: str
    inference_engines: List[str]
    disclaimer: str

class DomainSettings(BaseModel):
    app_name: str
    public_domain: str
    server_ip: str
    ssl_status: str
