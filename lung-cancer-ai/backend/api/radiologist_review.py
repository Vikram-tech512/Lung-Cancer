"""
backend/api/radiologist_review.py
=================================
REST endpoints for radiologist interactive review, annotation, editing, and clinical sign-off.

SAFETY & AUDIT INTEGRITY MANDATE (§25):
  - Stores AI findings, Radiologist annotations, and Final clinical verdicts in distinct columns.
  - The original AI model prediction is permanently preserved and NEVER overwritten.
  - Generates immutable audit trail entries upon any review or amendment.
"""
from __future__ import annotations

import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import (
    RadiologistReview, Study, Lesion, Measurement, Prediction, AuditLog, User
)

router = APIRouter(prefix="/reviews", tags=["Radiologist Review"])


class ReviewSubmission(BaseModel):
    study_id: str
    lesion_id: Optional[str] = None
    action: str                             # ACCEPT, REJECT, MODIFY, ADD_FINDING, MARK_FP, MARK_MISSED
    radiologist_verdict: Optional[str] = None
    radiologist_measurements: Optional[Dict[str, Any]] = None
    radiologist_comments: Optional[str] = None
    final_verdict: Optional[str] = None
    final_measurements: Optional[Dict[str, Any]] = None


@router.post("")
def submit_review(sub: ReviewSubmission, db: Session = Depends(get_db)):
    """Submits a radiologist review record, keeping original AI and human modifications separate."""
    study = db.query(Study).filter(Study.id == sub.study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    ai_verdict = None
    ai_prob = None
    ai_meas = {}

    if sub.lesion_id:
        lesion = db.query(Lesion).filter(Lesion.id == sub.lesion_id).first()
        pred = db.query(Prediction).filter(Prediction.lesion_id == sub.lesion_id).first()
        meas = db.query(Measurement).filter(Measurement.lesion_id == sub.lesion_id).first()

        if pred:
            ai_verdict = pred.consensus_verdict
            ai_prob = pred.consensus_probability
        if meas:
            ai_meas = {
                "diameter_mm": meas.effective_diameter_mm,
                "volume_mm3": meas.volume_mm3,
                "sphericity": meas.sphericity
            }

    review = RadiologistReview(
        study_id=sub.study_id,
        lesion_id=sub.lesion_id,
        action=sub.action,
        ai_verdict=ai_verdict,
        ai_probability=ai_prob,
        ai_measurements=ai_meas,
        radiologist_verdict=sub.radiologist_verdict,
        radiologist_measurements=sub.radiologist_measurements or {},
        radiologist_comments=sub.radiologist_comments,
        final_verdict=sub.final_verdict or sub.radiologist_verdict or ai_verdict,
        final_measurements=sub.final_measurements or sub.radiologist_measurements or ai_meas,
        model_version="v2.0.0-production",
        created_at=datetime.datetime.utcnow()
    )
    db.add(review)

    # Append to Audit Log
    audit = AuditLog(
        action=f"RADIOLOGIST_REVIEW_{sub.action}",
        resource_type="STUDY",
        resource_id=sub.study_id,
        timestamp=datetime.datetime.utcnow()
    )
    db.add(audit)

    db.commit()
    db.refresh(review)

    return {
        "review_id": review.id,
        "action": review.action,
        "ai_verdict_preserved": review.ai_verdict,
        "radiologist_verdict": review.radiologist_verdict,
        "final_verdict": review.final_verdict,
        "created_at": review.created_at.isoformat()
    }


@router.get("/study/{study_id}")
def get_study_reviews(study_id: str, db: Session = Depends(get_db)):
    """Fetches all radiologist reviews and annotations for a study."""
    reviews = db.query(RadiologistReview).filter(RadiologistReview.study_id == study_id).order_by(RadiologistReview.created_at.desc()).all()

    return [
        {
            "id": r.id,
            "study_id": r.study_id,
            "lesion_id": r.lesion_id,
            "action": r.action,
            "ai_verdict": r.ai_verdict,
            "ai_probability": r.ai_probability,
            "radiologist_verdict": r.radiologist_verdict,
            "radiologist_comments": r.radiologist_comments,
            "final_verdict": r.final_verdict,
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in reviews
    ]
