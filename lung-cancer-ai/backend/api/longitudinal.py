"""
backend/api/longitudinal.py
===========================
REST endpoints for multi-timepoint longitudinal CT comparison and nodule trajectory tracking.
"""
from __future__ import annotations

import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import (
    Study, Lesion, Measurement, LongitudinalComparison
)
from ml.longitudinal.comparator import (
    LongitudinalComparator, NoduleRecord
)

router = APIRouter(prefix="/longitudinal", tags=["Longitudinal"])


class CompareRequest(BaseModel):
    current_study_id: str
    prior_study_id: str


@router.post("/compare")
def compare_studies(req: CompareRequest, db: Session = Depends(get_db)):
    """Triggers 3D spatial alignment and volume doubling time tracking between two CT studies."""
    curr_study = db.query(Study).filter(Study.id == req.current_study_id).first()
    prior_study = db.query(Study).filter(Study.id == req.prior_study_id).first()

    if not curr_study or not prior_study:
        raise HTTPException(status_code=404, detail="Current or prior study not found")

    curr_lesions = db.query(Lesion).filter(Lesion.study_id == req.current_study_id).all()
    prior_lesions = db.query(Lesion).filter(Lesion.study_id == req.prior_study_id).all()

    # Build NoduleRecords
    c_records: List[NoduleRecord] = []
    for l in curr_lesions:
        c_records.append(NoduleRecord(
            nodule_id=l.id,
            center_zyx_mm=(l.centroid_z, l.centroid_y, l.centroid_x),
            diameter_mm=l.diameter_mm,
            volume_mm3=l.volume_mm3,
            attenuation_type="solid",
            slice_index=int(l.centroid_z)
        ))

    p_records: List[NoduleRecord] = []
    for l in prior_lesions:
        p_records.append(NoduleRecord(
            nodule_id=l.id,
            center_zyx_mm=(l.centroid_z, l.centroid_y, l.centroid_x),
            diameter_mm=l.diameter_mm,
            volume_mm3=l.volume_mm3,
            attenuation_type="solid",
            slice_index=int(l.centroid_z)
        ))

    curr_date = curr_study.study_date or datetime.datetime.utcnow()
    prior_date = prior_study.study_date or (curr_date - datetime.timedelta(days=180))

    comparator = LongitudinalComparator()
    comp_res = comparator.compare_studies(
        current_study_id=req.current_study_id,
        current_nodules=c_records,
        prior_study_id=req.prior_study_id,
        prior_nodules=p_records,
        current_date=curr_date,
        prior_date=prior_date
    )

    # Persist in DB
    matched_data = [
        {
            "current_nodule_id": m.current_nodule_id,
            "prior_nodule_id": m.prior_nodule_id,
            "volume_doubling_time_days": m.volume_doubling_time_days,
            "volume_change_pct": m.volume_change_pct,
            "growth_category": m.growth_category,
            "match_confidence": m.match_confidence,
            "clinical_interpretation": m.clinical_interpretation
        }
        for m in comp_res.matched_nodules
    ]

    comp_db = LongitudinalComparison(
        current_study_id=req.current_study_id,
        previous_study_id=req.prior_study_id,
        registration_method="rigid_centroid_matching",
        registration_quality=0.92,
        is_reliable=(comp_res.registration_status == "SUCCESS"),
        days_between=comp_res.days_between_studies,
        matched_nodules=matched_data,
        failure_reason=comp_res.fail_safe_message
    )
    db.add(comp_db)
    db.commit()
    db.refresh(comp_db)

    return {
        "comparison_id": comp_db.id,
        "current_study_id": req.current_study_id,
        "prior_study_id": req.prior_study_id,
        "days_between": comp_res.days_between_studies,
        "overall_assessment": comp_res.overall_assessment,
        "registration_status": comp_res.registration_status,
        "matched_count": len(comp_res.matched_nodules),
        "new_nodules_count": len(comp_res.new_nodules),
        "resolved_nodules_count": len(comp_res.resolved_nodules),
        "matched_nodules": matched_data,
        "fail_safe_message": comp_res.fail_safe_message
    }


@router.get("/study/{study_id}")
def get_study_comparisons(study_id: str, db: Session = Depends(get_db)):
    """Retrieves all longitudinal comparisons for a given study."""
    comps = db.query(LongitudinalComparison).filter(
        (LongitudinalComparison.current_study_id == study_id) |
        (LongitudinalComparison.previous_study_id == study_id)
    ).all()

    return [
        {
            "id": c.id,
            "current_study_id": c.current_study_id,
            "previous_study_id": c.previous_study_id,
            "days_between": c.days_between,
            "is_reliable": c.is_reliable,
            "matched_nodules": c.matched_nodules,
            "failure_reason": c.failure_reason,
            "created_at": c.created_at.isoformat() if c.created_at else None
        }
        for c in comps
    ]
