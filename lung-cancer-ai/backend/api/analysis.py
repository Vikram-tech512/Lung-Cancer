"""
backend/api/analysis.py
=======================
Analysis pipeline trigger and status polling endpoints.
Connects directly to Celery background task worker and Redis pub/sub.
"""

from __future__ import annotations
import uuid
import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import AnalysisJob, Study, User
from backend.schemas.schemas import AnalysisJobCreate, AnalysisJobOut
from backend.auth.security import get_current_user

router = APIRouter(prefix="/analysis", tags=["Analysis Pipeline"])

@router.post("", response_model=AnalysisJobOut)
def start_analysis(
    payload: AnalysisJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    study = db.query(Study).filter(Study.id == payload.study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
    job = AnalysisJob(
        id=job_id,
        study_id=study.id,
        user_id=current_user.id,
        status="QUEUED",
        current_stage="QUEUED",
        progress_percent=0
    )
    db.add(job)
    db.commit()

    # Dispatch to Celery task queue or resilient fallback worker
    try:
        from worker.tasks.analysis_pipeline import run_lung_ct_analysis
        # Attempt Celery async dispatch
        run_lung_ct_analysis.delay(job.id)
    except Exception:
        # Fallback background task if Celery broker (Redis) is in stand-by
        from worker.tasks.analysis_pipeline import run_lung_ct_analysis
        background_tasks.add_task(run_lung_ct_analysis, job.id)

    return AnalysisJobOut(
        job_id=job.id,
        study_id=study.id,
        status=job.status,
        current_stage=job.current_stage,
        progress_percent=job.progress_percent
    )

@router.get("/{job_id}", response_model=AnalysisJobOut)
def get_analysis_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return AnalysisJobOut(
        job_id=job.id,
        study_id=job.study_id,
        status=job.status,
        current_stage=job.current_stage,
        progress_percent=job.progress_percent,
        failure_reason=job.failure_reason
    )
