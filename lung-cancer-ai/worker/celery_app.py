"""
worker/celery_app.py
====================
Celery task queue initialization with Redis broker for asynchronous
background thoracic CT analysis and heavy 3D deep learning execution.
"""

from __future__ import annotations
import os
from celery import Celery
from backend.core.config import settings

celery_app = Celery(
    "lung_cancer_ai_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["worker.tasks.analysis_pipeline"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max for 3D fullres models
    worker_prefetch_multiplier=1,
)
