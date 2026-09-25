# src/celery_app.py
"""Celery application configuration and task definitions.
Broker and backend default to Redis.
"""
from celery import Celery
from src.config import CELERY_BROKER, CELERY_BACKEND

celery = Celery(
    "lung_seg",
    broker=CELERY_BROKER,
    backend=CELERY_BACKEND,
)

celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Import tasks so Celery discovers them
celery.autodiscover_tasks(["src.tasks"])
