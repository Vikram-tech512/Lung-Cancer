"""
backend/__init__.py
=====================
Lung Cancer AI Platform — Backend Package Root.

Exposes the top-level public surface:
  - app        : FastAPI application instance
  - pipeline   : PipelineOrchestrator (full ML DAG)
  - monitoring : logging, metrics, health utilities

Import order matters for circular-dependency avoidance:
  monitoring → pipeline → api
"""

# Monitoring must be initialised before any other import
from backend.monitoring import get_logger  # noqa: F401

__version__ = "2.0.0"
__author__  = "Lung Cancer AI Research Team"
