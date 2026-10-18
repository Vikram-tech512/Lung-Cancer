"""
backend/monitoring/__init__.py
================================
Centralised observability stack:
  - metrics   : Prometheus counters, histograms, gauges
  - logger    : Structured JSON logging (structlog)
  - tracer    : OpenTelemetry span tracing
  - health    : Liveness / readiness probe utilities
  - grafana   : Grafana dashboard provisioning helpers
"""

from .logger import get_logger
from .metrics import (
    MetricsRegistry,
    INFERENCE_LATENCY,
    INFERENCE_COUNTER,
    MODEL_ACCURACY_GAUGE,
    ACTIVE_CASES_GAUGE,
)
from .health import HealthChecker

__all__ = [
    "get_logger",
    "MetricsRegistry",
    "INFERENCE_LATENCY",
    "INFERENCE_COUNTER",
    "MODEL_ACCURACY_GAUGE",
    "ACTIVE_CASES_GAUGE",
    "HealthChecker",
]
