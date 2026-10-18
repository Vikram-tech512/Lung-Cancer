"""
backend/monitoring/metrics.py
================================
Prometheus metrics registry for the Lung Cancer AI Platform.

Exposes
-------
* /metrics  endpoint (prometheus_client WSGI handler mounted in FastAPI)
* Inference latency histogram  (per model)
* Inference request counter    (per model × outcome)
* Active cases gauge
* Model accuracy gauge         (per model)
* Training epoch loss & AUC
* GPU memory utilisation (optional)

Usage
-----
>>> from backend.monitoring.metrics import INFERENCE_LATENCY, record_inference
>>> with INFERENCE_LATENCY.labels(model="ResNet3D").time():
...     result = model(volume)

>>> record_inference("ResNet3D", outcome="malignant", latency_ms=312.4)
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator, Optional

# ---------------------------------------------------------------------------
# Optional prometheus_client import
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        multiprocess,
        REGISTRY,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

# ---------------------------------------------------------------------------
# Bucket boundaries for inference latency (milliseconds)
# ---------------------------------------------------------------------------
LATENCY_BUCKETS_MS = (
    50, 100, 200, 300, 500, 750,
    1_000, 1_500, 2_000, 3_000, 5_000,
    float("inf"),
)


# ---------------------------------------------------------------------------
# Lazy singleton registry wrapper
# ---------------------------------------------------------------------------
class _NullMetric:
    """Drop-in replacement when prometheus_client is unavailable."""

    def labels(self, **_kw) -> "_NullMetric":
        return self

    def inc(self, *_a, **_kw) -> None: ...
    def dec(self, *_a, **_kw) -> None: ...
    def set(self, *_a, **_kw) -> None: ...
    def observe(self, *_a, **_kw) -> None: ...
    def time(self) -> "_NullCtx":
        return _NullCtx()


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): ...


# ---------------------------------------------------------------------------
# Define metrics
# ---------------------------------------------------------------------------
if _HAS_PROMETHEUS:
    INFERENCE_LATENCY = Histogram(
        "lung_cancer_inference_latency_ms",
        "End-to-end inference latency in milliseconds",
        labelnames=["model"],
        buckets=LATENCY_BUCKETS_MS,
    )

    INFERENCE_COUNTER = Counter(
        "lung_cancer_inference_total",
        "Total inference requests processed",
        labelnames=["model", "outcome"],  # outcome: malignant|benign|error
    )

    MODEL_ACCURACY_GAUGE = Gauge(
        "lung_cancer_model_accuracy",
        "Latest evaluated accuracy/AUC for each model",
        labelnames=["model", "metric"],  # metric: accuracy|auc|sensitivity|specificity
    )

    ACTIVE_CASES_GAUGE = Gauge(
        "lung_cancer_active_cases",
        "Number of CT cases currently being processed",
    )

    TRAINING_EPOCH_LOSS = Gauge(
        "lung_cancer_training_epoch_loss",
        "Training loss at current epoch",
        labelnames=["model", "split"],   # split: train|val
    )

    TRAINING_EPOCH_AUC = Gauge(
        "lung_cancer_training_epoch_auc",
        "AUC-ROC at current epoch",
        labelnames=["model", "split"],
    )

    DICOM_VALIDATION_COUNTER = Counter(
        "lung_cancer_dicom_validation_total",
        "DICOM QA gate validation results",
        labelnames=["outcome"],   # passed|failed|warning
    )

    GPU_MEMORY_GAUGE = Gauge(
        "lung_cancer_gpu_memory_bytes",
        "GPU memory allocated in bytes",
        labelnames=["device"],
    )

    RADIOMICS_EXTRACTION_DURATION = Summary(
        "lung_cancer_radiomics_duration_seconds",
        "Time spent extracting PyRadiomics features",
        labelnames=["feature_class"],
    )

else:
    INFERENCE_LATENCY = _NullMetric()
    INFERENCE_COUNTER = _NullMetric()
    MODEL_ACCURACY_GAUGE = _NullMetric()
    ACTIVE_CASES_GAUGE = _NullMetric()
    TRAINING_EPOCH_LOSS = _NullMetric()
    TRAINING_EPOCH_AUC = _NullMetric()
    DICOM_VALIDATION_COUNTER = _NullMetric()
    GPU_MEMORY_GAUGE = _NullMetric()
    RADIOMICS_EXTRACTION_DURATION = _NullMetric()


# ---------------------------------------------------------------------------
# MetricsRegistry  — convenience facade
# ---------------------------------------------------------------------------
class MetricsRegistry:
    """
    High-level API for recording platform metrics.

    All methods are no-ops when prometheus_client is not installed,
    so the rest of the codebase never needs to guard metric calls.
    """

    @staticmethod
    def record_inference(
        model: str,
        outcome: str,
        latency_ms: float,
    ) -> None:
        """Record one completed inference pass."""
        INFERENCE_LATENCY.labels(model=model).observe(latency_ms)
        INFERENCE_COUNTER.labels(model=model, outcome=outcome).inc()

    @staticmethod
    def set_model_metric(model: str, metric: str, value: float) -> None:
        """Update a live model evaluation metric gauge."""
        MODEL_ACCURACY_GAUGE.labels(model=model, metric=metric).set(value)

    @staticmethod
    def set_training_metrics(
        model: str,
        epoch_loss_train: float,
        epoch_loss_val: float,
        auc_train: Optional[float] = None,
        auc_val: Optional[float] = None,
    ) -> None:
        """Push training epoch metrics to Prometheus."""
        TRAINING_EPOCH_LOSS.labels(model=model, split="train").set(epoch_loss_train)
        TRAINING_EPOCH_LOSS.labels(model=model, split="val").set(epoch_loss_val)
        if auc_train is not None:
            TRAINING_EPOCH_AUC.labels(model=model, split="train").set(auc_train)
        if auc_val is not None:
            TRAINING_EPOCH_AUC.labels(model=model, split="val").set(auc_val)

    @staticmethod
    def record_dicom_validation(outcome: str) -> None:
        """Increment DICOM QA gate counter. outcome ∈ {passed, failed, warning}."""
        DICOM_VALIDATION_COUNTER.labels(outcome=outcome).inc()

    @staticmethod
    def update_gpu_memory() -> None:
        """Refresh GPU memory gauges (no-op if torch/CUDA unavailable)."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem = torch.cuda.memory_allocated(i)
                    GPU_MEMORY_GAUGE.labels(device=f"cuda:{i}").set(mem)
        except Exception:
            pass

    @staticmethod
    @contextmanager
    def time_inference(model: str) -> Generator[None, None, None]:
        """Context manager — measures inference wall-clock time in ms."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            INFERENCE_LATENCY.labels(model=model).observe(elapsed_ms)

    @staticmethod
    def prometheus_response() -> tuple[bytes, str]:
        """
        Return (body_bytes, content_type) for a /metrics HTTP response.
        Mount this in FastAPI with:
            @app.get("/metrics")
            def metrics_endpoint():
                body, ctype = MetricsRegistry.prometheus_response()
                return Response(content=body, media_type=ctype)
        """
        if not _HAS_PROMETHEUS:
            return b"# prometheus_client not installed\n", "text/plain"
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
