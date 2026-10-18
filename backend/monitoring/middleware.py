"""
backend/monitoring/middleware.py
=================================
FastAPI middleware for:
  1. Structured request/response logging  (JSON lines)
  2. Automatic Prometheus metrics recording per request
  3. Request-ID injection (X-Request-ID header)
  4. Correlation-ID context binding (structlog contextvars)

Add to FastAPI app with:
    app.add_middleware(ObservabilityMiddleware)
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logger import get_logger, bind_request_context, clear_request_context
from .metrics import INFERENCE_COUNTER, INFERENCE_LATENCY

log = get_logger(__name__)

# Paths that are too noisy to log at INFO level
_SILENT_PATHS = {"/metrics", "/health/live", "/health/ready", "/favicon.ico"}


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Transparent observability layer — adds logging, metrics, and
    request-ID correlation to every HTTP request.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Assign or propagate request ID
        request_id: str = (
            request.headers.get("X-Request-ID") or str(uuid.uuid4())
        )
        bind_request_context(request_id=request_id)

        start_ns = time.perf_counter_ns()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            log.error(
                "unhandled_exception",
                path=request.url.path,
                method=request.method,
                error=str(exc),
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

            # Skip noise paths
            if request.url.path not in _SILENT_PATHS:
                log.info(
                    "http_request",
                    method=request.method,
                    path=request.url.path,
                    status=status_code,
                    latency_ms=round(elapsed_ms, 2),
                    request_id=request_id,
                )

            # Record generic HTTP metric (not model-specific)
            outcome = "2xx" if 200 <= status_code < 300 else f"{status_code // 100}xx"
            INFERENCE_COUNTER.labels(model="http", outcome=outcome).inc()

            clear_request_context()

        # Propagate request-ID back to caller
        response.headers["X-Request-ID"] = request_id
        return response
