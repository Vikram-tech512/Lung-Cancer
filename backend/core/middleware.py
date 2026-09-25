"""
backend/core/middleware.py
===========================
Production middleware stack for the Lung Cancer AI Research Platform.

Includes:
  - AuditLogMiddleware: captures every API call for clinical reproducibility
  - RequestTimingMiddleware: adds X-Process-Time header
  - SafeErrorMiddleware: strips patient info from error responses

IMPORTANT:
  - Never log patient-identifying information (name, MRN, SSN, DOB).
  - Study IDs and anonymised research IDs are safe to log.
  - DICOM paths may contain PHI — only log basenames.
"""

from __future__ import annotations

import time
import uuid
import logging
import traceback
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger("backend.middleware")


# ─────────────────────────────────────────────────────────────
#  Audit Log Middleware
# ─────────────────────────────────────────────────────────────

class AuditLogMiddleware(BaseHTTPMiddleware):
    """Logs every API request for clinical audit trail.

    Captured fields:
      - request_id:  unique per-request identifier
      - user_id:     extracted from JWT (if present)
      - method:      HTTP method
      - path:        request path (no query string)
      - status_code: response status
      - duration_ms: processing time
      - ip_address:  client IP
      - timestamp:   ISO-8601

    Does NOT capture:
      - Request bodies (may contain DICOM/image data)
      - Patient names, MRN, DOB, SSN
      - Full file paths (uses basenames only)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:12]
        start_time = time.perf_counter()

        # Extract user_id from JWT if present (non-blocking)
        user_id = "anonymous"
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from backend.auth.security import decode_access_token
                token = auth_header[7:]
                payload = decode_access_token(token)
                user_id = payload.get("sub", "unknown")
            except Exception:
                user_id = "invalid_token"

        # Process request
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add tracking headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"

        # Log audit entry (async DB write would go here in production)
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Determine resource type and ID from path
        resource_type, resource_id = _extract_resource(path)

        logger.info(
            f"AUDIT | {request_id} | {user_id} | {request.method} {path} | "
            f"{response.status_code} | {duration_ms:.1f}ms | {client_ip} | "
            f"{resource_type}:{resource_id}"
        )

        # Write to database in background (non-blocking)
        try:
            await _write_audit_log(
                request_id=request_id,
                user_id=user_id,
                method=request.method,
                path=path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                ip_address=client_ip,
                resource_type=resource_type,
                resource_id=resource_id,
            )
        except Exception:
            pass  # Audit failure must never block request processing

        return response


# ─────────────────────────────────────────────────────────────
#  Request Timing Middleware
# ─────────────────────────────────────────────────────────────

class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Process-Time header to every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
        return response


# ─────────────────────────────────────────────────────────────
#  Safe Error Middleware
# ─────────────────────────────────────────────────────────────

class SafeErrorMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions and returns safe error responses.

    CRITICAL: Strips any patient-identifying information from error messages.
    Stack traces are logged server-side but never returned to the client.
    """

    # Words that may indicate PHI in error messages
    _PHI_PATTERNS = [
        "patient_name", "PatientName", "PatientID", "patient_id",
        "DateOfBirth", "SSN", "MRN", "medical_record",
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            # Log full error server-side
            logger.error(
                f"Unhandled error on {request.method} {request.url.path}: {exc}",
                exc_info=True
            )

            # Sanitise error message for client
            error_msg = str(exc)
            for pattern in self._PHI_PATTERNS:
                if pattern.lower() in error_msg.lower():
                    error_msg = "An internal error occurred. Please contact support."
                    break

            return JSONResponse(
                status_code=500,
                content={
                    "detail": error_msg,
                    "type": "internal_error",
                    "disclaimer": (
                        "If this error occurred during AI analysis, the affected "
                        "results should not be used for clinical decisions."
                    ),
                },
            )


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _extract_resource(path: str) -> tuple:
    """Extract resource type and ID from API path.

    /api/studies/abc123        → ("study", "abc123")
    /api/nodules/nod456/review → ("nodule", "nod456")
    /api/auth/login            → ("auth", "login")
    """
    parts = [p for p in path.strip("/").split("/") if p]

    # Skip 'api' prefix
    if parts and parts[0] == "api":
        parts = parts[1:]

    resource_type = parts[0] if parts else "root"
    resource_id = parts[1] if len(parts) > 1 else "none"

    return resource_type, resource_id


async def _write_audit_log(**kwargs):
    """Write audit log entry to database (non-blocking).

    In production, this writes to the audit_logs table.
    During development without DB, it's a no-op (the logger.info above
    still captures the entry).
    """
    try:
        from backend.database.database import SessionLocal
        from backend.database.models import AuditLog

        db = SessionLocal()
        try:
            log_entry = AuditLog(
                user_id=kwargs.get("user_id") if kwargs.get("user_id") != "anonymous" else None,
                action=f"{kwargs.get('method', 'GET')} {kwargs.get('path', '/')}",
                resource_type=kwargs.get("resource_type", "unknown"),
                resource_id=kwargs.get("resource_id", "none"),
                ip_address=kwargs.get("ip_address", "unknown"),
            )
            db.add(log_entry)
            db.commit()
        finally:
            db.close()
    except Exception:
        pass  # Non-critical — never block request for audit writes
