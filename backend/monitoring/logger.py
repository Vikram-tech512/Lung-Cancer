"""
backend/monitoring/logger.py
==============================
Centralised structured JSON logging for the Lung Cancer AI Platform.

Features
--------
* structlog-based JSON output (machine-parseable for Grafana Loki / ELK)
* Request-ID and correlation-ID injection via contextvars
* Severity filtering via LOG_LEVEL env-var
* Rotating file sink  →  logs/platform.log  (10 MB × 5 backups)
* Pretty console sink in development mode
* Zero-dependency fallback to stdlib logging if structlog is absent
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional structlog import
# ---------------------------------------------------------------------------
try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE: Path = LOG_DIR / "platform.log"
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
_LOGGERS: dict[str, logging.Logger] = {}


# ---------------------------------------------------------------------------
# File handler (rotating)
# ---------------------------------------------------------------------------
def _rotating_file_handler() -> logging.handlers.RotatingFileHandler:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":%(message)s}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    return handler


# ---------------------------------------------------------------------------
# Stdlib kwargs adapter — matches structlog's log.info("event", key=val) API
# ---------------------------------------------------------------------------
class _KwargsLoggerAdapter:
    """
    Wraps a stdlib Logger and accepts keyword arguments like structlog does.

    Usage:
        log = _KwargsLoggerAdapter(logging.getLogger("foo"))
        log.info("event_name", key="value", count=42)
        # -> "event_name | key=value count=42"
    """

    def __init__(self, inner: logging.Logger) -> None:
        self._inner = inner

    def _fmt(self, event: str, kwargs: dict) -> str:
        if not kwargs:
            return event
        kv = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{event} | {kv}"

    def debug(self, event: str, **kwargs) -> None:
        self._inner.debug(self._fmt(event, kwargs))

    def info(self, event: str, **kwargs) -> None:
        self._inner.info(self._fmt(event, kwargs))

    def warning(self, event: str, **kwargs) -> None:
        self._inner.warning(self._fmt(event, kwargs))

    def error(self, event: str, **kwargs) -> None:
        self._inner.error(self._fmt(event, kwargs))

    def critical(self, event: str, **kwargs) -> None:
        self._inner.critical(self._fmt(event, kwargs))

    def exception(self, event: str, **kwargs) -> None:
        self._inner.exception(self._fmt(event, kwargs))

    # Allow attribute pass-through for any stdlib Logger attributes
    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _stdlib_logger(name: str) -> "_KwargsLoggerAdapter":
    if name in _LOGGERS:
        return _LOGGERS[name]

    inner = logging.getLogger(name)
    inner.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not inner.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        inner.addHandler(ch)
        inner.addHandler(_rotating_file_handler())

    adapter = _KwargsLoggerAdapter(inner)
    _LOGGERS[name] = adapter
    return adapter


# ---------------------------------------------------------------------------
# structlog configuration (called once)
# ---------------------------------------------------------------------------
_STRUCTLOG_CONFIGURED = False


def _configure_structlog() -> None:
    global _STRUCTLOG_CONFIGURED
    if _STRUCTLOG_CONFIGURED:
        return

    import structlog  # noqa: PLC0415

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if ENVIRONMENT == "development":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib so uvicorn / fastapi logs are captured
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        handlers=[_rotating_file_handler(), logging.StreamHandler(sys.stdout)],
    )

    _STRUCTLOG_CONFIGURED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_logger(name: str) -> "structlog.BoundLogger | logging.Logger":
    """
    Return a logger bound to *name*.

    Returns a structlog BoundLogger when structlog is available,
    otherwise falls back to a stdlib Logger with JSON file output.

    Usage
    -----
    >>> log = get_logger(__name__)
    >>> log.info("model_loaded", model="ResNet3D-50", checkpoint="best.pt")
    """
    if _HAS_STRUCTLOG:
        _configure_structlog()
        return structlog.get_logger(name)
    return _stdlib_logger(name)


def bind_request_context(
    request_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> None:
    """
    Bind correlation IDs into the structlog context for the current async task.
    Call this at the start of every FastAPI request handler.
    """
    if _HAS_STRUCTLOG:
        import structlog  # noqa: PLC0415
        ctx: dict = {}
        if request_id:
            ctx["request_id"] = request_id
        if patient_id:
            ctx["patient_id"] = patient_id
        if model_id:
            ctx["model_id"] = model_id
        structlog.contextvars.bind_contextvars(**ctx)


def clear_request_context() -> None:
    """Clear bound contextvars (call in response middleware finally block)."""
    if _HAS_STRUCTLOG:
        import structlog  # noqa: PLC0415
        structlog.contextvars.clear_contextvars()
