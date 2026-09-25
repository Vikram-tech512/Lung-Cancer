# src/config.py
"""Centralised application configuration.
All settings are loaded from environment variables with sensible defaults.
"""
import os
from pathlib import Path

# Load .env file if present (development convenience)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — rely on OS env vars


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", str(CHECKPOINT_DIR / "vit_lung_seg.pt"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
PATCH_SIZE = int(os.getenv("PATCH_SIZE", "16"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
DEPTH = int(os.getenv("DEPTH", "12"))
HEADS = int(os.getenv("HEADS", "12"))
MLP_DIM = int(os.getenv("MLP_DIM", "3072"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

# ──────────────────────────────────────────────
# Database (PostgreSQL)
# ──────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./lung_seg.db"          # fallback for local dev without Postgres
)

# ──────────────────────────────────────────────
# Redis / Celery
# ──────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER = os.getenv("CELERY_BROKER", REDIS_URL)
CELERY_BACKEND = os.getenv("CELERY_BACKEND", REDIS_URL)

# ──────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-super-secret-key-2024")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ALGORITHM = "HS256"

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
).split(",")

# ──────────────────────────────────────────────
# Claude AI (Anthropic)
# ──────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
# Maximum tokens Claude may return per request
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "2048"))

