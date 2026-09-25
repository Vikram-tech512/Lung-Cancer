# Dockerfile – Multi‑stage build for API + Celery Worker
# ─────────────────────────────────────────────────────────
# SECURITY NOTE: Never bake ANTHROPIC_API_KEY into the image.
# Pass it at runtime:
#   docker run -e ANTHROPIC_API_KEY="sk-ant-..." pulmotrust-api
# Or use docker-compose with an .env file (see docker-compose.yml).
# ─────────────────────────────────────────────────────────

# ─────────── Builder stage ───────────
FROM python:3.12-slim AS builder
WORKDIR /app

# System deps for psycopg2 & bcrypt compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

# ─────────── Runtime stage ───────────
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 && rm -rf /var/lib/apt/lists/*

# Copy installed packages and application code from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Remove any .env file that may have been copied (safety guard)
RUN rm -f /app/.env

# Create non‑root user for security
RUN useradd --create-home appuser
USER appuser

# ─── Runtime environment defaults (safe, non-secret values only) ───
# Secret values (ANTHROPIC_API_KEY, SECRET_KEY, DATABASE_URL) must be
# supplied at runtime via -e flags or docker-compose env_file.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Health check — hits /api/health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "src.inference.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

