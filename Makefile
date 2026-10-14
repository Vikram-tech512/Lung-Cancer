# ============================================================================
#  Lung Cancer AI Platform — Makefile
#  Unified dev commands for the entire monorepo
# ============================================================================

.PHONY: help install dev-backend dev-frontend test lint format \
        docker-up docker-down generate-dashboard clean

PYTHON   := python
PIP      := pip
NPM      := npm
DOCKER   := docker compose

# Default target
help:
	@echo ""
	@echo "  Lung Cancer AI Platform — Developer Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install          Install all Python dependencies"
	@echo "  make dev-backend      Start FastAPI backend (hot-reload)"
	@echo "  make dev-frontend     Start Vite frontend (hot-reload)"
	@echo "  make test             Run pytest test suite"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Ruff lint check"
	@echo "  make format           Ruff auto-fix formatting"
	@echo "  make docker-up        Start full stack (backend + frontend + Prometheus + Grafana)"
	@echo "  make docker-down      Stop and remove all containers"
	@echo "  make generate-dashboard  Generate Grafana dashboard JSON"
	@echo "  make clean            Remove __pycache__, .pyc, build artifacts"
	@echo ""

# ─── Python ──────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r backend/requirements.txt

dev-backend:
	$(PYTHON) -m backend.main --reload --port 8000

# ─── Frontend ────────────────────────────────────────────────────────────────
dev-frontend:
	cd frontend && $(NPM) run dev

# ─── Testing ─────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest backend/tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest backend/tests/ -v --tb=short \
		--cov=backend --cov-report=term-missing --cov-report=html

# ─── Code Quality ────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m ruff check backend/ --output-format=concise

format:
	$(PYTHON) -m ruff check backend/ --fix
	$(PYTHON) -m ruff format backend/

typecheck:
	$(PYTHON) -m mypy backend/ --ignore-missing-imports

# ─── Docker / Observability ───────────────────────────────────────────────────
docker-up:
	$(DOCKER) up --build -d
	@echo ""
	@echo "  ✅ Services started:"
	@echo "     Frontend    →  http://localhost:3000"
	@echo "     Backend API →  http://localhost:8000/docs"
	@echo "     Prometheus  →  http://localhost:9090"
	@echo "     Grafana     →  http://localhost:3001  (admin / lungcancerAI2024)"
	@echo ""

docker-down:
	$(DOCKER) down

docker-logs:
	$(DOCKER) logs -f backend

# ─── AWS CloudFront Deployment ──────────────────────────────────────────────
deploy-cloudfront:
	powershell -ExecutionPolicy Bypass -File deploy/deploy-cloudfront.ps1

# ─── Zoho Catalyst Deployment ────────────────────────────────────────────────
deploy-zoho:
	powershell -ExecutionPolicy Bypass -File deploy/deploy-zoho.ps1

# ─── Monitoring ──────────────────────────────────────────────────────────────
generate-dashboard:
	$(PYTHON) -c "from backend.monitoring.grafana import generate_dashboard; generate_dashboard('deploy/grafana/provisioning/dashboards')"
	@echo "Dashboard JSON written to deploy/grafana/provisioning/dashboards/"

# ─── Pipeline ────────────────────────────────────────────────────────────────
train-resnet3d:
	$(PYTHON) backend/training/train_luna16_3d.py --model resnet3d --epochs 100

train-densenet3d:
	$(PYTHON) backend/training/train_luna16_3d.py --model densenet3d --epochs 100

train-swin3d:
	$(PYTHON) backend/training/train_luna16_3d.py --model swin3d --epochs 50

smoke-test:
	$(PYTHON) -m pytest backend/tests/test_models.py -v

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .ruff_cache/ .mypy_cache/ .pytest_cache/
	@echo "Cleaned."
