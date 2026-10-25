# RESEARCH USE ONLY (RUO) — NOT A MEDICAL DEVICE

> [!CAUTION]
> **RESEARCH USE ONLY (RUO) — NOT A MEDICAL DEVICE**
> This repository and research platform are strictly intended for academic research, algorithm benchmarking, and clinical decision-support methodology exploration. This platform is **NOT a certified medical device** (FDA 510(k), CE-MDR), has not undergone clinical diagnostic validation, and must **NEVER** be used as a primary diagnostic tool, autonomous triage system, or substitute for licensed radiological evaluation. Pipeline verified end-to-end; model validation in progress.

# Lung Cancer AI Research Platform — v2.0

A modular research platform for 3D volumetric thoracic CT candidate analysis.
Built with 3D Deep Learning (UNet3D, nnUNet, RetinaNet3D, ResNet3D, DenseNet3D, Swin Transformer 3D),
IBSI-compliant Radiomics, Multimodal Ensemble models, Explainable AI (3D Grad-CAM, Integrated Gradients,
Attention Rollout), Monte Carlo Dropout uncertainty estimation, and an observability stack
(Prometheus + Grafana).

---

## Repository Structure

```text
Lung-Cancer/
├── backend/                            # FastAPI clinical backend service
│   ├── main.py                         # Application entry point
│   ├── api/                            # REST API routing layer
│   │   ├── service.py                  # Clinical endpoints, DICOM upload, /metrics, /health/*
│   │   └── v2/                         # API version 2 package namespace
│   ├── monitoring/                     # Centralized observability & telemetry
│   │   ├── logger.py                   # Structured JSON logger (structlog)
│   │   ├── metrics.py                  # Prometheus counters, histograms, and gauges
│   │   ├── health.py                   # Async health checker (disk, DB, GPU, MLflow)
│   │   ├── middleware.py               # ObservabilityMiddleware (request telemetry)
│   │   └── grafana.py                  # Grafana dashboard JSON generator
│   ├── pipeline/                       # Master ML pipeline layer (11-stage DAG)
│   │   ├── orchestrator.py             # PipelineOrchestrator DAG coordinating Stages 0–10
│   │   ├── stages/                     # Individual stage execution runners
│   │   │   ├── detection.py            # Stage 3: RetinaNet3D candidate generation
│   │   │   ├── classification.py       # Stage 4: ResNet3D / DenseNet3D / Swin3D backbone
│   │   │   ├── ensemble.py             # Stage 6: Multimodal fusion (DL + Radiomics + Clinical)
│   │   │   ├── xai.py                  # Stage 7: Grad-CAM, Integrated Gradients, Attention
│   │   │   └── report.py               # Stage 10: Lung-RADS v2022 clinical JSON generation
│   │   ├── data/                       # Data engineering & partitioning
│   │   │   ├── augmentation.py         # 3D CT volumetric spatial & intensity transforms
│   │   │   ├── splitter.py             # Patient-stratified train/val/test & k-fold splitter
│   │   │   └── loader_factory.py       # DataLoaderFactory (weighted sampling, collate)
│   │   ├── train/                      # Training subsystem
│   │   │   ├── trainer.py              # Unified trainer (AMP mixed precision, grad-accum)
│   │   │   ├── scheduler.py            # Cosine annealing & OneCycleLR schedulers
│   │   │   ├── early_stopping.py       # Validation metric patience monitoring
│   │   │   └── checkpoint_manager.py   # Versioned weights with SHA-256 verification
│   │   ├── evaluate/                   # Statistical evaluation subsystem
│   │   │   ├── evaluator.py            # Classification & segmentation metrics (AUC, F1, MCC)
│   │   │   ├── calibration_eval.py     # ECE, MCE, Brier score, reliability diagrams
│   │   │   ├── curves.py               # ROC and PR curve generation with bootstrap CIs
│   │   │   └── statistical_tests.py    # McNemar's test & Fast DeLong AUC comparison
│   │   └── radiomics/                  # Quantitative feature pipeline
│   │       ├── pipeline.py             # IBSI-compliant feature extraction with SQLite cache
│   │       └── feature_selector.py     # Variance, LASSO, RFE, and Mutual Information selection
│   ├── models/                         # 3D deep learning architectures
│   │   ├── classification/             # ResNet3D-50, ResNet3D-18 Half, DenseNet3D-121, Swin3D
│   │   ├── segmentation/               # 3D U-Net and nnU-Net wrapper
│   │   ├── detection/                  # 3D RetinaNet volumetric nodule detector
│   │   ├── radiomics/                  # PyRadiomics extractor, XGBoost, and Random Forest
│   │   └── ensemble/                   # Multimodal Ensemble Engine
│   ├── dicom/                          # DICOM processing & clinical interop
│   │   ├── loader.py                   # Pydicom volumetric series reader
│   │   ├── validator.py                # SOP Class, slice spacing, and metadata QA
│   │   ├── deidentifier.py             # HIPAA Safe Harbor DICOM de-identification
│   │   ├── compatibility.py            # Multi-vendor scanner adaptation (GE, Siemens, Philips)
│   │   └── pacs_service.py             # DICOMweb / PACS C-STORE and C-FIND service
│   ├── preprocessing/                  # Volumetric CT normalization (ct_pipeline.py)
│   ├── explainability/                 # 3D Explainable AI (gradcam3d.py, integrated_gradients.py, attention_viz3d.py)
│   ├── uncertainty/                    # Bayesian approximation (mc_dropout.py)
│   ├── calibration/                    # Post-hoc probability calibration (temperature_scaling.py)
│   ├── robustness/                     # Clinical stress-testing (ct_perturbations.py)
│   ├── datasets/                       # Real & synthetic cohort loaders (luna16.py, lidc_idri.py, synthetic_volumes.py)
│   ├── experiments/                    # Experiment tracking & model registry (models_registry.json, model_registry.py)
│   ├── database/                       # Relational database persistence (database.py, models.py, models_db.py)
│   ├── core/                           # Platform core configuration (config.py, constants.py, clinical_gate.py)
│   ├── training/                       # Dedicated model training scripts (train_luna16_3d.py, train_classifier.py)
│   ├── tests/                          # Automated pytest test suites
│   ├── requirements.txt                # Python backend dependencies
│   └── Dockerfile                      # Backend container definition
│
├── frontend/                           # React + TypeScript clinical workstation UI
│   ├── src/                            # Application source (App.tsx, main.tsx, index.css, panels/, components/, services/)
│   ├── public/                         # Static assets and reference sample cases
│   ├── Dockerfile                      # Multi-stage production container build
│   ├── nginx.conf                      # Production Nginx reverse-proxy configuration
│   ├── package.json                    # Frontend dependencies and Vite build scripts
│   └── vite.config.ts                  # Vite build and dev server configuration
│
├── deploy/                             # Observability & deployment configuration
│   ├── prometheus/                     # Prometheus metrics scrape configuration
│   └── grafana/provisioning/           # Automated dashboard and datasource provisioning
│
├── experiments/                        # Scientific benchmark experiments
│   └── run_ieee_experiments.py         # IEEE benchmark evaluation runner
│
├── paper/                              # IEEE manuscript & documentation
│   ├── ieee_lung_cancer_paper.tex      # IEEE conference / journal manuscript
│   ├── robust_explainable_hybrid_vit.tex
│   ├── template.tex
│   ├── references.bib                  # BibTeX bibliography
│   └── figures/                        # Generated figures and evaluation charts
│
├── scripts/                            # Automation scripts
│   └── run_all.py                      # Master platform test and verification suite
│
├── data/                               # Local datasets (gitignored; preserves .gitkeep)
├── checkpoints/                        # Model weights (gitignored; preserves .gitkeep)
├── results/                            # Experimental findings & reports (gitignored; preserves .gitkeep)
├── logs/                               # Runtime logs (gitignored; preserves .gitkeep)
├── docker-compose.yml                  # Full-stack orchestrator: backend + frontend + Prometheus + Grafana
├── pyproject.toml                      # Unified Python build tooling, ruff, mypy, pytest config
├── Makefile                            # Developer commands (dev, test, lint, docker-up, run-all)
├── .gitignore                          # Repository gitignore definitions
└── README.md                           # Master project documentation
```

---

## Quick Start

### Docker Compose (Full Observability Stack)

Copy `.env.example` to `.env` and set `GF_SECURITY_ADMIN_PASSWORD`:
```bash
cp .env.example .env
docker compose up --build -d
```

| Service    | URL                                          |
|------------|----------------------------------------------|
| Frontend   | http://localhost:3000                        |
| Backend API (Swagger) | http://localhost:8000/docs      |
| Prometheus | http://localhost:9090                        |
| Grafana    | http://localhost:3001 (auth via GF_SECURITY_ADMIN_PASSWORD) |

---

### Local Development

```bash
# Backend
pip install -r backend/requirements.txt
python -m backend.main --reload

# Frontend
cd frontend && npm install && npm run dev
```

Or use the Makefile:
```bash
make install          # install all Python deps
make dev-backend      # hot-reload FastAPI on :8000
make dev-frontend     # Vite dev server on :5173
make docker-up        # full observability stack
make run-all          # master platform validation suite
```

---

## Testing

```bash
# All tests
make test

# With coverage
make test-cov

# Smoke tests (architecture tensor shape verification)
make smoke-test

# Pipeline tests only
python -m pytest backend/tests/test_pipeline.py -v
```

### Smoke tests

The architecture smoke test suite verifies tensor dimension contracts, layer forward passes, and gradient backward propagation across all 3D volumetric models:

```text
backend/tests/test_models.py::test_unet3d_forward PASSED
backend/tests/test_models.py::test_resnet3d_forward PASSED
backend/tests/test_models.py::test_densenet3d_forward PASSED
backend/tests/test_models.py::test_swin3d_forward PASSED
backend/tests/test_models.py::test_ensemble_forward PASSED
```

---

## Observability Endpoints

| Endpoint          | Description                               |
|-------------------|-------------------------------------------|
| `GET /metrics`    | Prometheus text exposition (scrape target)|
| `GET /health/live`| Liveness probe (process alive)            |
| `GET /health/ready`| Readiness probe (all subsystems)         |

Prometheus Metrics exposed:
- `lung_cancer_inference_latency_ms` — histogram per model
- `lung_cancer_inference_total` — counter per model x outcome
- `lung_cancer_model_accuracy` — AUC / Sensitivity / Specificity gauges
- `lung_cancer_training_epoch_loss` — per model per split
- `lung_cancer_training_epoch_auc` — per model per split
- `lung_cancer_gpu_memory_bytes` — per CUDA device
- `lung_cancer_dicom_validation_total` — QA gate outcomes
- `lung_cancer_radiomics_duration_seconds` — PyRadiomics timing

---

## ML Pipeline Stages

The `PipelineOrchestrator` runs 11 stages as an async DAG:

| Stage Number | Stage Name | Module |
|---|-------|--------|
| 0 | DICOM QA | `dicom/validator.py` |
| 1 | CT Preprocessing | `preprocessing/ct_pipeline.py` |
| 2 | Segmentation | `backend/segmentation/service.py` |
| 3 | Detection | `pipeline/stages/detection.py` |
| 4 | Classification | `pipeline/stages/classification.py` |
| 5 | Radiomics | `pipeline/radiomics/pipeline.py` |
| 6 | Ensemble | `pipeline/stages/ensemble.py` |
| 7 | XAI | `pipeline/stages/xai.py` |
| 8 | Uncertainty | `uncertainty/mc_dropout.py` |
| 9 | Calibration | `calibration/temperature_scaling.py` |
| 10 | Report | `pipeline/stages/report.py` (Lung-RADS) |

Usage:
```python
from backend.pipeline import PipelineOrchestrator

orch = PipelineOrchestrator.from_config()
report = await orch.run("/data/cases/CASE_001")
print(report.to_dict())
```

---

## Core Capabilities

1. **Intake & QA Gate** — Automated 10-point DICOM check (slice thickness, gantry tilt, HU range, spatial resolution)
2. **3D MPR Viewer** — Multi-planar reconstruction (Axial, Coronal, Sagittal) with synchronized crosshairs and 3D Grad-CAM overlays
3. **Multimodal Ensemble** — Combines DL representations with IBSI PyRadiomics features and clinical metadata
4. **Epistemic Uncertainty** — MC Dropout estimation with predictive variance and temperature-scaled calibration
5. **Robustness Benchmark** — Automated stress-testing against photon noise, motion artifacts, and slice decimation
6. **Centralized Observability** — Structured JSON logs -> Prometheus metrics -> Grafana dashboards
7. **Statistical Evaluation** — Bootstrap AUC CI, McNemar test, Fast DeLong paired AUC comparison
8. **Findings & Structured Reporting** — Nodule candidate measurements and radiologist review workflow

---

## Pipeline Verification Status

Pipeline verified end-to-end; model validation in progress.

> [!NOTE]
> All unit and integration tests verify code execution, input/output tensor shape compliance, and interface contracts. Model weights are experimental benchmark checkpoints awaiting full cross-validation before clinical promotion. Not for clinical decisions.
