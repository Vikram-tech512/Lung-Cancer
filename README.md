# Lung Cancer AI Research Platform 

A **clinical-grade, full-stack AI platform** for volumetric thoracic CT analysis.
Built with 3D Deep Learning (UNet3D, nnUNet, RetinaNet3D, ResNet3D, DenseNet3D, Swin Transformer 3D),
IBSI-compliant Radiomics, Multimodal Ensemble learning, Explainable AI (3D Grad-CAM, Integrated Gradients,
Attention Rollout), Monte Carlo Dropout uncertainty estimation, and a **full observability stack**
(Prometheus + Grafana).

---

## Repository Structure

```
Lung Cancer/
│
├── backend/                            # FastAPI v2.0 Backend Service
│   ├── main.py                         # Entry point (uvicorn)
│   ├── __init__.py                     # Package root (monitoring-first init)
│   ├── requirements.txt                # All Python dependencies
│   ├── Dockerfile
│   │
│   ├── api/                            # REST API routers & endpoints
│   │   └── service.py                  # All clinical endpoints + /metrics + /health/*
│   │
│   ├── monitoring/                     # ★ NEW — Centralised Observability
│   │   ├── __init__.py                 # Public surface
│   │   ├── logger.py                   # Structured JSON logger (structlog)
│   │   ├── metrics.py                  # Prometheus counters, histograms, gauges
│   │   ├── health.py                   # Async health checker (disk/DB/GPU/MLflow)
│   │   ├── middleware.py               # ObservabilityMiddleware (auto request logging)
│   │   └── grafana.py                  # Grafana dashboard JSON generator
│   │
│   ├── pipeline/                       # ★ NEW — Full ML Pipeline Layer
│   │   ├── __init__.py
│   │   ├── orchestrator.py             # Master DAG orchestrator (11 stages)
│   │   │
│   │   ├── stages/                     # Individual stage runners
│   │   │   ├── __init__.py
│   │   │   ├── detection.py            # Stage 3: RetinaNet3D nodule detection
│   │   │   ├── classification.py       # Stage 4: ResNet3D / DenseNet3D / SwinTransformer3D
│   │   │   ├── ensemble.py             # Stage 6: Multimodal fusion (DL+Radiomics+Clinical)
│   │   │   ├── xai.py                  # Stage 7: Grad-CAM + IG + Attention Rollout
│   │   │   └── report.py               # Stage 10: Clinical JSON report (Lung-RADS)
│   │   │
│   │   ├── data/                       # Data pipeline utilities
│   │   │   ├── __init__.py
│   │   │   ├── augmentation.py         # 8-transform 3D CT augmentation pipeline
│   │   │   ├── splitter.py             # Patient-stratified train/val/test + k-fold
│   │   │   └── loader_factory.py       # DataLoaderFactory (balanced sampler, pad-collate)
│   │   │
│   │   ├── train/                      # Training pipeline
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py              # Unified trainer (AMP, grad-accum, MLflow)
│   │   │   ├── scheduler.py            # LR scheduler factory (cosine/onecycle/plateau)
│   │   │   ├── early_stopping.py       # Patience-based early stopping
│   │   │   └── checkpoint_manager.py   # Versioned checkpoints + SHA-256 integrity
│   │   │
│   │   ├── evaluate/                   # Evaluation pipeline
│   │   │   ├── __init__.py
│   │   │   ├── evaluator.py            # 12+ classification metrics (AUC, MCC, Kappa…)
│   │   │   ├── calibration_eval.py     # ECE, MCE, Brier score, reliability diagrams
│   │   │   ├── curves.py               # ROC/PR curves + bootstrap 95% CI
│   │   │   └── statistical_tests.py    # McNemar + Fast DeLong AUC comparison
│   │   │
│   │   └── radiomics/                  # Radiomics pipeline
│   │       ├── __init__.py
│   │       ├── pipeline.py             # IBSI PyRadiomics + SQLite cache + XGB scoring
│   │       └── feature_selector.py     # Variance / Correlation / LASSO / RFE / MI
│   │
│   ├── models/                         # 3D Neural Network Architectures
│   │   ├── classification/             # ResNet3D-50, DenseNet3D-121, SwinTransformer3D
│   │   ├── segmentation/               # 3D U-Net, nnU-Net wrapper
│   │   ├── detection/                  # 3D RetinaNet nodule detector
│   │   ├── radiomics/                  # PyRadiomics extractor, XGBoost, Random Forest
│   │   └── ensemble/                   # Multimodal Ensemble Engine
│   │
│   ├── dicom/                          # DICOM pipeline (loader, validator, de-id, PACS)
│   ├── preprocessing/                  # CT windowing & isotropic resampling
│   ├── explainability/                 # 3D Grad-CAM, IG, Attention Rollout
│   ├── uncertainty/                    # MC Dropout + predictive entropy
│   ├── calibration/                    # Temperature scaling
│   ├── robustness/                     # CT perturbation stress-testing
│   ├── datasets/                       # LIDC-IDRI, LUNA16, NLST loaders
│   ├── experiments/                    # MLflow tracker + model registry
│   ├── database/                       # SQLAlchemy models + case log
│   ├── core/                           # Config, constants, clinical gate middleware
│   ├── training/                       # Standalone training scripts
│   └── tests/                          # pytest suites (models + pipeline + monitoring)
│
├── frontend/                           # Medical Glassmorphism React + TypeScript App
│   ├── src/
│   │   ├── App.tsx                     # Main dashboard shell
│   │   ├── index.css                   # Dark-mode clinical UI
│   │   ├── types.ts                    # TypeScript interfaces
│   │   ├── panels/                     # 10 interactive clinical panels
│   │   ├── components/                 # Shared UI components
│   │   └── services/                   # API service layer
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
│
├── deploy/                             # Deployment & Infrastructure
│   ├── prometheus/
│   │   └── prometheus.yml              # Prometheus scrape config (10s interval)
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── prometheus.yml      # Auto-provisioned Prometheus datasource
│           └── dashboards/
│               ├── provider.yml        # Dashboard folder provider
│               └── lung_cancer_ai.json # ★ Auto-generated 10-panel dashboard
│
├── lung_cancer_segmentation/           # Standalone segmentation benchmark module
├── lung-cancer-ai/                     # Full AI monorepo variant
├── data/                               # CT datasets (LIDC-IDRI, LUNA16)
├── checkpoints/                        # Model weight files (.pt / .pth)
├── results/                            # XAI heatmaps, clinical reports, radiomics cache
├── logs/                               # Structured JSON log files (rotating 10MB × 5)
├── docker-compose.yml                  # ★ Full stack: backend+frontend+Prometheus+Grafana
├── pyproject.toml                      # Build config, pytest, ruff, mypy, coverage
├── Makefile                            # Unified developer commands
└── README.md
```

---

## Quick Start

### Docker Compose (Full Observability Stack)
```bash
docker-compose up --build -d
```

| Service    | URL                                          |
|------------|----------------------------------------------|
| Frontend   | http://localhost:3000                        |
| Backend API (Swagger) | http://localhost:8000/docs      |
| Prometheus | http://localhost:9090                        |
| Grafana    | http://localhost:3001  (admin / lungcancerAI2024) |

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
```

---

## Testing

```bash
# All tests
make test

# With coverage
make test-cov

# Model smoke tests only
make smoke-test

# Pipeline tests only
python -m pytest backend/tests/test_pipeline.py -v
```

---

## Observability Endpoints

| Endpoint          | Description                               |
|-------------------|-------------------------------------------|
| `GET /metrics`    | Prometheus text exposition (scrape target)|
| `GET /health/live`| Liveness probe (process alive)            |
| `GET /health/ready`| Readiness probe (all subsystems)         |

**Prometheus Metrics exposed:**
- `lung_cancer_inference_latency_ms` — histogram per model
- `lung_cancer_inference_total` — counter per model × outcome
- `lung_cancer_model_accuracy` — AUC / Sensitivity / Specificity gauges
- `lung_cancer_training_epoch_loss` — per model per split
- `lung_cancer_training_epoch_auc` — per model per split
- `lung_cancer_gpu_memory_bytes` — per CUDA device
- `lung_cancer_dicom_validation_total` — QA gate outcomes
- `lung_cancer_radiomics_duration_seconds` — PyRadiomics timing

---

## ML Pipeline Stages

The `PipelineOrchestrator` runs 11 stages as an async DAG:

| # | Stage | Module |
|---|-------|--------|
| 0 | DICOM QA | `dicom/validator.py` |
| 1 | CT Preprocessing | `preprocessing/ct_pipeline.py` |
| 2 | Segmentation | `segmentation/inference.py` |
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
2. **3D MPR Viewer** — Simultaneous Axial, Coronal, Sagittal viewports with synchronised crosshairs and 3D Grad-CAM overlays
3. **Multimodal Ensemble** — Combines DL representations with IBSI PyRadiomics features and clinical metadata
4. **Epistemic Uncertainty** — MC Dropout estimation with predictive variance and temperature-scaled calibration
5. **Robustness Benchmark** — Automated stress-testing against photon noise, motion artifacts, and slice decimation
6. **Centralised Observability** — Structured JSON logs → Prometheus metrics → Grafana dashboards (auto-provisioned)
7. **Statistical Evaluation** — Bootstrap AUC CI, McNemar's test, Fast DeLong paired AUC comparison
8. **Lung-RADS Classification** — Automated categorisation (1 → 4X) with clinical recommendations

---

## Model Test Results

```
UNet3D              (Volumetric Segmentation)          → PASS
ResNet3D-50         (Volumetric Classification)        → PASS
DenseNet3D-121      (Dense Feature Re-use)             → PASS
SwinTransformer3D   (Shifted-Window Attention)         → PASS
MultimodalEnsemble  (DL + Radiomics + Clinical)        → PASS
PipelineOrchestrator (Full 11-stage async DAG)         → PASS
CalibrationEvaluator (ECE / Brier / Reliability)       → PASS
CurveGenerator       (ROC + Bootstrap CI)              → PASS
StatisticalTests     (McNemar + DeLong AUC)            → PASS
RadiomicsPipeline    (IBSI + SQLite cache)             → PASS
```
