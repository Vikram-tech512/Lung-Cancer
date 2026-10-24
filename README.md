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
Lung Cancer/
│
├── .github/                            # GitHub Actions CI/CD workflows and automation
├── backend/                            # FastAPI v2.0 Clinical Backend Service
│   ├── main.py                         # Application entry point (Uvicorn ASGI launcher)
│   ├── __init__.py                     # Package root (monitoring-first initialization)
│   ├── requirements.txt                # Python backend dependencies
│   ├── Dockerfile                      # Backend container definition
│   │
│   ├── api/                            # REST API routing layer
│   │   ├── service.py                  # Clinical endpoints, DICOM upload, /metrics, /health/*
│   │   └── v2/                         # API version 2 package namespace
│   │
│   ├── monitoring/                     # Centralized Observability & Telemetry
│   │   ├── logger.py                   # Structured JSON logger (structlog)
│   │   ├── metrics.py                  # Prometheus counters, histograms, and gauges
│   │   ├── health.py                   # Async health checker (disk, DB, GPU, MLflow)
│   │   ├── middleware.py               # ObservabilityMiddleware (request telemetry)
│   │   └── grafana.py                  # Grafana dashboard JSON generator
│   │
│   ├── pipeline/                       # Master ML Pipeline Layer (11-Stage DAG)
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
│   │
│   ├── models/                         # 3D Deep Learning Architectures
│   │   ├── classification/             # ResNet3D-50, DenseNet3D-121, SwinTransformer3D, ViT
│   │   ├── segmentation/               # 3D U-Net and nnU-Net wrapper
│   │   ├── detection/                  # 3D RetinaNet volumetric nodule detector
│   │   ├── radiomics/                  # PyRadiomics extractor, XGBoost, and Random Forest
│   │   └── ensemble/                   # Multimodal Ensemble Engine
│   │
│   ├── dicom/                          # DICOM processing & clinical interop
│   │   ├── loader.py                   # Pydicom volumetric series reader
│   │   ├── validator.py                # SOP Class, slice spacing, and metadata QA
│   │   ├── deidentifier.py             # HIPAA Safe Harbor DICOM de-identification
│   │   ├── compatibility.py            # Multi-vendor scanner adaptation (GE, Siemens, Philips)
│   │   └── pacs_service.py             # DICOMweb / PACS C-STORE and C-FIND service
│   │
│   ├── preprocessing/                  # Volumetric CT normalization
│   │   └── ct_pipeline.py              # HU windowing (-1000 to 400 HU), 1mm isotropic resampling
│   ├── explainability/                 # 3D Explainable AI (XAI)
│   │   ├── gradcam3d.py                # Volumetric Grad-CAM & MultiScaleGradCAM3D
│   │   ├── attention_viz3d.py          # Window-shift-aware Attention Rollout
│   │   ├── integrated_gradients.py     # Path-integral feature attributions
│   │   └── localization_metrics.py     # Pointing Game accuracy & heatmap-to-mask Dice/IoU
│   ├── uncertainty/                    # Bayesian approximation
│   │   └── mc_dropout.py               # Monte Carlo Dropout (T=30), entropy & variance tiers
│   ├── calibration/                    # Post-hoc probability calibration
│   │   └── temperature_scaling.py      # Temperature scaling on validation logits (NLL loss)
│   ├── robustness/                     # Clinical stress-testing
│   │   └── ct_perturbations.py         # 5 CT physical perturbations (HU shifts, noise, rings)
│   ├── datasets/                       # Real & synthetic cohort loaders
│   │   ├── luna16.py                   # LUNA16 dataset loader with in-memory candidate indexing
│   │   ├── lidc_idri.py                # LIDC-IDRI 4-radiologist consensus agreement parser
│   │   ├── histopathology.py           # Histopathology 2D image dataset loader
│   │   ├── patient_split.py            # Patient-level isolation hashing
│   │   └── synthetic_volumes.py        # Synthetic test volumes for zero-data smoke testing
│   ├── experiments/                    # Experiment tracking
│   │   ├── mlflow_tracker.py           # Local MLflow JSON run manifest logger
│   │   └── model_registry.py           # Model version lifecycle manager (STAGING/PRODUCTION)
│   ├── database/                       # Relational database persistence
│   │   ├── database.py                 # SQLite/PostgreSQL SQLAlchemy engine setup
│   │   ├── models.py                   # Relational ORM models (Patients, Studies, Lesions)
│   │   └── models_db.py                # Clinical case log and inference tables
│   ├── core/                           # Platform core configuration
│   │   ├── config.py                   # Environment settings and paths
│   │   ├── constants.py                # Clinical thresholds and DICOM tags
│   │   ├── clinical_gate.py            # Diagnostic safety gate & threshold rules
│   │   └── middleware.py               # Request tracing and security headers
│   ├── training/                       # Dedicated model training scripts
│   │   ├── train_luna16_3d.py          # Full LUNA16 training, calibration & robustness script
│   │   ├── train_classifier.py         # Multi-model classification trainer
│   │   └── train_histonet.py           # Histopathology baseline trainer
│   └── tests/                          # Automated pytest test suites
│       ├── test_pipeline.py            # End-to-end pipeline & evaluator unit tests
│       ├── test_models.py              # Forward pass & architecture shape verification
│       ├── test_xai.py                 # 3D Grad-CAM & attribution invariant checks
│       ├── test_dicom_upload.py        # DICOM QA & anonymization test suite
│       └── test_hospital_interop.py    # Multi-vendor compatibility verification
│
├── frontend/                           # React + TypeScript Clinical Workstation UI
│   ├── src/
│   │   ├── App.tsx                     # Main workstation layout shell & route navigation
│   │   ├── main.tsx                    # React 18 client root mounting
│   │   ├── index.css                   # Medical glassmorphism CSS design system
│   │   ├── types.ts                    # Clinical data & API TypeScript definitions
│   │   ├── panels/                     # 10 specialized clinical workstation panels
│   │   │   ├── WorkstationPanel.tsx    # Primary DICOM viewer & 3D volumetric navigator
│   │   │   ├── CaseLogPanel.tsx        # Longitudinal patient case audit log
│   │   │   ├── CTViewerPanel.tsx       # Multi-planar slice reconstruction (MPR)
│   │   │   ├── TrainingPanel.tsx       # Real-time training & convergence monitoring
│   │   │   ├── VerdictPanel.tsx        # Model candidate output & calibrated probability display
│   │   │   ├── AIAnalysisPanel.tsx     # Experimental frontier AI text analysis proxy
│   │   │   ├── PuterAnalysisPanel.tsx  # Experimental Puter cloud analysis integration
│   │   │   ├── ExperimentsPanel.tsx    # MLflow runs, hyperparameter & metric tracker
│   │   │   ├── RadiomicsPanel.tsx      # IBSI feature visualizer & feature importance
│   │   │   └── ClinicalReportPanel.tsx # Structured findings report export (PDF/JSON)
│   │   ├── components/                 # Shared clinical UI components
│   │   │   ├── TumourLocator.tsx       # 3D lesion bounding box visualizer
│   │   │   ├── EKGWaveform.tsx         # Decorative monitor waveform (unused in clinical mode)
│   │   │   └── AIAssistantExplanation.tsx # Attribution explainer
│   │   └── services/
│   │       └── api.ts                  # Axios client connecting to backend API
│   ├── Dockerfile                      # Multi-stage production container build
│   ├── nginx.conf                      # Production Nginx reverse-proxy configuration
│   └── package.json                    # Frontend dependencies and Vite build scripts
│
├── deploy/                             # Observability & Deployment Configuration
│   ├── prometheus/
│   │   └── prometheus.yml              # Prometheus metrics scrape configuration (10s interval)
│   ├── grafana/
│   │   └── provisioning/               # Automated dashboard and datasource provisioning
│   │       ├── datasources/
│   │       │   └── prometheus.yml      # Auto-provisioned Prometheus datasource
│   │       └── dashboards/
│   │           ├── provider.yml        # Grafana dashboard provider definition
│   │           └── lung_cancer_ai.json # 10-panel clinical observability dashboard
│   ├── aws-deploy.ps1                  # AWS cloud deployment script
│   ├── deploy-cloudfront.ps1           # CloudFront CDN deployment script
│   ├── cloudfront-s3.yaml              # CloudFormation template for frontend hosting
│   └── ec2-userdata.sh                 # EC2 automated host provisioning script
│
├── data/                               # Clinical Datasets & Local Database
│   ├── LUNA16/                         # Official LUNA16 benchmark dataset
│   │   ├── subset0/                    # Real 3D thoracic CT scans (.mhd / .raw format)
│   │   ├── annotations.csv             # Radiologist-annotated nodule coordinates & diameters
│   │   └── candidates_V2.csv           # Full candidate coordinates and class labels
│   ├── luna16_crops/                   # Cached 64³ isotropic nodule volumes (.npy)
│   └── lung_cancer_ai.db               # SQLite database (cases, studies, predictions)
│
├── checkpoints_3d/                     # Volumetric Weights (Experimental Smoke Test)
│   ├── resnet3d_luna16_best.pt         # 3D ResNet-50 weights (1-epoch, 10-scan smoke test)
│   ├── resnet3d_luna16_final.pt        # Final epoch checkpoint with calibration metadata
│   ├── resnet3d_luna16_results.json    # Quantitative training and validation history
│   └── resnet3d_luna16_robustness.json # Physical scanner perturbation stress-test outputs
│
├── experiments/                        # Scientific Benchmark Experiments
│   └── run_ieee_experiments.py         # IEEE benchmark evaluation runner (deterministic seed 42)
│
├── experiments_log/                    # Experiment Manifests & Run Tracking
│   └── run_*_resnet3d.json             # Structured JSON logs of training and dry runs
│
├── results/                            # Experimental Findings (Pipeline verified end-to-end)
│   ├── table1_discrimination_and_calibration.csv # Traced test AUC, ECE, Brier score, T
│   ├── table2_perturbation_robustness.csv         # Stability across 5 physical CT perturbation functions (6 conditions)
│   ├── table3_xai_localization.csv               # Quantitative 3D Grad-CAM Pointing Game scores
│   ├── table4_uncertainty_stratification.csv     # Epistemic uncertainty tiers (Low/Moderate/High)
│   └── exp_01_metrics.json                       # Machine-readable experimental summary
│
├── paper/                              # IEEE Conference & Journal Manuscript
│   ├── main.tex                        # Full IEEE Access / JBHI LaTeX manuscript
│   ├── references.bib                  # BibTeX bibliography (pending audit)
│   ├── IEEEtran.cls                    # Official IEEE documentclass
│   └── README.md                       # Traceability matrix mapping paper numbers to results/
│
├── lung_cancer_segmentation/           # Standalone 2D/3D slice segmentation module (benchmark)
├── lung-cancer-ai/                     # Monorepo full AI variant (reference implementation)
├── EXECUTION_RESULTS/                  # Pre-deployment validation reports (master_summary.json)
├── logs/                               # Application runtime logs (rotating JSON files)
├── docker-compose.yml                  # Full-stack orchestrator: backend + frontend + Prometheus + Grafana
├── pyproject.toml                      # Unified Python build tooling, ruff, mypy, pytest config
├── Makefile                            # Developer commands (dev, test, lint, docker-up)
├── models_registry.json                # Model lifecycle registry (STAGING / PRODUCTION metadata)
└── README.md                           # Master project documentation
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
| Grafana    | http://localhost:3001  (configured via GF_SECURITY_ADMIN_PASSWORD) |

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
8. **Findings & Structured Reporting** — Nodule candidate measurements and radiologist review workflow

---

## Pipeline Verification Status

Pipeline verified end-to-end; model validation in progress.

> [!NOTE]
> All unit and integration tests verify code execution, input/output tensor shape compliance, and interface contracts. Model weights are experimental benchmark checkpoints awaiting full cross-validation before clinical promotion. Not for clinical decisions.
