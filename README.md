# Lung Cancer AI Research Platform

A comprehensive, clinical-grade full-stack artificial intelligence platform for volumetric thoracic Computed Tomography (CT) analysis. Built with 3D Deep Learning architectures (UNet3D, nnUNet, RetinaNet3D, ResNet3D, DenseNet3D, Swin Transformer 3D), IBSI-compliant Radiomics, Multimodal Ensemble learning, Explainable AI (3D Grad-CAM, Integrated Gradients, Attention Rollout), and Monte Carlo Dropout uncertainty estimation.

---

## 📁 Repository Structure

```
Lung Cancer/
├── backend/                               # Full-Stack Backend Service (FastAPI v2.0)
│   ├── main.py                            # Unified backend server entry point
│   ├── Dockerfile                         # Production Docker container definition
│   ├── requirements.txt                   # Complete Python dependencies
│   ├── api/                               # REST API routers & clinical endpoints
│   │   └── service.py                     # Intake, 3D inference, XAI, streaming
│   ├── models/                            # 3D Neural Network Architectures
│   │   ├── classification/                # ResNet3D-50, DenseNet3D-121, Swin3D
│   │   ├── segmentation/                  # 3D U-Net, nnU-Net wrapper
│   │   ├── detection/                     # 3D RetinaNet nodule detector
│   │   ├── radiomics/                     # PyRadiomics, XGBoost, Random Forest
│   │   └── ensemble/                      # Multimodal Ensemble Engine
│   ├── dicom/                             # Volumetric DICOM Pipeline
│   │   ├── loader.py                      # HU conversion, isotropic resampling
│   │   ├── validator.py                   # 10-point clinical QA audit
│   │   └── deidentifier.py                # HIPAA safe-harbor de-identification
│   ├── preprocessing/                     # Thoracic CT windowing & normalization
│   ├── explainability/                    # 3D Grad-CAM, IG, Attention Rollout
│   ├── uncertainty/                       # MC Dropout & predictive entropy
│   ├── calibration/                       # Platt & Temperature Scaling
│   ├── robustness/                        # Stress testing & CT perturbations
│   ├── segmentation/                      # Segmentation inference & morphology
│   ├── datasets/                          # LIDC-IDRI loader & patient-stratified splits
│   ├── experiments/                       # MLflow tracking & model registry
│   ├── database/                          # SQLite schema, case logs, models_db
│   └── tests/                             # Pytest model smoke tests & validation
│
├── frontend/                              # Medical Glassmorphism Web Application
│   ├── src/
│   │   ├── App.tsx                        # Main dashboard & navigation shell
│   │   ├── index.css                      # Modern dark-mode clinical UI styling
│   │   ├── types.ts                       # TypeScript interfaces
│   │   └── panels/                        # 9 Interactive Clinical Panels
│   │       ├── IntakePanel.tsx            # DICOM intake & 10-point QA check
│   │       ├── CaseLogPanel.tsx           # Patient cohort case audit log
│   │       ├── CTViewerPanel.tsx          # 3D Multi-Planar Reconstruction (MPR)
│   │       ├── VerdictPanel.tsx           # Ensemble verdict & uncertainty bands
│   │       ├── MorphometryPanel.tsx       # 3D shape, sphericity & IBSI radiomics
│   │       ├── BenchmarkPanel.tsx         # Cross-model performance matrix
│   │       ├── TrainingPanel.tsx          # Live training loss & AUC curves
│   │       ├── ExperimentsPanel.tsx       # MLflow run lineage & metrics
│   │       └── RegistryPanel.tsx          # Model version cards & deployment
│   ├── Dockerfile                         # Multi-stage production Nginx container
│   ├── nginx.conf                         # Reverse proxy config for API routing
│   ├── package.json                       # Dependencies (React 18, Vite, Lucide)
│   └── vite.config.ts                     # Vite build configuration
│
├── lung-cancer-xai-phase1/                # Phase 1: Dataset Engineering & EDA
├── lung-cancer-xai-phase2/                # Phase 2: Model Architectures (CNN/ViT/Hybrid)
├── lung-cancer-xai-phase3/                # Phase 3: Evaluation, Calibration & Curves
├── lung-cancer-xai-phase4/                # Phase 4: Robustness Engine & Dashboard
├── lung_cancer_segmentation/              # Standalone segmentation benchmark module
├── docker-compose.yml                     # Multi-container orchestration (Backend + Frontend)
└── run_all.py                             # Master execution runner
```

---

## 🚀 Quick Start

### 1. Running with Docker Compose (Recommended)
```bash
docker-compose up --build
```
- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health Check**: [http://localhost:8000/api/v2/health](http://localhost:8000/api/v2/health)

---

### 2. Running Locally

#### Backend (Python 3.11+)
```bash
# Navigate to backend and install requirements
pip install -r backend/requirements.txt

# Start backend server
python -m backend.main
```

#### Frontend (Node 18+)
```bash
# Navigate to frontend and install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

---

## 🧪 Testing

Run model smoke tests (validates forward-pass tensor dimensions across all 3D neural architectures):
```bash
python -m pytest backend/tests/test_models.py -v
```
All 5 model smoke tests:
- `UNet3D` (Volumetric Segmentation) -> **PASS**
- `ResNet3D-50` (Volumetric Classification) -> **PASS**
- `DenseNet3D-121` (Dense Feature Re-use) -> **PASS**
- `SwinTransformer3D` (Shifted-Window Attention) -> **PASS**
- `MultimodalEnsemble` (DL + Radiomics + Clinical) -> **PASS**

---

## 🔬 Core Capabilities
1. **Intake & QA Gate**: Automated 10-point check (slice thickness, gantry tilt, HU range, spatial resolution).
2. **3D MPR Viewer**: Simultaneous Axial, Coronal, and Sagittal viewports with synchronous crosshairs and 3D Grad-CAM heatmap overlays.
3. **Multimodal Ensemble**: Combines 3D Deep Learning representations with IBSI PyRadiomics features and clinical metadata.
4. **Epistemic Uncertainty**: Monte Carlo Dropout estimation with predictive variance and temperature-scaled calibration.
5. **Robustness Benchmark**: Automated stress-testing against photon noise, motion artifacts, and slice decimation.
