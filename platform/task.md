# Lung Cancer AI Platform — Build Tasks

## Phase 1 — 3D Model Implementations
- [x] Scaffold `platform/` directory structure
- [x] 3D U-Net (`unet3d/model.py`)
- [x] nnU-Net wrapper (`nnunet/wrapper.py`)
- [x] 3D RetinaNet detection (`detector3d/retinanet3d.py`)
- [x] 3D ResNet-50 classifier (`resnet3d/model.py`)
- [x] 3D DenseNet-121 classifier (`densenet3d/model.py`)
- [x] 3D Swin Transformer (`swin3d/model.py`)

## Phase 2 — DICOM & Preprocessing
- [x] DICOM validator (`dicom/validator.py`)
- [x] DICOM loader + HU + resampling (`dicom/loader.py`)
- [x] CT preprocessing pipeline (`preprocessing/ct_pipeline.py`)

## Phase 3 — Radiomics + ML
- [x] PyRadiomics extractor (`radiomics/extractor.py`)
- [x] XGBoost classifier (`radiomics/xgboost_classifier.py`)
- [x] Random Forest (`radiomics/random_forest.py`)
- [x] Ensemble (`ensemble/ensemble.py`)

## Phase 4 — XAI Extensions
- [x] Grad-CAM 3D (`explainability/gradcam3d.py`)
- [x] Integrated Gradients (`explainability/integrated_gradients.py`)
- [x] Attention Rollout 3D (`explainability/attention_viz3d.py`)

## Phase 5 — Uncertainty & Calibration
- [x] MC Dropout (`uncertainty/mc_dropout.py`)
- [x] Temperature Scaling (`calibration/temperature_scaling.py`)

## Phase 6 — Robustness CT Extensions
- [x] CT perturbations (`robustness/ct_perturbations.py`)

## Phase 7 — Experiment Tracking
- [x] MLflow tracker (`experiments/mlflow_tracker.py`)
- [x] Model registry (`experiments/model_registry.py`)
- [x] Patient-level splits (`datasets/patient_split.py`)
- [x] LIDC-IDRI loader (`datasets/lidc_idri.py`)

## Phase 8 — FastAPI v2 Endpoints
- [x] New API service (`api/service.py`)
- [x] Model smoke test suite (`tests/test_models.py`)

## Phase 9 — Frontend (Vite + React + TypeScript + Medical Glassmorphism)
- [x] Scaffold frontend (`package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`)
- [x] CT Viewer with MPR and XAI overlays (`panels/CTViewerPanel.tsx`)
- [x] Intake panel with 10-point QC audit (`panels/IntakePanel.tsx`)
- [x] Case log panel (`panels/CaseLogPanel.tsx`)
- [x] Verdict panel with ensemble breakdown & uncertainty (`panels/VerdictPanel.tsx`)
- [x] Morphometry panel with 3D shape & IBSI radiomics (`panels/MorphometryPanel.tsx`)
- [x] Benchmark panel with patient-stratified comparison (`panels/BenchmarkPanel.tsx`)
- [x] Training panel with live loss logs (`panels/TrainingPanel.tsx`)
- [x] Experiments panel with MLflow lineage (`panels/ExperimentsPanel.tsx`)
- [x] Registry panel with deployment version cards (`panels/RegistryPanel.tsx`)

## Phase 10 — Docker & Config
- [x] platform docker-compose.yml
- [x] backend requirements.txt
- [x] backend Dockerfile
