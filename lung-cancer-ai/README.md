# LUNG CANCER AI — MULTIMODAL CT ANALYSIS PLATFORM

> **Production-Ready Medical AI Platform for Volumetric Thoracic CT Segmentation, Detection, Classification, Radiomics & Multiplanar Explainable AI.**

---

## 1. System Architecture

```
                          INTERNET
                             │
                      HTTPS / WSS (443)
                             │
                   CUSTOM DOMAIN (DNS A/CNAME)
                             │
                 NGINX Reverse Proxy (SSL / Certbot)
                 ┌───────────┴───────────┐
                 │                       │
        HTTP / Static Assets       REST / WebSocket
                 │                       │
                 ▼                       ▼
          React + Vite + TS        FastAPI Application
         (Tailwind / shadcn)             │
                                         ├─► PostgreSQL (20 tables)
                                         ├─► Object Storage (MinIO/S3)
                                         ├─► MLflow Tracking Server
                                         │
                                   Redis Task Broker
                                         │
                                   Celery Workers
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 3D DL & Transformers           Radiomics & ML
                 - nnU-Net / 3D U-Net           - PyRadiomics (107 feats)
                 - 3D RetinaNet Detection       - XGBoost + SHAP
                 - 3D ResNet-50 Classifier      - Random Forest
                 - 3D Swin Transformer          - Multimodal Soft Ensemble
                 - 3D DenseNet-121              - Temperature Scaling
                 - Hybrid CNN-ViT               - MC Dropout (T=30)
                         │
                         ▼
                Real-Time Event Stream
             (16 WebSocket Task States)
                         │
                         ▼
               Interactive Workstation UI
         (Exact Match to Medical Reference)
```

---

## 2. Key Capabilities & Architectural Principles

1. **Real Client-Server Architecture**:
   - Built on a real decoupled frontend (React + TypeScript + Vite) and backend (FastAPI + PostgreSQL + Celery + Redis).
   - Zero frontend timer simulations: task states (`QUEUED`, `VALIDATING`, `PREPROCESSING`, `SEGMENTING_LUNGS`, `DETECTING_NODULES`, `CLASSIFYING`, `RUNNING_TRANSFORMER`, `RUNNING_RADIOMICS`, `CALIBRATING`, `GENERATING_EXPLANATION`, `GENERATING_REPORT`, `COMPLETED`) stream live over WebSockets.
2. **24/7 Cloud Operations**:
   - Runs independently of any developer computer or IDE.
   - Configured with `restart: always` Docker daemon policies, systemd units, and NGINX reverse proxy.
3. **Configurable Brand & Custom Domain**:
   - `APP_NAME` and `PUBLIC_DOMAIN` are configurable via `/settings` and environment variables.
   - Built-in automatic SSL certification with Let's Encrypt.
4. **Concrete 3D Deep Learning & Transformer Models**:
   - **3D ResNet-50**: Volumetric bottleneck blocks with Monte Carlo Dropout ($T=30$ forward passes) for epistemic uncertainty.
   - **3D Swin Transformer**: Shifted-window volumetric self-attention partitioning $64^3$ nodule crops into $4\times 4\times 4$ voxel tokens ($96$-dim embeddings).
   - **3D U-Net & nnU-Net v2**: Full-resolution parenchyma and nodule segmentation with DiceCELoss.
   - **3D RetinaNet**: Volumetric candidate detection with 3D spherical anchors ($5\text{--}25\text{mm}$).
   - **3D DenseNet-121**: 4 dense blocks ($k=32$) with transition compression.
   - **PyRadiomics**: 107 standardized IBSI features spanning GLCM, GLRLM, GLSZM, GLDM, first-order, and 3D shape descriptors with XGBoost and Random Forest.
   - **Multimodal Soft Ensemble**: Learned reliability-calibrated weighting with transparent disagreement entropy.
5. **Multiplanar Explainable AI (XAI)**:
   - 3D Grad-CAM, 3D Grad-CAM++, Integrated Gradients, and Swin Attention Rollout with mandatory regulatory notices (*"MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE"*).
6. **Clinical Workstation UI**:
   - Built to match the clinical workstation layout:
     - **01 · INTAKE**: DICOM drag-and-drop and 10-point QC audit report.
     - **02 · CASE LOG**: Session history and cohort archive.
     - **03 · MULTIMODAL VIEWER**: Multiplanar slice viewer with layer toggles (Segmentation, Attention, Grad-CAM, Saliency), opacity slider, and pink lesion contour.
     - **05 · VERDICT**: Consensus probability (`84.9% MODERATE`), Benign/Malignant percentage bars, and dynamic EKG confidence signature waveform.
     - **06 · MORPHOMETRY**: Area ($13125$), Volume ($\text{mm}^3$), Centroid ($140.5, 104.4$), Compactness ($0.44$), Lobularity ($0.40$), and latency ($4768\text{ ms}$).

---

## 3. Quick Start (Local Development)

### Backend & Database:
```bash
cd lung-cancer-ai
python scripts/init_db.py
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend:
```bash
cd lung-cancer-ai/frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 4. Production Cloud Deployment (24/7)

Deploy on any Ubuntu cloud VPS (AWS, DigitalOcean, Hetzner, GCP, Azure):
```bash
cd /opt/lung-cancer-ai/deployment/docker
docker compose -f docker-compose.prod.yml up -d
```
See [DEPLOYMENT.md](file:///c:/Users/ashok/OneDrive/Desktop/Lung%20Cancer/lung-cancer-ai/DEPLOYMENT.md) for domain DNS configuration, Let's Encrypt SSL setup, and systemd 24/7 auto-boot.
