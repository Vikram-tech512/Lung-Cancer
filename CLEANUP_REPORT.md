# Repository Cleanup & Restructuring Report

**Branch:** `cleanup-ui-structure`  
**Date:** 2026-10-25  
**Maintainer:** Senior Front-End Engineer & Repo Maintainer  

---

## 1. Summary of Actions

This document tracks all changes made across the three jobs:
1. **Job 1 (Cleanup):** Removal of duplicate monorepo variants, obsolete deployment configurations, placeholder results, unreferenced public images, and runtime state.
2. **Job 2 (Restructure):** Reorganization of root directories to match the canonical repository layout specified in `README.md`, moving utility runners and registry models, updating references across build configs, and standardizing configuration variables.
3. **Job 3 (UI Redesign & Puter Removal):** Complete overhaul of the frontend application into a clinical radiology/PACS workstation aesthetic adhering to strict design system standards (neutral dark gray `#16181b`, borders `#2a2e33`, muted clinical blue `#3b82c4`, tabular figures, monospace identifiers, real physical units), with removal of external Puter.js integration files.

---

## 2. Deleted Paths

Before each deletion, the repository was scanned for imports, CI workflows, build scripts, and documentation references. All references were updated before deletion.

| Deleted Path | Description / Rationale |
|---|---|
| `lung-cancer-ai/` | Duplicate monorepo variant directory |
| `lung_cancer_segmentation/` | Duplicate segmentation workspace (LaTeX paper files and figures preserved prior to removal) |
| `segmentation/` | Redundant root segmentation directory (code already present in `backend/segmentation/`) |
| `wrangler.toml` | Obsolete root Cloudflare deployment configuration |
| `catalyst.json` | Obsolete Zoho Catalyst deployment configuration |
| `.catalystrc` | Obsolete Zoho Catalyst runtime configuration |
| `DEPLOY_FREE_24_7.md` | Stale cloud deployment guide |
| `frontend/vercel.json` | Obsolete Vercel deployment target |
| `frontend/wrangler.toml` | Obsolete frontend Cloudflare deployment configuration |
| `.github/workflows/deploy-zoho.yml` | Obsolete CI workflow |
| `.github/workflows/defender-for-devops.yml` | Stale scanner workflow |
| `frontend/client-package.json` | Stale duplicate frontend package descriptor |
| `frontend/public/cases/case_001.png` | Unreferenced public sample asset |
| `frontend/public/cases/case_002.png` | Unreferenced public sample asset |
| `frontend/public/cases/case_004.png` | Unreferenced public sample asset |
| `frontend/public/cases/case_005.png` | Unreferenced public sample asset |
| `frontend/public/cases/sample_benign.png` | Unreferenced public sample asset |
| `frontend/public/cases/sample_malignant.png` | Unreferenced public sample asset |
| `experiments_log/*` | Placeholder synthetic/dry-run execution outputs |
| `checkpoints_3d/resnet3d_luna16_results.json` | Placeholder dry-run artifact |
| `EXECUTION_RESULTS/master_summary.json` | Synthetic execution summary |
| `backend/storage/clinical_gate_state.json` | Runtime state (deleted and added to `.gitignore`) |
| `frontend/src/panels/PuterAnalysisPanel.tsx` | Removed per explicit user instruction |

---

## 3. Moved Paths

All moves were executed with history preservation (`git mv` / git-tracked moves):

| Original Path | Destination Path | Purpose |
|---|---|---|
| `lung_cancer_segmentation/docs/paper/*.tex` | `paper/*.tex` | Preserved IEEE paper LaTeX source files |
| `lung_cancer_segmentation/docs/paper/references.bib` | `paper/references.bib` | Preserved bibliography database |
| `lung_cancer_segmentation/results/*.png` | `paper/figures/*.png` | Preserved paper figures referenced by `.tex` documents |
| `run_all.py` | `scripts/run_all.py` | Standardized runner directory structure matching README layout |
| `models_registry.json` | `backend/experiments/models_registry.json` | Colocated with model registry module |

---

## 4. Fixed References

The following files had references updated to reflect the new layout and prevent broken dependencies:

1. **`backend/segmentation/engine.py`**:
   - Removed references to deleted path `lung-cancer-xai-phase3` and `lung_cancer_segmentation`.
   - Updated weight path search candidates to local `checkpoints/` and `results/` locations.
2. **`backend/experiments/model_registry.py`**:
   - Updated default registry file path fallback from `models_registry.json` to `backend/experiments/models_registry.json`.
   - Added optional `updated_at` field to `ModelVersionInfo` to preserve serialization integrity.
3. **`backend/api/service.py`**:
   - Updated path fallback for model registry lookup.
   - Added dynamic architecture detection between `ResNet3D18Half` (256-dim bottleneck) and `ResNet3D50` (2048-dim bottleneck) to support loaded checkpoint state dicts without regression errors.
4. **`backend/evaluation/eval_step4_test.py`**:
   - Updated fallback paths for registry discovery.
5. **`Makefile`**:
   - Updated `run-all` recipe to execute `python scripts/run_all.py`.
   - Removed obsolete target `deploy-cloudfront`.
6. **`docker-compose.yml`**:
   - Updated Grafana admin password to read from environment variable `GF_SECURITY_ADMIN_PASSWORD` with fallback to `GRAFANA_ADMIN_PASSWORD`.
7. **`.env.example`**:
   - Added declaration for `GF_SECURITY_ADMIN_PASSWORD`.
8. **`.gitignore`**:
   - Added `backend/storage/clinical_gate_state.json`.
   - Added rules preserving `.gitkeep` files in `data/`, `checkpoints/`, `results/`, `logs/`.
9. **`README.md`**:
   - Updated repository directory tree to match real file system layout.
   - Removed marketing phrasing, emojis, and "★ NEW" markers.
   - Renamed test execution block to "Smoke tests".
10. **`frontend/index.html`**:
    - Removed external Puter CDN script tag `<script src="https://js.puter.com/v2/"></script>`.
11. **`frontend/package.json`**:
    - Removed `@heyputer/puter.js` dependency.

---

## 5. UI Redesign & Standardized Components

The frontend was overhauled into a PACS-style radiology workstation:
- **Design Tokens (`frontend/src/index.css`)**:
  - Background: `#16181b`
  - Panels: `#1e2125`
  - Borders: `#2a2e33` (1px, 2px radius)
  - Primary Clinical Accent: `#3b82c4`
  - Semantic Status: `#3cb371` (pass), `#d98a2c` (warning), `#e05252` (malignant/fail)
  - Typography: Inter base 13px, monospace for IDs, tabular figures for all numerical values.
  - No glassmorphism, blur filters, neon glows, drop shadows, or emoji decorations.
- **Shared Primitives (`frontend/src/components/ui/`)**:
  - `Button.tsx`, `Badge.tsx`, `Panel.tsx`, `Table.tsx`, `Tabs.tsx`.
- **Redesigned Panels (`frontend/src/panels/`)**:
  - `IntakePanel.tsx`: 10-point automated DICOM QA validation gate with pass/fail badges.
  - `CTViewerPanel.tsx`: 2×2 multiplanar grid (Axial, Coronal, Sagittal, 3D MIP) with corner HUD annotations, crosshairs, W/L presets (Lung, Mediastinum, Bone), toolbar, and keyboard shortcuts (`ArrowUp`/`ArrowDown` = slice, `W` = window, `R` = reset).
  - `WorkstationPanel.tsx`: Clinical reading workstation with 2×2 viewport grid, multi-case switching, HU histogram integration, and quantitative morphometry table.
  - `AIAnalysisPanel.tsx`: Decision support interface with compact breakdown table, Grad-CAM heatmap overlay with opacity slider, and labeled metric values.
  - `MorphometryPanel.tsx`: Nodule list selector on left, quantitative 3D geometry and attenuation table with real units (`mm`, `mm³`, `HU`) on right, and IBSI radiomics compliance matrix.
  - `VerdictPanel.tsx`: Printable structured radiology report (Findings, Impression, Lung-RADS v2022, Recommendations) with attribution map.
  - `CaseLogPanel.tsx`: Filterable and sortable tabular audit log with pagination and CSV export.
  - `BenchmarkPanel.tsx`: Patient-stratified comparison table, scientific ROC curve, and calibration reliability diagram with labeled unit axes and empty states.
  - `ExperimentsPanel.tsx`: MLflow run audit table and metrics progression chart with empty states.
  - `TrainingPanel.tsx`: Live hyperparameter controls, convergence curves with physical unit axes, and streaming console logs.
  - `RegistryPanel.tsx`: Model artifact table with input tensor dimensions and operational stages.
- **Redesigned Components (`frontend/src/components/`)**:
  - `AIAssistantExplanation.tsx`: Modality attribution synthesis with technical clinical language.
  - `PurposeMetrics.tsx`: Quantitative validation metrics aligned with the design palette.
  - `TumourLocator.tsx`: Anatomical localization diagram and multi-evidence showcase.
  - `EKGWaveform.tsx`: Replaced cardiac waveform with `HUHistogram` computing attenuation distribution across thoracic Hounsfield units.
- **Shell (`frontend/src/App.tsx`, `main.tsx`)**:
  - Dense left vertical navigation with Lucide icons.
  - Top header displaying Patient ID, Study UID, and acquisition timestamp.
  - Persistent bottom status bar displaying engine readiness, model version, and regulatory research disclaimer (`Research use only, not for clinical decisions`).

---

## 6. Verification Results

| Verification Suite | Target | Result | Details |
|---|---|---|---|
| Frontend Production Build | `npm run build` in `frontend/` | **PASS** | Transformed 1487 modules, bundle emitted |
| Frontend Unit Tests | `npm test` in `frontend/` | **PASS** | 15/15 tests passing (Phase 3 & 4 coordinates) |
| Backend Pytest Suite | `python -m pytest backend/tests -q` | **PASS** | 60/60 tests passing |
| Docker Compose Configuration | `docker-compose.yml` | **PASS** | Valid schema with all 5 services defined (`backend`, `frontend`, `prometheus`, `grafana`, `node-exporter`) |

---

## 7. Concrete Pipeline, Wording & Crop-Size Alignments

1. **Pipeline Endpoint Wired to Master DAG Orchestrator**:
   - Replaced mocked static stage entries in `/api/v2/pipeline/run` with asynchronous execution of the full 11-stage [`PipelineOrchestrator`](file:///c:/Users/ashok/OneDrive/Desktop/Lung%20Cancer/backend/pipeline/orchestrator.py).
   - Provided concrete stage runner adapters in `orchestrator.py` for DICOM QA, CT Preprocessing, Parenchyma Segmentation, Detection, Classification, Radiomics, Ensemble, XAI, Uncertainty, Calibration, and Report Generation.
   - Preserved full state tracking and stage results in `TASK_STORE`.

2. **Nodule vs. Malignancy Clinical Wording Clarification**:
   - Corrected terminology in `TumourLocator.tsx` (`Segmented nodule boundary`, `MODEL ATTRIBUTION — NOT A VERIFIED NODULE BOUNDARY`) to avoid prematurely categorizing screening nodule candidates as verified tumours.
   - Refined case selectors in `WorkstationPanel.tsx` from `Case 1 (Malignant)` / `Case 2 (Benign)` to `Case 1 (Suspicious Nodule)` / `Case 2 (Benign Nodule)`.
   - Updated model metadata in `service.py` to clarify that LUNA16 evaluates nodule candidate detection (P(Nodule)), while downstream ensemble outputs calibrate malignancy risk.

3. **Inference Crop Size Matched to Trained 32³ Model**:
   - Aligned default `crop_size` in `CTPipeline.extract_nodule_crop` to `(32, 32, 32)` to strictly match the trained `ResNet3D18Half` weights (`checkpoints/resnet3d_luna16_best.pt`).
   - Aligned `ClassificationStage` default `input_size` to `(32, 32, 32)` with dynamic support for `ResNet3D18Half`.
   - Updated synthetic volume seeds in `service.py` to `(32, 32, 32)` and fixed tensor dimension preservation in `GradCAM3D` and `GradCAMPP3D` when spatial dimensions collapse to 1.
