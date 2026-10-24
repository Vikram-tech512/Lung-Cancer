# Platform Results & Artifacts

This directory stores all generated outputs from the 11-stage ML pipeline:

- `xai/`: 3D Volumetric Explainable AI heatmaps (Grad-CAM 3D, Integrated Gradients, Swin3D Attention Rollout)
- `reports/`: Clinical decision reports conforming to Lung-RADS v2022 & Fleischner Society guidelines (JSON / PDF)
- `radiomics/` & `radiomics_cache.db`: SQLite cache of IBSI-compliant quantitative imaging features
- `master_summary.json`: Full validation & benchmarking summary report
