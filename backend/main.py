"""
backend/main.py
================
Unified entry point for the Lung Cancer AI Full-Stack Platform Backend.
Exposes the FastAPI application with:
- Clinical DICOM QA validation & intake
- 3D CT Preprocessing & Hounsfield Unit windowing
- 3D Nodule Detection (RetinaNet3D) & Localization
- 3D Deep Learning Classification (ResNet3D, DenseNet3D, Swin Transformer 3D)
- Multimodal Ensemble (DL + Radiomics + Clinical)
- Explainable AI (3D Grad-CAM, Integrated Gradients, Attention Rollout)
- Uncertainty Estimation (MC Dropout) & Temperature Scaling Calibration
- Segmentation Engine & Morphology Metrics
- MLflow Experiment Lineage & Model Registry
"""

import sys
import pathlib
import uvicorn

# Ensure repository root is on sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment variables
try:
    import dotenv
    dotenv.load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

from backend.api.service import app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lung Cancer AI Platform Backend")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    args, _ = parser.parse_known_args()

    print("=" * 70)
    print(f"  Starting Lung Cancer AI Platform Backend Service on port {args.port}...")
    print(f"  Swagger UI: http://localhost:{args.port}/docs")
    print(f"  Health Check: http://localhost:{args.port}/api/v2/health")
    print("=" * 70)

    if args.reload:
        uvicorn.run(
            "backend.main:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(ROOT_DIR / "backend")]
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)

