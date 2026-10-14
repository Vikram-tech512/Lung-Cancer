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
import os
import pathlib
import types
import uvicorn

CURR_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = CURR_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(CURR_DIR) not in sys.path:
    sys.path.insert(0, str(CURR_DIR))

# Register 'backend' package alias if running directly inside the backend directory on AppSail
if "backend" not in sys.modules:
    backend_pkg = types.ModuleType("backend")
    backend_pkg.__path__ = [str(CURR_DIR)]
    backend_pkg.__file__ = str(CURR_DIR / "__init__.py")
    sys.modules["backend"] = backend_pkg

# Load environment variables
try:
    import dotenv
    dotenv.load_dotenv(ROOT_DIR / ".env")
    dotenv.load_dotenv(CURR_DIR / ".env")
except ImportError:
    pass

from backend.api.service import app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lung Cancer AI Platform Backend")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    listen_port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", os.environ.get("PORT", 8000)))
    parser.add_argument("--port", type=int, default=listen_port, help="Port to bind")
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
            reload_dirs=[str(ROOT_DIR / "backend"), str(CURR_DIR)]
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port)


