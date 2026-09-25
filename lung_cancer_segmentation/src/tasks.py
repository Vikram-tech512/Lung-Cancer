# src/tasks.py
"""Celery task definitions for async segmentation and training."""
import base64
import io
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.celery_app import celery
from src.config import MODEL_CHECKPOINT, THRESHOLD, UPLOAD_DIR
from src.database import SessionLocal
from src.models_db import SegmentationRequest, TaskStatus


def _load_model():
    """Load the ViTSeg model (lazy, cached in worker memory)."""
    from src.model.vit_seg import ViTSeg
    model = ViTSeg()
    ckpt = Path(MODEL_CHECKPOINT)
    if ckpt.exists():
        state = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


# Module‑level cache so the model is loaded once per worker
_model = None


def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@celery.task(bind=True, name="segment_image", max_retries=2)
def segment_image(self, request_id: str, image_path: str):
    """Run segmentation on a single image and store results in the DB."""
    db = SessionLocal()
    try:
        req = db.query(SegmentationRequest).filter(SegmentationRequest.id == request_id).first()
        if not req:
            return {"error": "Request not found"}

        req.status = TaskStatus.PROCESSING
        db.commit()

        model = get_model()

        img = Image.open(image_path).convert("RGB")
        x = _preprocess(img).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = torch.sigmoid(model(x))
            mask = (pred > THRESHOLD).float()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Encode mask to base64 PNG
        mask_pil = transforms.ToPILImage()(mask.squeeze(0).cpu())
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()

        req.status = TaskStatus.COMPLETED
        req.mask_base64 = encoded
        req.inference_time_ms = round(elapsed_ms, 2)
        req.completed_at = datetime.now(timezone.utc)
        db.commit()

        return {"request_id": request_id, "status": "completed", "inference_ms": elapsed_ms}

    except Exception as exc:
        db.rollback()
        req = db.query(SegmentationRequest).filter(SegmentationRequest.id == request_id).first()
        if req:
            req.status = TaskStatus.FAILED
            db.commit()
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()
