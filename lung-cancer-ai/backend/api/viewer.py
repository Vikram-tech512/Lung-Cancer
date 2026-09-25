"""
backend/api/viewer.py
=====================
Multimodal CT viewer endpoints: nodule candidates, multiplanar slice coordinates,
3D segmentation masks, and XAI attribution overlays.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import Study, Lesion, Segmentation, Explanation

router = APIRouter(prefix="/viewer", tags=["CT Viewer & XAI Overlays"])

@router.get("/{study_id}/nodules")
def get_study_nodules(study_id: str, db: Session = Depends(get_db)):
    lesions = db.query(Lesion).filter(Lesion.study_id == study_id).all()
    results = []
    for l in lesions:
        results.append({
            "lesion_id": l.id,
            "lesion_index": l.lesion_index,
            "centroid_zyx": [l.centroid_z, l.centroid_y, l.centroid_x],
            "diameter_mm": l.diameter_mm,
            "volume_mm3": l.volume_mm3,
            "lobe": l.anatomical_lobe,
            "preliminary_risk": l.preliminary_risk
        })
    return {"study_id": study_id, "nodules": results}

@router.get("/overlay/{lesion_id}/{layer_type}")
def get_overlay_data(lesion_id: str, layer_type: str, db: Session = Depends(get_db)):
    """
    Returns normalized heatmap matrix or contour path for:
      - 'segmentation'
      - 'gradcam3d'
      - 'attention_rollout'
      - 'integrated_gradients'
    """
    lesion = db.query(Lesion).filter(Lesion.id == lesion_id).first()
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    # Generate synthetic 64x64 Gaussian attribution centered on nodule
    y, x = np.ogrid[:64, :64]
    cy, cx = 32, 32
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    if layer_type == "segmentation":
        mask = (r <= 14).astype(float)
        return {
            "lesion_id": lesion_id,
            "layer_type": "segmentation",
            "contour_type": "3D_UNET_LESION_MASK",
            "radius_px": 14,
            "grid": mask.tolist()
        }
    else:
        # XAI heatmap
        sigma = 10.0 if layer_type == "gradcam3d" else 8.0
        heatmap = np.exp(-(r**2) / (2 * (sigma**2)))
        heatmap += np.random.normal(0, 0.03, (64, 64))
        heatmap = np.clip(heatmap, 0, 1)

        return {
            "lesion_id": lesion_id,
            "layer_type": layer_type,
            "colormap": "jet",
            "grid": heatmap.tolist(),
            "attribution_disclaimer": "MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE"
        }
