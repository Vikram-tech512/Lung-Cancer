# src/inference/morphology.py
"""Quantitative Pulmonary Nodule Morphology & Lung-RADS Assessment Engine.
Extracts dynamic, image-derived anatomical measurements directly from the
CT pixel array and neural activation/segmentation contours:
  - Max Feret Diameter (mm)
  - Ellipsoidal Volume (mm³)
  - Mean Hounsfield Units (HU)
  - Spiculation Boundary Roughness Index (0.0–1.0)
  - Ground-Glass Opacity (GGO) Attenuation Ratio (%)
  - Sphericity / Compactness Index (0.0–1.0)
  - Anatomical Lobe Localization (RUL, RML, RLL, LUL, Lingula, LLL)
  - Algorithmic ACR Lung-RADS v2022 Staging Criteria
"""
import io
import base64
import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image


def analyze_nodule_morphology(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str,
    pixel_spacing_mm: float = 0.703,
    hu_array: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate authentic morphological nodule metrics derived dynamically
    from the CT image and neural activation heatmap.
    """
    h, w = heatmap.shape[:2]

    # Case 1: Normal parenchymal scan (No focal lesion)
    if predicted_class.lower() == "normal":
        # Compute background parenchymal HU
        if hu_array is not None:
            mean_hu = int(round(float(np.mean(hu_array))))
        else:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if len(image_rgb.shape) == 3 else image_rgb
            mean_hu = int(round(float(np.mean(gray) / 255.0 * 2000.0 - 1000.0)))

        return {
            "nodule_detected": False,
            "measurements": {
                "max_diameter_mm": None,
                "volume_mm3": None,
                "mean_hu": mean_hu,
                "spiculation_index": 0.0,
                "ggo_ratio_pct": 0,
                "compactness_index": 0.0,
                "notes": "No focal pulmonary nodule identified on current axial slice."
            },
            "lung_rads": {
                "available": True,
                "category": "Lung-RADS 1",
                "code": "LR-1",
                "stage_idx": 0,
                "assessment": "Negative",
                "clinical_criteria": "No actionable pulmonary nodule or consolidation identified.",
                "recommendation": "Routine annual low-dose screening examination (LDCT)."
            },
            "lobes": {
                "RUL": "Clear", "RML": "Clear", "RLL": "Clear",
                "LUL": "Clear", "Lingula": "Clear", "LLL": "Clear"
            },
            "segmentation": {
                "contour_points": [],
                "mask_overlay_base64": None
            }
        }

    # Case 2: Lesion detected (Malignant or Benign)
    # Threshold the model activation map to isolate the primary focus
    norm_hm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    thresh_val = max(0.48, float(np.percentile(norm_hm, 90)))
    binary_mask = (norm_hm >= thresh_val).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or max(cv2.contourArea(c) for c in contours) < 12:
        # Activation was diffuse or sub-threshold
        return {
            "nodule_detected": False,
            "measurements": {
                "max_diameter_mm": None,
                "volume_mm3": None,
                "mean_hu": -600,
                "spiculation_index": 0.0,
                "ggo_ratio_pct": 0,
                "compactness_index": 0.0,
                "notes": "Diffuse activation; no discrete solid nodule boundary resolvable."
            },
            "lung_rads": {
                "available": True,
                "category": "Lung-RADS 2",
                "code": "LR-2",
                "stage_idx": 1,
                "assessment": "Benign Appearance",
                "clinical_criteria": "Sub-threshold opacity or diffuse parenchymal change.",
                "recommendation": "Follow-up at 12-month routine screening."
            },
            "lobes": {
                "RUL": "Clear", "RML": "Clear", "RLL": "Clear",
                "LUL": "Clear", "Lingula": "Clear", "LLL": "Clear"
            },
            "segmentation": {
                "contour_points": [],
                "mask_overlay_base64": None
            }
        }

    # Extract primary contour (largest lesion component)
    contour = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(contour)
    perimeter_px = max(1.0, cv2.arcLength(contour, True))

    # 1. Max Feret Diameter (Calculated from contour points)
    pts = contour.reshape(-1, 2)
    if len(pts) > 1:
        # Euclidean distance matrix approximation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = (diff ** 2).sum(axis=-1)
        max_dist_px = float(np.sqrt(dist_sq.max()))
    else:
        max_dist_px = 10.0

    max_diameter_mm = round(max_dist_px * pixel_spacing_mm, 1)

    # 2. Minor Diameter & Volume
    rect = cv2.minAreaRect(contour)
    min_dim_px = min(rect[1]) if min(rect[1]) > 0 else max_dist_px * 0.65
    minor_d_mm = max(1.0, float(min_dim_px * pixel_spacing_mm))

    # Ellipsoidal / Spheroid volume: 4/3 * pi * (d_max/2) * (d_min/2)^2
    vol_mm3 = int(round((4.0 / 3.0) * math.pi * (max_diameter_mm / 2.0) * ((minor_d_mm / 2.0) ** 2)))

    # 3. Mean HU within lesion
    mask_filled = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_filled, [contour], -1, 255, -1)

    if hu_array is not None and hu_array.shape[:2] == (h, w):
        nodule_hu = hu_array[mask_filled == 255]
        mean_hu = int(round(float(np.mean(nodule_hu))))
    else:
        # Standard CT window HU estimate from grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if len(image_rgb.shape) == 3 else image_rgb
        if gray.shape[:2] != (h, w):
            gray = cv2.resize(gray, (w, h))
        nodule_px = gray[mask_filled == 255]
        # Standard thoracic mapping: 0 -> -1000 HU, 255 -> +1000 HU
        mean_hu = int(round(float(np.mean(nodule_px) / 255.0 * 2000.0 - 1000.0))) if len(nodule_px) > 0 else -150

    # 4. Spiculation Index: Perimeter / Convex-Hull Perimeter irregularity
    hull = cv2.convexHull(contour)
    hull_perimeter = max(1.0, cv2.arcLength(hull, True))
    spic_raw = (perimeter_px - hull_perimeter) / hull_perimeter
    spiculation_index = min(1.0, max(0.0, round(float(spic_raw * 2.2), 2)))

    # 5. Compactness / Sphericity Index: 4 * pi * Area / Perimeter^2
    compactness_raw = (4.0 * math.pi * area_px) / (perimeter_px ** 2)
    compactness_index = min(1.0, max(0.0, round(float(compactness_raw), 2)))

    # 6. Ground-Glass Opacity (GGO) Ratio
    # Pixels between -750 and -400 HU are typically GGO
    if hu_array is not None:
        nodule_hu = hu_array[mask_filled == 255]
        ggo_count = np.sum((nodule_hu >= -750) & (nodule_hu <= -400))
        ggo_ratio_pct = int(round(float(ggo_count / len(nodule_hu) * 100))) if len(nodule_hu) > 0 else 0
    else:
        # Grayscale equivalent of -750 to -400 HU in lung window: ~30 to ~75 in 0-255
        nodule_px = gray[mask_filled == 255]
        ggo_count = np.sum((nodule_px >= 32) & (nodule_px <= 78))
        ggo_ratio_pct = int(round(float(ggo_count / len(nodule_px) * 100))) if len(nodule_px) > 0 else 0

    # 7. Centroid & Anatomical Lobe Localization
    M = cv2.moments(contour)
    cx_norm = (M["m10"] / (M["m00"] + 1e-8)) / w
    cy_norm = (M["m01"] / (M["m00"] + 1e-8)) / h

    lobes = {"RUL": "Clear", "RML": "Clear", "RLL": "Clear", "LUL": "Clear", "Lingula": "Clear", "LLL": "Clear"}
    primary_lobe = "RUL"
    # Radiological convention: Left side of image is Patient Right
    if cx_norm < 0.50:
        if cy_norm < 0.38:
            primary_lobe = "RUL"
        elif cy_norm < 0.64:
            primary_lobe = "RML"
        else:
            primary_lobe = "RLL"
    else:
        if cy_norm < 0.42:
            primary_lobe = "LUL"
        elif cy_norm < 0.62:
            primary_lobe = "Lingula"
        else:
            primary_lobe = "LLL"

    lobes[primary_lobe] = "Primary" if predicted_class.lower() == "malignant" else "Nodule"

    # 8. Dynamic ACR Lung-RADS v2022 Staging Criteria
    is_mal = predicted_class.lower() == "malignant"
    if not is_mal and max_diameter_mm < 6.0:
        rads_cat = "Lung-RADS 2"
        rads_code = "LR-2"
        stage_idx = 1
        assessment = "Benign Appearance"
        criteria = f"Solid benign-appearing nodule measuring {max_diameter_mm} mm (< 6 mm threshold)."
        recommendation = "Continue annual low-dose screening examination at 12 months."
    elif not is_mal and max_diameter_mm < 8.0:
        rads_cat = "Lung-RADS 3"
        rads_code = "LR-3"
        stage_idx = 2
        assessment = "Probably Benign"
        criteria = f"Indeterminate pulmonary nodule measuring {max_diameter_mm} mm (6–8 mm range)."
        recommendation = "Low-dose chest CT follow-up at 6 months to evaluate stability."
    elif max_diameter_mm < 15.0 and spiculation_index < 0.40:
        rads_cat = "Lung-RADS 4A"
        rads_code = "LR-4A"
        stage_idx = 3
        assessment = "Suspicious"
        criteria = f"Solid nodule measuring {max_diameter_mm} mm with intermediate morphometric risk."
        recommendation = "Low-dose CT at 3 months; PET-CT recommended if solid component >= 8 mm."
    else:
        rads_cat = "Lung-RADS 4B"
        rads_code = "LR-4B"
        stage_idx = 4
        assessment = "Very Suspicious"
        criteria = f"High-risk lesion measuring {max_diameter_mm} mm with irregular/spiculated borders (spiculation index {spiculation_index:.2f})."
        recommendation = "Urgent multidisciplinary thoracic oncology consultation, PET-CT, and tissue biopsy."

    # 9. Create High-Quality Mask Overlay Image
    overlay_rgb = image_rgb.copy()
    if overlay_rgb.shape[:2] != (h, w):
        overlay_rgb = cv2.resize(overlay_rgb, (w, h))

    color = (255, 45, 107) if is_mal else (0, 230, 118)
    cv2.drawContours(overlay_rgb, [contour], -1, color, 2)

    # Crosshair measurement line
    center_pt = (int(cx_norm * w), int(cy_norm * h))
    half_d = int(max_dist_px / 2.0)
    cv2.line(overlay_rgb, (center_pt[0] - half_d, center_pt[1]), (center_pt[0] + half_d, center_pt[1]), color, 1, cv2.LINE_AA)
    cv2.line(overlay_rgb, (center_pt[0], center_pt[1] - half_d), (center_pt[0], center_pt[1] + half_d), color, 1, cv2.LINE_AA)

    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    mask_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return {
        "nodule_detected": True,
        "measurements": {
            "max_diameter_mm": max_diameter_mm,
            "volume_mm3": vol_mm3,
            "mean_hu": mean_hu,
            "spiculation_index": spiculation_index,
            "ggo_ratio_pct": ggo_ratio_pct,
            "compactness_index": compactness_index,
            "centroid": [round(cx_norm, 3), round(cy_norm, 3)],
            "notes": f"Measured across {len(pts)} contour coordinates using {pixel_spacing_mm} mm/pixel spacing."
        },
        "lung_rads": {
            "available": True,
            "category": rads_cat,
            "code": rads_code,
            "stage_idx": stage_idx,
            "assessment": assessment,
            "clinical_criteria": criteria,
            "recommendation": recommendation,
            "assessment_type": "Algorithmic ACR Lung-RADS 2022 approximation based on 2D axial geometry."
        },
        "lobes": lobes,
        "segmentation": {
            "contour_points": [[int(p[0]), int(p[1])] for p in pts[::max(1, len(pts) // 30)]],
            "mask_overlay_base64": mask_b64
        }
    }
