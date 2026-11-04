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

    # In 2D axial slices, determining 3D lobe (RUL/RML/RLL) without volumetric Z series
    # is anatomically unverifiable. Only laterality (hemithorax) is observable.
    is_patient_right = cx_norm < 0.50
    hemi = "Right Hemithorax" if is_patient_right else "Left Hemithorax"
    primary_lobe = f"{hemi} (Lobe localization unavailable on 2D slice)"
    lobes = {
        "RUL": "Unknown (2D slice)",
        "RML": "Unknown (2D slice)",
        "RLL": "Unknown (2D slice)",
        "LUL": "Unknown (2D slice)",
        "Lingula": "Unknown (2D slice)",
        "LLL": "Unknown (2D slice)",
        "laterality": hemi,
        "status": "Volumetric CT series required for anatomical lobe segmentation"
    }

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


"""
3D Region-Growing Nodule Segmentation Module.

Implements spacing-aware 3D region growing from physical seed coordinates (X, Y, Z in mm)
on volumetric CT scans (Hounsfield Units).
Computes equivalent spherical diameter and flags empty or oversized/leaked regions.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from collections import deque


def physical_to_voxel(
    coord_xyz_mm: Tuple[float, float, float],
    origin_xyz_mm: Tuple[float, float, float],
    spacing_zyx_mm: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Convert physical coordinates (X, Y, Z in mm) to voxel indices (Z, Y, X).
    spacing_zyx_mm is (spacing_z, spacing_y, spacing_x).
    """
    cx, cy, cz = coord_xyz_mm
    ox, oy, oz = origin_xyz_mm
    sz, sy, sx = spacing_zyx_mm

    vx = int(round((cx - ox) / sx))
    vy = int(round((cy - oy) / sy))
    vz = int(round((cz - oz) / sz))
    return vz, vy, vx


def region_grow_3d(
    volume_hu: np.ndarray,
    seed_zyx: Tuple[int, int, int],
    spacing_zyx_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_min: float = -600.0,
    hu_max: float = 200.0,
    intensity_tol_hu: float = 250.0,
    max_radius_mm: float = 20.0,
    max_diameter_mm: float = 30.0,
) -> Dict[str, Any]:
    """
    Perform 3D region growing from seed_zyx within volume_hu.

    Args:
        volume_hu: 3D numpy array of CT Hounsfield Units (Z, Y, X).
        seed_zyx: Voxel seed coordinate (z, y, x).
        spacing_zyx_mm: Voxel spacing in mm (sz, sy, sx).
        hu_min: Lower HU threshold for parenchymal / nodular tissue.
        hu_max: Upper HU threshold to exclude calcifications/bones.
        intensity_tol_hu: Maximum absolute HU deviation from seed voxel.
        max_radius_mm: Maximum physical Euclidean distance from seed to stop leakage.
        max_diameter_mm: Threshold above which a segmented region is flagged as oversized.

    Returns:
        Dict containing:
            - voxel_count: int
            - volume_mm3: float
            - equivalent_diameter_mm: float
            - is_empty: bool
            - is_oversized: bool
            - mask: 3D bool array of segmented region (cropped bbox or full)
    """
    sz, sy, sx = spacing_zyx_mm
    voxel_vol_mm3 = sz * sy * sx
    nz, ny, nx = volume_hu.shape
    vz, vy, vx = seed_zyx

    # Check bounds
    if vz < 0 or vz >= nz or vy < 0 or vy >= ny or vx < 0 or vx >= nx:
        return {
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "equivalent_diameter_mm": 0.0,
            "is_empty": True,
            "is_oversized": False,
            "reason": "Seed out of volume bounds",
        }

    seed_val = float(volume_hu[vz, vy, vx])

    # Check if seed itself is within plausible nodule density
    if seed_val < hu_min or seed_val > hu_max:
        return {
            "voxel_count": 0,
            "volume_mm3": 0.0,
            "equivalent_diameter_mm": 0.0,
            "is_empty": True,
            "is_oversized": False,
            "reason": f"Seed HU {seed_val:.1f} outside range [{hu_min}, {hu_max}]",
        }

    # Bounding box in voxels for efficiency
    max_rz = int(np.ceil(max_radius_mm / sz))
    max_ry = int(np.ceil(max_radius_mm / sy))
    max_rx = int(np.ceil(max_radius_mm / sx))

    z_min = max(0, vz - max_rz)
    z_max = min(nz, vz + max_rz + 1)
    y_min = max(0, vy - max_ry)
    y_max = min(ny, vy + max_ry + 1)
    x_min = max(0, vx - max_rx)
    x_max = min(nx, vx + max_rx + 1)

    visited = np.zeros((z_max - z_min, y_max - y_min, x_max - x_min), dtype=bool)
    segmented = np.zeros_like(visited)

    queue = deque()
    # local coordinates
    lz = vz - z_min
    ly = vy - y_min
    lx = vx - x_min

    queue.append((lz, ly, lx))
    visited[lz, ly, lx] = True
    segmented[lz, ly, lx] = True

    # 6-connectivity 3D neighborhood
    neighbors = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    voxel_count = 0
    oversized = False

    while queue:
        cz, cy, cx = queue.popleft()
        voxel_count += 1

        # Check early termination if voxel count yields oversized region
        cur_vol = voxel_count * voxel_vol_mm3
        cur_diam = 2.0 * ((3.0 * cur_vol) / (4.0 * np.pi)) ** (1.0 / 3.0)
        if cur_diam > max_diameter_mm:
            oversized = True
            break

        gz = cz + z_min
        gy = cy + y_min
        gx = cx + x_min

        for dz, dy, dx in neighbors:
            nz_l = cz + dz
            ny_l = cy + dy
            nx_l = cx + dx

            if 0 <= nz_l < visited.shape[0] and 0 <= ny_l < visited.shape[1] and 0 <= nx_l < visited.shape[2]:
                if not visited[nz_l, ny_l, nx_l]:
                    visited[nz_l, ny_l, nx_l] = True

                    # Physical distance from seed
                    dist_mm = np.sqrt(
                        ((nz_l - lz) * sz) ** 2 +
                        ((ny_l - ly) * sy) ** 2 +
                        ((nx_l - lx) * sx) ** 2
                    )
                    if dist_mm <= max_radius_mm:
                        val = float(volume_hu[gz + dz, gy + dy, gx + dx])
                        if hu_min <= val <= hu_max and abs(val - seed_val) <= intensity_tol_hu:
                            segmented[nz_l, ny_l, nx_l] = True
                            queue.append((nz_l, ny_l, nx_l))

    tot_volume_mm3 = float(voxel_count * voxel_vol_mm3)
    if voxel_count == 0:
        eq_diameter_mm = 0.0
        is_empty = True
    else:
        eq_diameter_mm = float(2.0 * ((3.0 * tot_volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0))
        is_empty = False

    return {
        "voxel_count": voxel_count,
        "volume_mm3": round(tot_volume_mm3, 2),
        "equivalent_diameter_mm": round(eq_diameter_mm, 2),
        "is_empty": is_empty,
        "is_oversized": oversized or (eq_diameter_mm > max_diameter_mm),
        "seed_hu": seed_val,
    }

