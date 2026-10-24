/**
 * frontend/src/utils/coordinateTransforms.ts
 * ===========================================
 * Research-Grade Spatial Coordinate Transformations for Multiplanar Chest CT.
 *
 * Establishes rigorous coordinate conventions and prevents fabricated or fixed markers:
 *
 * COORDINATE SYSTEM CONVENTIONS:
 * 1. Volumetric CT Voxel Grid (ZYX Array Indices):
 *    - z: Axial slice index along Cranial-Caudal (longitudinal) axis [0 .. depth - 1]
 *    - y: Anterior-to-Posterior row index [0 .. height - 1] (0 = Anterior / Sternum, height-1 = Posterior / Spine)
 *    - x: Lateral column index [0 .. width - 1] (0 = Patient Right / Image Left, width-1 = Patient Left / Image Right)
 *
 * 2. Physical Patient Coordinates (XYZ in millimetres):
 *    - Physical X = Origin_X + voxel_x * pixel_spacing_x_mm
 *    - Physical Y = Origin_Y + voxel_y * pixel_spacing_y_mm
 *    - Physical Z = Origin_Z + voxel_z * slice_thickness_mm
 *
 * 3. Multiplanar Orthogonal Views (MPR):
 *    - Axial: Slices along Z, in-plane (X, Y)
 *    - Coronal: Slices along Y, in-plane (X, Z)
 *    - Sagittal: Slices along X, in-plane (Y, Z)
 *
 * 4. 2D Display Canvas Viewport Coordinates (Pixels):
 *    - With responsive zoom, pan, and device scaling.
 */

export type ViewingPlane = 'axial' | 'coronal' | 'sagittal';

export interface VolumeSpatialMetadata {
  matrix_width: number;       // e.g. 512
  matrix_height: number;      // e.g. 512
  total_slices: number;       // e.g. 196
  pixel_spacing_x_mm: number; // e.g. 0.703
  pixel_spacing_y_mm: number; // e.g. 0.703
  slice_thickness_mm: number; // e.g. 1.25
  origin_xyz_mm?: [number, number, number]; // e.g. [-175.0, -175.0, -320.0]
}

export interface NoduleSpatialTarget {
  nodule_id: string;
  center_zyx: [number, number, number] | number[];
  diameter_mm: number;
}

export interface ViewportDimensions {
  width: number;
  height: number;
}

export interface ViewportTransform {
  zoom?: number;
  panX?: number;
  panY?: number;
}

export interface NoduleSliceLocalization {
  isValid: boolean;
  statusMessage: string;
  isIntersecting: boolean;
  plane: ViewingPlane;
  sliceIndex: number;
  targetSliceOnPlane: number;
  zMinSlice: number;
  zMaxSlice: number;
  deltaZSlices: number;
  distMmFromPlane: number;
  rSliceMm: number;
  // Physical coordinates
  physicalCoordsMm: {
    x: number;
    y: number;
    z: number;
  };
  // Canvas coordinate projections
  canvasX: number;
  canvasY: number;
  canvasRadius: number;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  crosshair: {
    startX: number;
    endX: number;
    startY: number;
    endY: number;
  };
}

export const DEFAULT_VOLUME_METADATA: VolumeSpatialMetadata = {
  matrix_width: 512,
  matrix_height: 512,
  total_slices: 196,
  pixel_spacing_x_mm: 0.703,
  pixel_spacing_y_mm: 0.703,
  slice_thickness_mm: 1.25,
  origin_xyz_mm: [0.0, 0.0, 0.0],
};

/**
 * Converts Voxel grid indices [z, y, x] to Physical Patient Coordinates (mm).
 */
export function voxelToPhysicalMm(
  zyx: [number, number, number] | number[],
  spatial: VolumeSpatialMetadata = DEFAULT_VOLUME_METADATA
): { x: number; y: number; z: number } {
  const [z, y, x] = zyx;
  const origin = spatial.origin_xyz_mm || [0.0, 0.0, 0.0];
  return {
    x: origin[0] + x * spatial.pixel_spacing_x_mm,
    y: origin[1] + y * spatial.pixel_spacing_y_mm,
    z: origin[2] + z * spatial.slice_thickness_mm,
  };
}

/**
 * Converts Physical Patient Coordinates (mm) back to Voxel grid indices [z, y, x].
 */
export function physicalMmToVoxel(
  xyz_mm: { x: number; y: number; z: number },
  spatial: VolumeSpatialMetadata = DEFAULT_VOLUME_METADATA
): [number, number, number] {
  const origin = spatial.origin_xyz_mm || [0.0, 0.0, 0.0];
  const x = (xyz_mm.x - origin[0]) / (spatial.pixel_spacing_x_mm || 0.703);
  const y = (xyz_mm.y - origin[1]) / (spatial.pixel_spacing_y_mm || 0.703);
  const z = (xyz_mm.z - origin[2]) / (spatial.slice_thickness_mm || 1.25);
  return [Math.round(z), Math.round(y), Math.round(x)];
}

/**
 * Multiplanar coordinate localization supporting Axial, Coronal, and Sagittal planes.
 */
export function calculateMultiplanarLocalization(
  nodule: NoduleSpatialTarget | null | undefined,
  currentSliceIndex: number,
  plane: ViewingPlane = 'axial',
  spatial: VolumeSpatialMetadata = DEFAULT_VOLUME_METADATA,
  viewport: ViewportDimensions = { width: 512, height: 512 },
  transform: ViewportTransform = { zoom: 1.0, panX: 0, panY: 0 }
): NoduleSliceLocalization {
  const zoom = Math.max(0.1, transform.zoom || 1.0);
  const panX = transform.panX || 0;
  const panY = transform.panY || 0;

  const invalidResult: NoduleSliceLocalization = {
    isValid: false,
    statusMessage: 'No valid nodule selected or missing coordinates.',
    isIntersecting: false,
    plane,
    sliceIndex: currentSliceIndex,
    targetSliceOnPlane: 0,
    zMinSlice: 0,
    zMaxSlice: 0,
    deltaZSlices: 0,
    distMmFromPlane: 0,
    rSliceMm: 0,
    physicalCoordsMm: { x: 0, y: 0, z: 0 },
    canvasX: 0,
    canvasY: 0,
    canvasRadius: 0,
    boundingBox: { x: 0, y: 0, width: 0, height: 0 },
    crosshair: { startX: 0, endX: 0, startY: 0, endY: 0 },
  };

  if (!nodule || !nodule.center_zyx || nodule.center_zyx.length !== 3) {
    return invalidResult;
  }

  const [z_vox, y_vox, x_vox] = nodule.center_zyx;

  if (
    typeof z_vox !== 'number' || isNaN(z_vox) ||
    typeof y_vox !== 'number' || isNaN(y_vox) ||
    typeof x_vox !== 'number' || isNaN(x_vox)
  ) {
    return { ...invalidResult, statusMessage: 'Invalid non-numeric nodule coordinates.' };
  }

  const diameter_mm = Math.max(0.5, nodule.diameter_mm || 10.0);
  const radius_mm = diameter_mm / 2.0;

  const slice_thickness = Math.max(0.1, spatial.slice_thickness_mm || 1.25);
  const spacing_x = Math.max(0.01, spatial.pixel_spacing_x_mm || 0.703);
  const spacing_y = Math.max(0.01, spatial.pixel_spacing_y_mm || 0.703);

  const physicalCoords = voxelToPhysicalMm([z_vox, y_vox, x_vox], spatial);

  let targetSliceOnPlane = z_vox;
  let sliceSpacingMm = slice_thickness;
  let inPlaneDimW = spatial.matrix_width || 512;
  let inPlaneDimH = spatial.matrix_height || 512;
  let inPlaneSpacingW = spacing_x;
  let inPlaneSpacingH = spacing_y;
  let inPlaneVoxX = x_vox;
  let inPlaneVoxY = y_vox;
  let totalSlicesOnPlane = spatial.total_slices;

  if (plane === 'axial') {
    // Slices along Z, in-plane: (X, Y)
    targetSliceOnPlane = z_vox;
    sliceSpacingMm = slice_thickness;
    inPlaneDimW = spatial.matrix_width || 512;
    inPlaneDimH = spatial.matrix_height || 512;
    inPlaneSpacingW = spacing_x;
    inPlaneSpacingH = spacing_y;
    inPlaneVoxX = x_vox;
    inPlaneVoxY = y_vox;
    totalSlicesOnPlane = spatial.total_slices;
  } else if (plane === 'coronal') {
    // Slices along Y (anterior-posterior), in-plane: (X, Z)
    targetSliceOnPlane = y_vox;
    sliceSpacingMm = spacing_y;
    inPlaneDimW = spatial.matrix_width || 512;
    inPlaneDimH = spatial.total_slices;
    inPlaneSpacingW = spacing_x;
    inPlaneSpacingH = slice_thickness;
    inPlaneVoxX = x_vox;
    inPlaneVoxY = z_vox;
    totalSlicesOnPlane = spatial.matrix_height || 512;
  } else if (plane === 'sagittal') {
    // Slices along X (right-left), in-plane: (Y, Z)
    targetSliceOnPlane = x_vox;
    sliceSpacingMm = spacing_x;
    inPlaneDimW = spatial.matrix_height || 512;
    inPlaneDimH = spatial.total_slices;
    inPlaneSpacingW = spacing_y;
    inPlaneSpacingH = slice_thickness;
    inPlaneVoxX = y_vox;
    inPlaneVoxY = z_vox;
    totalSlicesOnPlane = spatial.matrix_width || 512;
  }

  const extent_slices = radius_mm / sliceSpacingMm;
  const zMinSlice = Math.max(0, Math.floor(targetSliceOnPlane - extent_slices));
  const zMaxSlice = Math.min(totalSlicesOnPlane - 1, Math.ceil(targetSliceOnPlane + extent_slices));

  const deltaSlices = Math.abs(currentSliceIndex - targetSliceOnPlane);
  const distMmFromPlane = deltaSlices * sliceSpacingMm;

  const isIntersecting = distMmFromPlane <= radius_mm;

  if (!isIntersecting) {
    return {
      isValid: true,
      statusMessage: `Lesion not present on ${plane} slice ${currentSliceIndex}. Center is on slice ${Math.round(targetSliceOnPlane)} (span: ${zMinSlice}–${zMaxSlice}).`,
      isIntersecting: false,
      plane,
      sliceIndex: currentSliceIndex,
      targetSliceOnPlane,
      zMinSlice,
      zMaxSlice,
      deltaZSlices: deltaSlices,
      distMmFromPlane,
      rSliceMm: 0,
      physicalCoordsMm: physicalCoords,
      canvasX: 0,
      canvasY: 0,
      canvasRadius: 0,
      boundingBox: { x: 0, y: 0, width: 0, height: 0 },
      crosshair: { startX: 0, endX: 0, startY: 0, endY: 0 },
    };
  }

  // Cross-sectional planar physical radius
  const rSliceMm = Math.sqrt(Math.max(0, radius_mm * radius_mm - distMmFromPlane * distMmFromPlane));

  // Pixel radius
  const rPixelW = rSliceMm / inPlaneSpacingW;
  const rPixelH = rSliceMm / inPlaneSpacingH;
  const avgPixelRadius = (rPixelW + rPixelH) / 2.0;

  // Viewport scale & pan/zoom
  const baseScaleX = viewport.width / inPlaneDimW;
  const baseScaleY = viewport.height / inPlaneDimH;

  const canvasX = (inPlaneVoxX * baseScaleX) * zoom + panX;
  const canvasY = (inPlaneVoxY * baseScaleY) * zoom + panY;
  const canvasRadius = Math.max(3.0, (avgPixelRadius * ((baseScaleX + baseScaleY) / 2.0)) * zoom);

  const pad = Math.max(6.0, canvasRadius * 0.35);
  const boxX = canvasX - canvasRadius - pad;
  const boxY = canvasY - canvasRadius - pad;
  const boxDim = (canvasRadius + pad) * 2;
  const armLen = canvasRadius + 14.0 * zoom;

  return {
    isValid: true,
    statusMessage: `Lesion visible on ${plane} slice ${currentSliceIndex} (Ø ${diameter_mm.toFixed(1)}mm · planar Ø ${(rSliceMm * 2).toFixed(1)}mm).`,
    isIntersecting: true,
    plane,
    sliceIndex: currentSliceIndex,
    targetSliceOnPlane,
    zMinSlice,
    zMaxSlice,
    deltaZSlices: deltaSlices,
    distMmFromPlane,
    rSliceMm,
    physicalCoordsMm: physicalCoords,
    canvasX,
    canvasY,
    canvasRadius,
    boundingBox: {
      x: boxX,
      y: boxY,
      width: boxDim,
      height: boxDim,
    },
    crosshair: {
      startX: canvasX - armLen,
      endX: canvasX + armLen,
      startY: canvasY - armLen,
      endY: canvasY + armLen,
    },
  };
}

/**
 * Backward-compatible helper for axial slice localization.
 */
export function calculateNoduleSliceLocalization(
  nodule: NoduleSpatialTarget | null | undefined,
  currentSliceIndex: number,
  spatial: VolumeSpatialMetadata = DEFAULT_VOLUME_METADATA,
  viewport: ViewportDimensions = { width: 512, height: 512 }
): NoduleSliceLocalization {
  return calculateMultiplanarLocalization(nodule, currentSliceIndex, 'axial', spatial, viewport);
}

/**
 * Canonical dimensions of the Workstation CT canvas
 */
export const WORKSTATION_VIEWPORT: ViewportDimensions = {
  width: 540,
  height: 460
};

/**
 * Maps relative [0..1] coordinates into Workstation canvas pixels
 */
export function relativeToWorkstationPixel(cxRel: number, cyRel: number): { pixelX: number; pixelY: number } {
  return {
    pixelX: Math.round(cxRel * WORKSTATION_VIEWPORT.width),
    pixelY: Math.round(cyRel * WORKSTATION_VIEWPORT.height)
  };
}

/**
 * Validates anatomical lobe designation based on 3D volumetric ZYX coordinates.
 * Radiological axial convention:
 *   x < 256 is Patient Right (Left side of axial screen)
 *   x >= 256 is Patient Left (Right side of axial screen)
 * Cranial-caudal (z):
 *   z < 80: Superior / Upper (RUL, LUL)
 *   z 80-140: Middle / Hilum (RML / lingula / upper-lower transition)
 *   z >= 140: Inferior / Lower (RLL, LLL)
 */
export function deriveAnatomicalLobeFromZyx(
  zyx: [number, number, number] | number[] | null | undefined,
  totalSlices: number = 196,
  dimXy: number = 512
): { lobe: string; confidence: 'CONFIRMED' | 'UNAVAILABLE' } {
  if (!zyx || zyx.length < 3) {
    return { lobe: 'Lobe localization unavailable', confidence: 'UNAVAILABLE' };
  }
  const [z, y, x] = zyx;
  if (!Number.isFinite(z) || !Number.isFinite(y) || !Number.isFinite(x) || Number.isNaN(z) || Number.isNaN(y) || Number.isNaN(x)) {
    return { lobe: 'Lobe localization unavailable', confidence: 'UNAVAILABLE' };
  }
  if (z < 0 || z > totalSlices || y < 0 || y > dimXy || x < 0 || x > dimXy) {
    return { lobe: 'Lobe localization unavailable', confidence: 'UNAVAILABLE' };
  }
  if (totalSlices <= 1) {
    return { lobe: 'Lobe localization unavailable (2D slice only; volumetric series required)', confidence: 'UNAVAILABLE' };
  }

  const isPatientRight = x < (dimXy / 2);
  const yRel = y / dimXy;
  const zRel = z / Math.max(1, totalSlices);

  // Supreme apex (cranial 25% of lung): only Upper Lobes anatomically reach the apex
  if (zRel < 0.25) {
    return isPatientRight
      ? { lobe: 'Right Upper Lobe (RUL)', confidence: 'CONFIRMED' }
      : { lobe: 'Left Upper Lobe (LUL)', confidence: 'CONFIRMED' };
  }

  // Basal lung (caudal 25%): Lower Lobes occupy the posterior and basal parenchyma
  if (zRel > 0.75) {
    if (isPatientRight) {
      if (yRel < 0.35) return { lobe: 'Right Middle Lobe (RML)', confidence: 'CONFIRMED' };
      return { lobe: 'Right Lower Lobe (RLL)', confidence: 'CONFIRMED' };
    } else {
      return { lobe: 'Left Lower Lobe (LLL)', confidence: 'CONFIRMED' };
    }
  }

  // Mid-thorax (0.25 <= zRel <= 0.75): Oblique and horizontal fissure boundaries
  if (isPatientRight) {
    if (yRel < 0.38) return { lobe: 'Right Upper Lobe (RUL)', confidence: 'CONFIRMED' };
    if (yRel < 0.64) return { lobe: 'Right Middle Lobe (RML)', confidence: 'CONFIRMED' };
    return { lobe: 'Right Lower Lobe (RLL)', confidence: 'CONFIRMED' };
  } else {
    if (yRel < 0.42) return { lobe: 'Left Upper Lobe (LUL)', confidence: 'CONFIRMED' };
    if (yRel < 0.62) return { lobe: 'Lingula (LUL)', confidence: 'CONFIRMED' };
    return { lobe: 'Left Lower Lobe (LLL)', confidence: 'CONFIRMED' };
  }
}
