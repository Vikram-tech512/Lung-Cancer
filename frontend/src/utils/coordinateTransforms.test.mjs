/**
 * frontend/src/utils/coordinateTransforms.test.mjs
 * =================================================
 * Unit tests validating CT coordinate transformations across 3 distinct synthetic volumes
 * with deliberately different nodule locations.
 *
 * Validates:
 * 1. Each marker appears at the correct transformed location.
 * 2. Changing selected nodule changes the marker location.
 * 3. Changing slices updates marker visibility and position correctly.
 * 4. Multiple nodules can be displayed independently without interference.
 * 5. Missing or invalid coordinates never trigger a fabricated marker.
 * 6. Resizing the canvas maintains anatomical correspondence.
 * 7. Multiplanar orthogonal reconstruction (Axial, Coronal, Sagittal) transforms properly.
 * 8. Physical Patient Coordinates (mm) match Voxel coordinates bidirectionally.
 * 9. Viewport Zoom & Pan preserve anatomical target anchoring.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  calculateNoduleSliceLocalization,
  calculateMultiplanarLocalization,
  voxelToPhysicalMm,
  physicalMmToVoxel,
  WORKSTATION_VIEWPORT,
  relativeToWorkstationPixel,
  deriveAnatomicalLobeFromZyx
} from './coordinateTransforms.ts';

describe('Phase 3: Coordinate-Driven CT Localization Validation', () => {

  // Synthetic Volume 1: Standard Siemens Thorax (512x512, 196 slices, RUL lesion)
  const vol1 = {
    matrix_width: 512,
    matrix_height: 512,
    total_slices: 196,
    pixel_spacing_x_mm: 0.703,
    pixel_spacing_y_mm: 0.703,
    slice_thickness_mm: 1.25,
    origin_xyz_mm: [-180.0, -180.0, -245.0],
  };

  // Synthetic Volume 2: GE Apex Large Matrix (512x512, 250 slices, LLL subpleural lesion)
  const vol2 = {
    matrix_width: 512,
    matrix_height: 512,
    total_slices: 250,
    pixel_spacing_x_mm: 0.625,
    pixel_spacing_y_mm: 0.625,
    slice_thickness_mm: 1.0,
    origin_xyz_mm: [-160.0, -160.0, -250.0],
  };

  // Synthetic Volume 3: High-Res Research Volume (768x768, 300 slices, RML lesion)
  const vol3 = {
    matrix_width: 768,
    matrix_height: 768,
    total_slices: 300,
    pixel_spacing_x_mm: 0.50,
    pixel_spacing_y_mm: 0.50,
    slice_thickness_mm: 0.75,
    origin_xyz_mm: [0.0, 0.0, 0.0],
  };

  const noduleA = {
    nodule_id: 'SYNTH-NODULE-A',
    center_zyx: [92, 180, 310], // z=92, y=180, x=310
    diameter_mm: 14.8,
  };

  const noduleB = {
    nodule_id: 'SYNTH-NODULE-B',
    center_zyx: [160, 350, 140], // z=160, y=350, x=140
    diameter_mm: 20.0,
  };

  const noduleC = {
    nodule_id: 'SYNTH-NODULE-C',
    center_zyx: [210, 420, 520], // z=210, y=420, x=520 (on 768x768)
    diameter_mm: 12.0,
  };

  test('Test 1: Correct transformed coordinates at exact center slice', () => {
    const locA = calculateNoduleSliceLocalization(noduleA, 92, vol1, { width: 512, height: 512 });
    assert.equal(locA.isValid, true);
    assert.equal(locA.isIntersecting, true);
    assert.equal(locA.canvasX, 310);
    assert.equal(locA.canvasY, 180);
    assert.ok(locA.canvasRadius > 5);

    // Crosshair centered at (310, 180)
    assert.ok(locA.crosshair.startX < 310);
    assert.ok(locA.crosshair.endX > 310);
    assert.ok(locA.crosshair.startY < 180);
    assert.ok(locA.crosshair.endY > 180);
  });

  test('Test 2: Changing selected nodule changes marker location', () => {
    const locA = calculateNoduleSliceLocalization(noduleA, 92, vol1, { width: 512, height: 512 });
    const locB = calculateNoduleSliceLocalization(noduleB, 160, vol1, { width: 512, height: 512 });

    assert.notEqual(locA.canvasX, locB.canvasX);
    assert.notEqual(locA.canvasY, locB.canvasY);
    assert.equal(locB.canvasX, 140);
    assert.equal(locB.canvasY, 350);
  });

  test('Test 3: Slice visibility and extent updates correctly', () => {
    const sliceExact = calculateNoduleSliceLocalization(noduleA, 92, vol1);
    assert.equal(sliceExact.isIntersecting, true);

    const sliceEdge = calculateNoduleSliceLocalization(noduleA, 96, vol1);
    assert.equal(sliceEdge.isIntersecting, true);
    assert.ok(sliceEdge.canvasRadius < sliceExact.canvasRadius);

    // Slice 120 is far away (> 7.4mm): MUST NOT INTERSECT
    const sliceFar = calculateNoduleSliceLocalization(noduleA, 120, vol1);
    assert.equal(sliceFar.isIntersecting, false);
    assert.equal(sliceFar.canvasRadius, 0);
    assert.ok(sliceFar.statusMessage.includes('not present on axial slice 120'));
  });

  test('Test 4: Multiple nodules displayed independently', () => {
    const studyNodules = [noduleA, noduleB];
    const locsOnSlice92 = studyNodules.map(n => calculateNoduleSliceLocalization(n, 92, vol1));

    assert.equal(locsOnSlice92[0].isIntersecting, true);
    assert.equal(locsOnSlice92[0].canvasX, 310);

    assert.equal(locsOnSlice92[1].isIntersecting, false);

    const locsOnSlice160 = studyNodules.map(n => calculateNoduleSliceLocalization(n, 160, vol1));
    assert.equal(locsOnSlice160[0].isIntersecting, false);
    assert.equal(locsOnSlice160[1].isIntersecting, true);
    assert.equal(locsOnSlice160[1].canvasX, 140);
  });

  test('Test 5: Missing or invalid coordinates never trigger fabricated markers', () => {
    const invalidNodule = { nodule_id: 'INVALID', center_zyx: null, diameter_mm: 10 };
    const loc = calculateNoduleSliceLocalization(invalidNodule, 92, vol1);
    assert.equal(loc.isValid, false);
    assert.equal(loc.isIntersecting, false);
    assert.equal(loc.canvasX, 0);
    assert.equal(loc.canvasY, 0);
    assert.equal(loc.canvasRadius, 0);

    const nanNodule = { nodule_id: 'NAN', center_zyx: [NaN, 100, 200], diameter_mm: 10 };
    const locNaN = calculateNoduleSliceLocalization(nanNodule, 92, vol1);
    assert.equal(locNaN.isValid, false);
    assert.equal(locNaN.isIntersecting, false);
  });

  test('Test 6: Canvas resizing maintains anatomical correspondence', () => {
    const loc512 = calculateNoduleSliceLocalization(noduleA, 92, vol1, { width: 512, height: 512 });
    assert.equal(loc512.canvasX, 310);
    assert.equal(loc512.canvasY, 180);

    const loc256 = calculateNoduleSliceLocalization(noduleA, 92, vol1, { width: 256, height: 256 });
    assert.equal(loc256.canvasX, 155);
    assert.equal(loc256.canvasY, 90);

    const loc1024 = calculateNoduleSliceLocalization(noduleA, 92, vol1, { width: 1024, height: 1024 });
    assert.equal(loc1024.canvasX, 620);
    assert.equal(loc1024.canvasY, 360);

    assert.equal(loc512.canvasX / 512, loc256.canvasX / 256);
    assert.equal(loc512.canvasX / 512, loc1024.canvasX / 1024);
  });

  test('Test 7: Synthetic Volume 3 (768x768) transforms high-res matrix properly', () => {
    const locC = calculateNoduleSliceLocalization(noduleC, 210, vol3, { width: 512, height: 512 });
    assert.equal(locC.isValid, true);
    assert.equal(locC.isIntersecting, true);
    assert.ok(Math.abs(locC.canvasX - (520 * 512 / 768)) < 1e-4);
  });

  test('Test 8: Multiplanar Localization (Coronal and Sagittal views)', () => {
    // Coronal view: slices along Y (noduleA center y=180)
    // In coronal slice 180, noduleA must be intersecting! In-plane: (X=310, Z=92)
    const corExact = calculateMultiplanarLocalization(noduleA, 180, 'coronal', vol1, { width: 512, height: 512 });
    assert.equal(corExact.isValid, true);
    assert.equal(corExact.isIntersecting, true);
    assert.equal(corExact.canvasX, 310);
    // Z coordinate scaled to canvas: 92 * (512 / 196) = 240.326...
    assert.ok(Math.abs(corExact.canvasY - (92 * 512 / 196)) < 1e-3);

    // Coronal slice far away (y=240, > 7.4mm): must NOT intersect
    const corFar = calculateMultiplanarLocalization(noduleA, 240, 'coronal', vol1, { width: 512, height: 512 });
    assert.equal(corFar.isIntersecting, false);

    // Sagittal view: slices along X (noduleA center x=310)
    // In sagittal slice 310, noduleA must be intersecting! In-plane: (Y=180, Z=92)
    const sagExact = calculateMultiplanarLocalization(noduleA, 310, 'sagittal', vol1, { width: 512, height: 512 });
    assert.equal(sagExact.isValid, true);
    assert.equal(sagExact.isIntersecting, true);
    assert.equal(sagExact.canvasX, 180);
    assert.ok(Math.abs(sagExact.canvasY - (92 * 512 / 196)) < 1e-3);
  });

  test('Test 9: Physical Patient Coordinates bidirectional mapping', () => {
    const phys = voxelToPhysicalMm([92, 180, 310], vol1);
    // Origin: [-180, -180, -245]
    // x_mm = -180 + 310 * 0.703 = 37.93 mm
    // y_mm = -180 + 180 * 0.703 = -53.46 mm
    // z_mm = -245 + 92 * 1.25 = -130.0 mm
    assert.ok(Math.abs(phys.x - (-180 + 310 * 0.703)) < 1e-3);
    assert.ok(Math.abs(phys.y - (-180 + 180 * 0.703)) < 1e-3);
    assert.ok(Math.abs(phys.z - (-245 + 92 * 1.25)) < 1e-3);

    const voxBack = physicalMmToVoxel(phys, vol1);
    assert.deepEqual(voxBack, [92, 180, 310]);
  });

  test('Test 10: Viewport Zoom & Pan preserve anatomical target locking', () => {
    // Zoom 2x with pan (+50px, +30px)
    const locZoom = calculateMultiplanarLocalization(
      noduleA, 92, 'axial', vol1,
      { width: 512, height: 512 },
      { zoom: 2.0, panX: 50, panY: 30 }
    );
    // Base: X=310, Y=180 -> With 2x + pan: X = 310 * 2 + 50 = 670, Y = 180 * 2 + 30 = 390
    assert.equal(locZoom.canvasX, 670);
    assert.equal(locZoom.canvasY, 390);
    // Radius scales with zoom
    const locBase = calculateNoduleSliceLocalization(noduleA, 92, vol1, { width: 512, height: 512 });
    assert.ok(Math.abs(locZoom.canvasRadius - locBase.canvasRadius * 2.0) < 1e-3);
  });
});

describe('Phase 4: Workstation Dynamic Coordinates & Anatomical Lobe Derivation', () => {

  test('Test 11: Workstation Canvas Canonical Dimensions (540x460)', () => {
    assert.equal(WORKSTATION_VIEWPORT.width, 540);
    assert.equal(WORKSTATION_VIEWPORT.height, 460);
  });

  test('Test 12: Workstation Relative to Pixel Coordinates mapping consistency', () => {
    // Relative coordinates [cxRel = 0.3515, cyRel = 0.3326] (CASE-DEMO-LIDC-0087)
    const { pixelX, pixelY } = relativeToWorkstationPixel(180 / 512.0, 170 / 512.0);
    assert.equal(pixelX, Math.round((180 / 512.0) * 540));
    assert.equal(pixelY, Math.round((170 / 512.0) * 460));
    assert.equal(pixelX, 190);
    assert.equal(pixelY, 153);
  });

  test('Test 13: Volumetric Lobe Derivation matches radiological axial conventions', () => {
    // Radiological Axial: x < 256 is Patient Right, x >= 256 is Patient Left
    // 1. Superior Right -> RUL
    const resRUL = deriveAnatomicalLobeFromZyx([70, 170, 180], 196);
    assert.equal(resRUL.lobe, 'Right Upper Lobe (RUL)');
    assert.equal(resRUL.confidence, 'CONFIRMED');

    // 2. Middle Right -> RML
    const resRML = deriveAnatomicalLobeFromZyx([110, 240, 160], 196);
    assert.equal(resRML.lobe, 'Right Middle Lobe (RML)');
    assert.equal(resRML.confidence, 'CONFIRMED');

    // 3. Inferior Right -> RLL
    const resRLL = deriveAnatomicalLobeFromZyx([165, 360, 175], 196);
    assert.equal(resRLL.lobe, 'Right Lower Lobe (RLL)');
    assert.equal(resRLL.confidence, 'CONFIRMED');

    // 4. Superior Left -> LUL
    const resLUL = deriveAnatomicalLobeFromZyx([75, 200, 320], 196);
    assert.equal(resLUL.lobe, 'Left Upper Lobe (LUL)');
    assert.equal(resLUL.confidence, 'CONFIRMED');

    // 5. Inferior Left -> LLL
    const resLLL = deriveAnatomicalLobeFromZyx([150, 330, 350], 196);
    assert.equal(resLLL.lobe, 'Left Lower Lobe (LLL)');
    assert.equal(resLLL.confidence, 'CONFIRMED');
  });

  test('Test 14: Non-finite, missing, or 2D coordinates return honest unavailable state', () => {
    assert.equal(deriveAnatomicalLobeFromZyx(null).lobe, 'Lobe localization unavailable');
    assert.equal(deriveAnatomicalLobeFromZyx(undefined).lobe, 'Lobe localization unavailable');
    assert.equal(deriveAnatomicalLobeFromZyx([NaN, 180, 200]).lobe, 'Lobe localization unavailable');
    assert.equal(deriveAnatomicalLobeFromZyx([92, Infinity, 200]).lobe, 'Lobe localization unavailable');
    // 2D slice only (totalSlices <= 1) honestly reports unavailable without guessing 3D lobes
    const res2D = deriveAnatomicalLobeFromZyx([0, 240, 160], 1);
    assert.equal(res2D.lobe, 'Lobe localization unavailable (2D slice only; volumetric series required)');
    assert.equal(res2D.confidence, 'UNAVAILABLE');
  });

  test('Test 15: Multi-nodule case isolation and independent coordinates', () => {
    // Study CASE-DEMO-LIDC-0124 has 2 distinct nodules
    const nodule1 = { nodule_id: 'NODULE-01', center_zyx: [135, 240, 160], diameter_mm: 12.4 };
    const nodule2 = { nodule_id: 'NODULE-02', center_zyx: [65, 340, 380], diameter_mm: 6.8 };

    const lobe1 = deriveAnatomicalLobeFromZyx(nodule1.center_zyx, 196);
    const lobe2 = deriveAnatomicalLobeFromZyx(nodule2.center_zyx, 196);

    const pix1 = relativeToWorkstationPixel(nodule1.center_zyx[2] / 512.0, nodule1.center_zyx[1] / 512.0);
    const pix2 = relativeToWorkstationPixel(nodule2.center_zyx[2] / 512.0, nodule2.center_zyx[1] / 512.0);

    // Completely distinct positions and lobes
    assert.notEqual(pix1.pixelX, pix2.pixelX);
    assert.notEqual(pix1.pixelY, pix2.pixelY);
    assert.notEqual(lobe1.lobe, lobe2.lobe);

    assert.equal(lobe1.lobe, 'Right Middle Lobe (RML)');
    assert.equal(lobe2.lobe, 'Left Lower Lobe (LLL)');

    // Apex lesion at z=20 on patient right correctly assigns RUL, preventing fabricated RML
    const resApex = deriveAnatomicalLobeFromZyx([20, 240, 160], 196);
    assert.equal(resApex.lobe, 'Right Upper Lobe (RUL)');
  });
});
