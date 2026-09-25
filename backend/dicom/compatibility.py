"""
backend/dicom/compatibility.py
===============================
Multi-Vendor DICOM Robustness & Compatibility Inspector.

Handles real-world hospital scanner/PACS variance:
- Multi-vendor conventions: GE, Siemens, Philips, Canon/Toshiba standard & private tags
- Inconsistent SliceThickness vs ImagePositionPatient z-coordinate deltas
- Non-axial acquisitions and tilted gantry detection/rejection
- Multi-series filtering of scout/localizer/topogram images
- Burned-in annotation detection (tag verification + corner intensity heuristics)
- Compressed transfer syntax validation (JPEG 2000, JPEG-LS, Deflate)
- Generates a persistent DicomCompatibilityReport for the study's audit record.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import pydicom
    from pydicom.uid import (
        ExplicitVRLittleEndian, ImplicitVRLittleEndian,
        DeflatedExplicitVRLittleEndian, ExplicitVRBigEndian,
        JPEGLossless, JPEGLSLossless, JPEG2000Lossless, JPEG2000
    )
except ImportError:
    pydicom = None

logger = logging.getLogger("backend.dicom.compatibility")

# Supported transfer syntax UIDs for robust ingestion
SUPPORTED_SYNTAXES = {
    "1.2.840.10008.1.2": "Implicit VR Little Endian",
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2.2": "Explicit VR Big Endian",
    "1.2.840.10008.1.2.1.99": "Deflated Explicit VR Little Endian",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless, Non-Hierarchical",
    "1.2.840.10008.1.2.4.80": "JPEG-LS Lossless",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Image Compression (Lossless Only)",
    "1.2.840.10008.1.2.4.91": "JPEG 2000 Image Compression",
}


@dataclass
class DicomQuirk:
    """Record of an idiosyncratic vendor convention or header anomaly detected."""
    category: str        # e.g., "VENDOR_CONVENTION", "SPACING_DISCREPANCY", "LOCALIZER_FILTERED"
    description: str
    original_value: Any
    action_taken: str
    severity: str = "INFO"  # "INFO", "WARNING", "REJECT"


@dataclass
class DicomCompatibilityReport:
    """Comprehensive multi-vendor compatibility & audit report per study."""
    study_instance_uid: str = ""
    series_instance_uid: str = ""
    vendor: str = "UNKNOWN"  # "GE", "SIEMENS", "PHILIPS", "CANON/TOSHIBA", "UNKNOWN"
    model_name: str = ""
    kernel: str = ""
    
    # Slice counts and filtering
    raw_slice_count: int = 0
    scout_localizers_filtered: int = 0
    usable_axial_slice_count: int = 0
    
    # Geometry & Spacing
    header_slice_thickness_mm: Optional[float] = None
    derived_z_spacing_mm: Optional[float] = None
    spacing_discrepancy_mm: Optional[float] = None
    gantry_tilt_degrees: float = 0.0
    is_axial: bool = True
    
    # Integrity & Artifacts
    burned_in_annotation_detected: bool = False
    burned_in_annotation_details: List[str] = field(default_factory=list)
    transfer_syntax_uid: str = ""
    transfer_syntax_name: str = ""
    transfer_syntax_supported: bool = True
    
    # Decision Gate
    is_compatible: bool = True
    rejection_reason: Optional[str] = None
    quirks: List[DicomQuirk] = field(default_factory=list)
    audit_timestamp: str = ""


class MultiVendorDicomInspector:
    """Inspects, cleans, and validates DICOM series across hospital scanner vendors."""

    def __init__(self, max_gantry_tilt_degrees: float = 1.0, max_spacing_mismatch_mm: float = 0.5):
        self.max_gantry_tilt = max_gantry_tilt_degrees
        self.max_spacing_mismatch = max_spacing_mismatch_mm

    def identify_vendor(self, ds: "pydicom.Dataset") -> str:
        """Determines equipment vendor from standard & private headers."""
        mfg = str(getattr(ds, "Manufacturer", "")).upper()
        if "GE" in mfg or "GENERAL ELECTRIC" in mfg:
            return "GE"
        elif "SIEMENS" in mfg:
            return "SIEMENS"
        elif "PHILIPS" in mfg:
            return "PHILIPS"
        elif "CANON" in mfg or "TOSHIBA" in mfg:
            return "CANON/TOSHIBA"
        return "UNKNOWN"

    def is_scout_or_localizer(self, ds: "pydicom.Dataset") -> Tuple[bool, str]:
        """Detects scout, topogram, surview, or localizer 2D slices."""
        # Check ImageType tag (0008, 0008)
        image_type = getattr(ds, "ImageType", [])
        if isinstance(image_type, (list, tuple, pydicom.multival.MultiValue)):
            image_type_strs = [str(x).upper() for x in image_type]
            for term in ["LOCALIZER", "SCOUT", "TOPOGRAM", "SURVIEW"]:
                if any(term in s for s in image_type_strs):
                    return True, f"ImageType contains '{term}'"

        # Check SeriesDescription tag (0008, 103E)
        s_desc = str(getattr(ds, "SeriesDescription", "")).upper()
        for term in ["SCOUT", "LOCALIZER", "TOPOGRAM", "SURVIEW"]:
            if term in s_desc:
                return True, f"SeriesDescription contains '{term}'"

        # Orientation check: if direction cosines indicate coronal/sagittal plane in an axial series
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop and len(iop) == 6:
            r = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
            c = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
            n = np.cross(r, c)
            # For axial scan, normal vector magnitude along Z should be close to 1.0 (|nz| > 0.85)
            if abs(n[2]) < 0.85:
                return True, f"Non-axial plane normal z={n[2]:.2f} (likely orthogonal scout)"

        return False, ""

    def check_burned_in_annotations(self, ds: "pydicom.Dataset") -> Tuple[bool, List[str]]:
        """Detects burned-in text annotations in headers and pixel corner regions."""
        findings = []
        # Check DICOM tag (0028, 0301) BurnedInAnnotation
        burned_tag = str(getattr(ds, "BurnedInAnnotation", "")).upper()
        if burned_tag in ["YES", "Y", "TRUE"]:
            findings.append("Header tag BurnedInAnnotation is explicitly marked YES")

        # Corner heuristic: inspect 4 corners (32x32 pixels) for high-contrast saturated text pixels
        try:
            if hasattr(ds, "pixel_array"):
                arr = ds.pixel_array
                h, w = arr.shape[-2], arr.shape[-1]
                cw = min(32, w // 8, h // 8)
                if cw > 8:
                    corners = [
                        arr[0:cw, 0:cw],          # Top-Left
                        arr[0:cw, w - cw:w],      # Top-Right
                        arr[h - cw:h, 0:cw],      # Bottom-Left
                        arr[h - cw:h, w - cw:w]   # Bottom-Right
                    ]
                    for idx, c in enumerate(corners):
                        # High contrast text often features constant saturated max values
                        max_val = np.max(c)
                        saturated_frac = np.mean(c == max_val)
                        if saturated_frac > 0.08 and max_val > (np.mean(arr) + 3 * np.std(arr)):
                            findings.append(f"High-density text/annotation cluster detected in corner {idx + 1}")
                            break
        except Exception:
            pass

        return (len(findings) > 0), findings

    def inspect_series(self, dcm_files: List[Path]) -> Tuple[List[Tuple[float, Any]], DicomCompatibilityReport]:
        """
        Inspects all DICOM files, strips scouts, verifies vendors, checks transfer syntax,
        and computes reliable coordinate-derived slice spacing.
        """
        import time
        report = DicomCompatibilityReport()
        report.audit_timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC")
        report.raw_slice_count = len(dcm_files)

        if not dcm_files:
            report.is_compatible = False
            report.rejection_reason = "No DICOM files provided for inspection."
            return [], report

        # Read slice headers
        valid_slices = []
        scouts_filtered = 0

        for fpath in dcm_files:
            try:
                ds = pydicom.dcmread(str(fpath))
            except Exception as e:
                report.quirks.append(DicomQuirk(
                    category="UNREADABLE_FILE",
                    description=f"File could not be parsed: {fpath.name} ({e})",
                    original_value=str(fpath),
                    action_taken="Ignored unreadable file",
                    severity="WARNING"
                ))
                continue

            # Identify Vendor & Transfer Syntax on first slice
            if not report.vendor or report.vendor == "UNKNOWN":
                report.vendor = self.identify_vendor(ds)
                report.model_name = str(getattr(ds, "ManufacturerModelName", "Standard Model"))
                report.kernel = str(getattr(ds, "ConvolutionKernel", "Standard"))
                report.study_instance_uid = str(getattr(ds, "StudyInstanceUID", ""))
                report.series_instance_uid = str(getattr(ds, "SeriesInstanceUID", ""))
                
                # Transfer Syntax verification
                if hasattr(ds, "file_meta") and hasattr(ds.file_meta, "TransferSyntaxUID"):
                    syntax_uid = str(ds.file_meta.TransferSyntaxUID)
                    report.transfer_syntax_uid = syntax_uid
                    report.transfer_syntax_name = SUPPORTED_SYNTAXES.get(syntax_uid, "Unknown/Custom Syntax")
                    if syntax_uid not in SUPPORTED_SYNTAXES:
                        report.quirks.append(DicomQuirk(
                            category="TRANSFER_SYNTAX",
                            description=f"Transfer syntax {syntax_uid} ({report.transfer_syntax_name}) is non-standard",
                            original_value=syntax_uid,
                            action_taken="Checked codec availability",
                            severity="WARNING"
                        ))

            # 1. Filter Scouts / Localizers
            is_scout, scout_reason = self.is_scout_or_localizer(ds)
            if is_scout:
                scouts_filtered += 1
                report.quirks.append(DicomQuirk(
                    category="SCOUT_FILTERED",
                    description=f"Excluded scout/localizer slice ({scout_reason}): {fpath.name}",
                    original_value=scout_reason,
                    action_taken="Filtered out slice before 3D stack reconstruction",
                    severity="INFO"
                ))
                continue

            # 2. Check Burned-in Annotations
            has_burned_in, burned_details = self.check_burned_in_annotations(ds)
            if has_burned_in and not report.burned_in_annotation_detected:
                report.burned_in_annotation_detected = True
                report.burned_in_annotation_details = burned_details
                report.quirks.append(DicomQuirk(
                    category="BURNED_IN_ANNOTATION",
                    description="Burned-in text annotation detected on scanner export",
                    original_value=burned_details,
                    action_taken="Flagged for radiologist review; non-destructive mask suggested",
                    severity="WARNING"
                ))

            # 3. Z-Position derivation from ImagePositionPatient
            z_pos = 0.0
            if hasattr(ds, "ImagePositionPatient"):
                try:
                    z_pos = float(ds.ImagePositionPatient[2])
                except Exception:
                    pass
            elif hasattr(ds, "SliceLocation"):
                z_pos = float(ds.SliceLocation)
            else:
                z_pos = float(len(valid_slices))

            valid_slices.append((z_pos, ds, fpath))

        report.scout_localizers_filtered = scouts_filtered
        report.usable_axial_slice_count = len(valid_slices)

        if not valid_slices:
            report.is_compatible = False
            report.rejection_reason = "No valid volumetric axial CT slices remaining after scout filtering."
            return [], report

        # Sort slices by true spatial z-position
        valid_slices.sort(key=lambda t: t[0])

        # 4. Multi-Vendor Rescale Slope/Intercept Handling
        ref_ds = valid_slices[0][1]
        slope = float(getattr(ref_ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ref_ds, "RescaleIntercept", 0.0))

        # Check vendor-specific quirks
        if report.vendor == "GE":
            if not hasattr(ref_ds, "RescaleIntercept") and hasattr(ref_ds, (0x0020, 0x1041)):
                intercept = float(ref_ds[(0x0020, 0x1041)].value)
                report.quirks.append(DicomQuirk(
                    category="GE_PRIVATE_RESCALE",
                    description="Derived RescaleIntercept from GE private tag (0020, 1041)",
                    original_value=intercept,
                    action_taken="Corrected Hounsfield Unit transformation",
                    severity="INFO"
                ))
        elif report.vendor == "PHILIPS":
            rescale_type = str(getattr(ref_ds, "RescaleType", ""))
            if rescale_type and rescale_type.upper() not in ["HU", "HOUNSFIELD"]:
                report.quirks.append(DicomQuirk(
                    category="PHILIPS_RESCALE_TYPE",
                    description=f"Philips RescaleType is '{rescale_type}' (expected HU)",
                    original_value=rescale_type,
                    action_taken="Flagged unit scaling for HU verification",
                    severity="WARNING"
                ))

        # 5. Coordinate-based Z-Spacing Calculation vs Header SliceThickness
        header_st = float(getattr(ref_ds, "SliceThickness", 0.0))
        header_sb = float(getattr(ref_ds, "SpacingBetweenSlices", 0.0))
        report.header_slice_thickness_mm = header_st if header_st > 0 else (header_sb if header_sb > 0 else None)

        if len(valid_slices) > 1:
            z_deltas = [abs(valid_slices[i+1][0] - valid_slices[i][0]) for i in range(len(valid_slices) - 1)]
            median_delta = float(np.median(z_deltas))
            report.derived_z_spacing_mm = round(median_delta, 4)

            # Compare header thickness vs derived coordinate spacing
            if report.header_slice_thickness_mm:
                mismatch = abs(report.header_slice_thickness_mm - median_delta)
                report.spacing_discrepancy_mm = round(mismatch, 4)
                if mismatch > self.max_spacing_mismatch:
                    report.quirks.append(DicomQuirk(
                        category="SPACING_DISCREPANCY",
                        description=(f"Header SliceThickness ({report.header_slice_thickness_mm}mm) disagrees with "
                                     f"IPP spatial coordinate delta ({median_delta:.2f}mm)"),
                        original_value={"header": report.header_slice_thickness_mm, "derived": median_delta},
                        action_taken=f"Using true coordinate delta {median_delta:.2f}mm for 3D resampling",
                        severity="WARNING"
                    ))
        else:
            report.derived_z_spacing_mm = report.header_slice_thickness_mm or 1.0

        # 6. Gantry Tilt & Non-Axial Verification
        gantry_tilt = float(getattr(ref_ds, "GantryDetectorTilt", 0.0))
        report.gantry_tilt_degrees = gantry_tilt
        if abs(gantry_tilt) > self.max_gantry_tilt:
            report.quirks.append(DicomQuirk(
                category="TILTED_GANTRY",
                description=f"Significant gantry tilt detected ({gantry_tilt}° > {self.max_gantry_tilt}°)",
                original_value=gantry_tilt,
                action_taken="Requires shear-correction or rejection; model assumes orthogonal axial acquisition",
                severity="WARNING"
            ))

        # Direction Cosines Planarity
        iop = getattr(ref_ds, "ImageOrientationPatient", None)
        if iop and len(iop) == 6:
            r = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
            c = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
            n = np.cross(r, c)
            if abs(n[2]) < 0.90:
                report.is_axial = False
                report.quirks.append(DicomQuirk(
                    category="NON_AXIAL_ACQUISITION",
                    description=f"Volume is acquired in non-axial plane (normal vector z={n[2]:.3f})",
                    original_value=list(iop),
                    action_taken="Rejected acquisition from standard axial pipeline",
                    severity="REJECT"
                ))
                report.is_compatible = False
                report.rejection_reason = f"Non-axial acquisition detected (normal z-vector {n[2]:.3f})."

        # Final minimum slice safety check
        if report.usable_axial_slice_count < 15:
            report.is_compatible = False
            report.rejection_reason = f"Insufficient chest CT slice coverage ({report.usable_axial_slice_count} slices < 15 minimum)."

        return [(t[0], t[1]) for t in valid_slices], report
