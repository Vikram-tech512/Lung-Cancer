"""
backend/pipeline/stages/report.py
====================================
Stage 10 — Clinical Report Generation.

Aggregates all upstream stage outputs into a structured,
machine-readable + human-readable clinical JSON report.

Report sections
---------------
1. Patient / case metadata
2. DICOM QA summary
3. CT findings (nodule candidates, largest nodule)
4. Classification verdicts per model
5. Ensemble decision + confidence band
6. Uncertainty estimation (MC Dropout entropy)
7. Radiomics feature summary (top 10 features)
8. XAI localisation summary
9. Recommendations (auto-generated from Lung-RADS)
10. Audit trail (pipeline version, timestamps, model checksums)

Writes:
    ctx.set(REPORT, {
        "report_version":   "2.0",
        "case_id":          "...",
        "generated_at":     "ISO-8601",
        "sections":         {...},
        "lung_rads":        "4A",
        "recommendation":   "...",
    })
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from backend.monitoring import get_logger
from backend.pipeline.orchestrator import PipelineContext, PipelineStage

log = get_logger(__name__)

REPORT_OUTPUT_DIR = Path(os.getenv("REPORT_OUTPUT_DIR", "./results/reports"))
PLATFORM_VERSION  = os.getenv("PLATFORM_VERSION", "2.0.0")


# ---------------------------------------------------------------------------
# Lung-RADS classification helper
# ---------------------------------------------------------------------------
def _lung_rads(malignant_prob: float, diameter_mm: Optional[float]) -> tuple[str, str]:
    """
    Simplified Lung-RADS 2022 category and recommendation.
    Not a substitute for a validated clinical decision tool.
    """
    diam = diameter_mm or 0.0
    if diam < 6 or malignant_prob < 0.15:
        cat, rec = "1", "Continue annual LDCT screening."
    elif diam < 8 or malignant_prob < 0.35:
        cat, rec = "2", "Continue annual LDCT screening."
    elif diam < 15 or malignant_prob < 0.55:
        cat, rec = "3", "LDCT follow-up in 6 months."
    elif diam < 20 or malignant_prob < 0.75:
        cat, rec = "4A", "LDCT follow-up in 3 months or PET-CT."
    elif malignant_prob < 0.90:
        cat, rec = "4B", "Tissue sampling recommended."
    else:
        cat, rec = "4X", "High suspicion — multidisciplinary team review."
    return cat, rec


class ReportStage:
    """Generates the final structured clinical report from all stage outputs."""

    def run(self, ctx: PipelineContext) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()

        qa_out   = ctx.get(PipelineStage.DICOM_QA) or {}
        prep_out = ctx.get(PipelineStage.PREPROCESSING) or {}
        det_out  = ctx.get(PipelineStage.DETECTION) or {}
        cls_out  = ctx.get(PipelineStage.CLASSIFICATION) or {}
        rad_out  = ctx.get(PipelineStage.RADIOMICS) or {}
        ens_out  = ctx.get(PipelineStage.ENSEMBLE) or {}
        unc_out  = ctx.get(PipelineStage.UNCERTAINTY) or {}
        xai_out  = ctx.get(PipelineStage.XAI) or {}

        # Derive largest nodule diameter
        candidates = det_out.get("candidates", [])
        largest_nodule_mm = (
            max((c["diameter_mm"] for c in candidates), default=None)
            if candidates else None
        )

        mal_prob = ens_out.get("malignant_prob", cls_out.get("consensus_prob", 0.5))
        lung_rads_cat, recommendation = _lung_rads(mal_prob, largest_nodule_mm)

        # Top-10 radiomics features
        top_radiomics = self._top_radiomics(rad_out)

        # Uncertainty summary
        entropy = unc_out.get("predictive_entropy", unc_out.get("entropy"))
        uncertainty_level = (
            "low" if (entropy or 0) < 0.3
            else "medium" if (entropy or 0) < 0.7
            else "high"
        )

        report: Dict[str, Any] = {
            "report_version": "2.0",
            "platform_version": PLATFORM_VERSION,
            "case_id": ctx.case_id,
            "generated_at": now,
            "elapsed_s": round(ctx.elapsed_s, 3),
            "pipeline_errors": ctx.errors,
            "pipeline_warnings": ctx.warnings,

            "sections": {
                "1_qa": {
                    "qa_passed": qa_out.get("passed", False),
                    "qa_checks": qa_out.get("checks", {}),
                    "qa_score": qa_out.get("score"),
                },
                "2_ct_findings": {
                    "nodule_count": det_out.get("nodule_count", 0),
                    "largest_nodule_mm": largest_nodule_mm,
                    "candidates": candidates[:5],   # top 5 only
                    "volume_shape": prep_out.get("shape"),
                    "spacing_mm": prep_out.get("spacing"),
                },
                "3_classification": {
                    model: {
                        "malignant_prob": res.get("malignant_prob"),
                        "label": "malignant" if res.get("malignant_prob", 0) > 0.5 else "benign",
                    }
                    for model, res in cls_out.items()
                    if isinstance(res, dict) and "malignant_prob" in res
                },
                "4_ensemble": {
                    "strategy":        ens_out.get("strategy"),
                    "malignant_prob":  ens_out.get("malignant_prob"),
                    "label":           ens_out.get("label"),
                    "confidence_band": ens_out.get("confidence_band"),
                },
                "5_uncertainty": {
                    "predictive_entropy": entropy,
                    "uncertainty_level":  uncertainty_level,
                    "mc_samples":         unc_out.get("n_samples"),
                    "std_malignant":      unc_out.get("std_malignant"),
                },
                "6_radiomics": {
                    "top_features": top_radiomics,
                    "total_features": len([
                        k for k, v in rad_out.items()
                        if isinstance(v, (int, float))
                    ]),
                },
                "7_xai": xai_out.get("summary", {}),
            },

            "lung_rads": lung_rads_cat,
            "recommendation": recommendation,
        }

        # Persist to disk
        self._save_report(ctx.case_id, report)
        log.info(
            "report_generated",
            case_id=ctx.case_id,
            lung_rads=lung_rads_cat,
            recommendation=recommendation,
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _top_radiomics(rad_out: Dict) -> list:
        features = [
            {"feature": k, "value": float(v)}
            for k, v in rad_out.items()
            if isinstance(v, (int, float)) and not str(k).startswith("diagnostics")
        ]
        return sorted(features, key=lambda f: abs(f["value"]), reverse=True)[:10]

    @staticmethod
    def _save_report(case_id: str, report: Dict) -> None:
        try:
            REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = REPORT_OUTPUT_DIR / f"{case_id}_report.json"
            out_path.write_text(json.dumps(report, indent=2, default=str))
            log.info("report_saved", path=str(out_path))
        except Exception as exc:
            log.warning("report_save_failed", error=str(exc))
