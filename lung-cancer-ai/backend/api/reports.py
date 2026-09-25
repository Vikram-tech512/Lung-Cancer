"""
backend/api/reports.py
======================
Clinical PDF report listing and file download endpoints.
"""

from __future__ import annotations
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.database.database import get_db
from backend.database.models import Report

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("")
def list_reports(db: Session = Depends(get_db)):
    reports = db.query(Report).order_by(Report.created_at.desc()).all()
    if not reports:
        return [
            {
                "report_id": "REP-DEMO-01",
                "study_id": "CASE-DEMO-LIDC-0087",
                "summary": "Malignant nodule candidate (Lung-RADS 4B) in RUL. Consensus: 84.9%.",
                "created_at": "2026-09-22 13:30:00",
                "download_url": "/api/reports/download/demo"
            }
        ]
    return [
        {
            "report_id": r.id,
            "study_id": r.study_id,
            "summary": r.summary_text,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "download_url": f"/api/reports/download/{r.id}"
        }
        for r in reports
    ]

@router.get("/download/{report_id}")
def download_report(report_id: str, db: Session = Depends(get_db)):
    # Serve demo or DB report
    if report_id == "demo":
        from backend.reports.generator import report_generator
        path = report_generator.generate_pdf(
            study_data={"study_id": "CASE-DEMO-LIDC-0087", "modality": "CT", "slice_count": 196, "slice_thickness": 1.25, "qc_status": "PASS"},
            verdict_data={"ensemble_verdict": "MALIGNANT", "ensemble_probability": 0.849, "ci_lower": 0.792, "ci_upper": 0.906, "disagreement_entropy": 0.38},
            morph_data={"volume_mm3": 1696.5, "effective_diameter_mm": 14.8, "sphericity": 0.74, "spiculation_index": 0.42, "mean_hu": 38.4, "min_hu": -110.0, "max_hu": 92.0}
        )
        return FileResponse(path, filename="Lung_Cancer_AI_Report_DEMO.pdf", media_type="application/pdf")

    report = db.query(Report).filter(Report.id == report_id).first()
    if not report or not os.path.exists(report.report_pdf_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(report.report_pdf_path, filename=f"Report_{report.study_id}.pdf", media_type="application/pdf")
