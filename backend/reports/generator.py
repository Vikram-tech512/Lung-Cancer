"""
backend/reports/generator.py
============================
Automated clinical PDF report generator.
Compiles study metadata, DICOM QC audit, 3D lesion morphometry,
multimodal ensemble verdicts, and regulatory disclaimers.
"""

from __future__ import annotations
import os
import datetime
from pathlib import Path
from typing import Dict, Any

class ReportGenerator:
    def __init__(self, output_dir: str = "storage_bucket/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_pdf(self, study_data: Dict[str, Any], verdict_data: Dict[str, Any], morph_data: Dict[str, Any]) -> str:
        study_id = study_data.get("study_id", "UNKNOWN")
        report_filename = f"Report_{study_id}_{int(datetime.datetime.utcnow().timestamp())}.pdf"
        output_path = self.output_dir / report_filename

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

            doc = SimpleDocTemplate(str(output_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()

            # Custom styles
            title_style = ParagraphStyle(
                'ReportTitle',
                parent=styles['Heading1'],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor('#0f172a')
            )
            sub_style = ParagraphStyle(
                'SubTitle',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                textColor=colors.HexColor('#64748b')
            )
            section_heading = ParagraphStyle(
                'SectionHeading',
                parent=styles['Heading2'],
                fontSize=13,
                leading=17,
                textColor=colors.HexColor('#0369a1'),
                spaceBefore=12,
                spaceAfter=6
            )
            disclaimer_style = ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontSize=8,
                leading=11,
                textColor=colors.HexColor('#be123c'),
                backColor=colors.HexColor('#ffe4e6'),
                borderPadding=6
            )

            story = []

            # Header
            story.append(Paragraph("LUNG CANCER AI RESEARCH PLATFORM", title_style))
            story.append(Paragraph("Multimodal Thoracic CT Volumetric Analysis & Nodule Assessment", sub_style))
            story.append(Spacer(1, 10))

            # Regulatory Disclaimer Box
            story.append(Paragraph(
                "<b>MANDATORY NOTICE: RESEARCH USE ONLY — NOT AN APPROVED CLINICAL DIAGNOSTIC DEVICE.</b> "
                "This report provides algorithmic predictions and quantitative morphometry for decision-support and academic study. "
                "Definitive diagnosis requires qualified board-certified physician consultation.",
                disclaimer_style
            ))
            story.append(Spacer(1, 14))

            # Study & Patient Info Table
            study_table_data = [
                ["Study Identifier", study_data.get("study_id", "N/A"), "Patient ID", study_data.get("patient_id", "DE-IDENTIFIED")],
                ["Modality", study_data.get("modality", "CT"), "Slice Count", str(study_data.get("slice_count", 0))],
                ["Slice Thickness", f"{study_data.get('slice_thickness', 1.0)} mm", "QC Audit Status", study_data.get("qc_status", "PASS")],
            ]
            t1 = Table(study_table_data, colWidths=[120, 150, 120, 150])
            t1.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ]))
            story.append(Paragraph("1. Study & Series Verification", section_heading))
            story.append(t1)
            story.append(Spacer(1, 10))

            # Consensus Verdict
            story.append(Paragraph("2. Multimodal Consensus Verdict", section_heading))
            verdict_val = verdict_data.get("ensemble_verdict", "INDETERMINATE")
            prob_val = verdict_data.get("ensemble_probability", 0.5) * 100
            
            verdict_table_data = [
                ["Consensus Verdict", verdict_val],
                ["Malignancy Probability", f"{prob_val:.1f}% (95% CI: [{verdict_data.get('ci_lower', 0)*100:.1f}% - {verdict_data.get('ci_upper', 1)*100:.1f}%])"],
                ["Disagreement Entropy", f"{verdict_data.get('disagreement_entropy', 0.0):.3f}"],
                ["Active Models Evaluated", "3D ResNet-50, 3D Swin Transformer, HybridViT, Radiomics XGBoost"]
            ]
            t2 = Table(verdict_table_data, colWidths=[160, 380])
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#94a3b8')),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ]))
            story.append(t2)
            story.append(Spacer(1, 10))

            # 3D Morphometry
            story.append(Paragraph("3. Quantitative 3D Nodule Morphometry", section_heading))
            morph_table_data = [
                ["Volume", f"{morph_data.get('volume_mm3', 0.0):.1f} mm³", "Effective Diameter", f"{morph_data.get('effective_diameter_mm', 0.0):.1f} mm"],
                ["Sphericity Index", f"{morph_data.get('sphericity', 0.0):.2f}", "Spiculation Index", f"{morph_data.get('spiculation_index', 0.0):.2f}"],
                ["Mean HU", f"{morph_data.get('mean_hu', 0.0):.1f} HU", "HU Range", f"[{morph_data.get('min_hu', 0):.0f} to {morph_data.get('max_hu', 0):.0f}] HU"],
            ]
            t3 = Table(morph_table_data, colWidths=[120, 150, 120, 150])
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ]))
            story.append(t3)
            story.append(Spacer(1, 14))

            # Footer
            story.append(Paragraph(
                f"Generated on {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC | Engine Version: 2.0.0-production",
                sub_style
            ))

            doc.build(story)
        except ImportError:
            # Simple text/HTML fallback if reportlab is not yet installed in host environment
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"LUNG CANCER AI REPORT\nStudy: {study_id}\nVerdict: {verdict_data.get('ensemble_verdict')}\n")

        return str(output_path.as_posix())

report_generator = ReportGenerator()
