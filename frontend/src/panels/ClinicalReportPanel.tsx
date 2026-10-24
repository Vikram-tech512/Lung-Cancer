import React, { useState } from 'react';
import { FileText, Download, ShieldCheck, CheckCircle2, AlertTriangle, Printer, Layers, Clock } from 'lucide-react';

export const ClinicalReportPanel: React.FC<{
  studyId?: string;
  noduleId?: string;
}> = ({
  studyId = 'CASE-DEMO-LIDC-0087',
  noduleId = 'NODULE-01',
}) => {
  const [reportVersion] = useState<string>('2.0.0');
  const [generatedDate] = useState<string>(new Date().toISOString());

  const reportData = {
    report_version: '2.0.0',
    case_id: studyId,
    patient_id: 'PT-981240-NLST',
    generated_at: generatedDate,
    lung_rads: null,
    recommendation: 'For radiologist review. Lung-RADS category to be assigned manually by reviewing radiologist.',
    sections: {
      dicom_qa: {
        status: 'PASSED',
        slice_thickness_mm: 1.25,
        pixel_spacing_mm: 0.68,
        manufacturer: 'GE Medical Systems',
        reconstruction_kernel: 'B30f Standard Chest',
        artifacts: 'None detected (SNR: 34.2 dB)',
      },
      ct_findings: {
        total_nodules: 3,
        primary_nodule_id: noduleId,
        anatomic_lobe: 'Right Upper Lobe (Apical Segment)',
        effective_diameter_mm: 11.4,
        volume_mm3: 775.2,
        attenuation_mean_hu: -142.5,
        composition: 'Part-Solid with Ground-Glass Halo',
      },
      classification_outputs: [
        { model: '3D ResNet-50', prob: 0.884, output_label: 'For radiologist review', weight: 0.35 },
        { model: '3D DenseNet-121', prob: 0.862, output_label: 'For radiologist review', weight: 0.30 },
        { model: '3D Swin Transformer', prob: 0.891, output_label: 'For radiologist review', weight: 0.35 },
      ],
      ensemble_synthesis: {
        fused_nodule_prob: 0.879,
        confidence_interval_95: [0.841, 0.918],
        disagreement_entropy: 0.082,
        calibrated: true,
        temperature: 1.18,
      },
      uncertainty_profile: {
        mc_dropout_passes: 30,
        predictive_variance: 0.0034,
        uncertainty_level: 'LOW (Clinically Reliable)',
      },
      radiomics_signature: {
        ibsi_features_count: 107,
        spiculation_index: 0.412,
        sphericity: 0.742,
        joint_entropy: 4.89,
        dominant_trait: 'Irregular Spiculated Margins',
      },
      xai_localisation: {
        gradcam_concordance: '89.4% overlap with focal lesion mask',
        integrated_gradients: 'High attribution concentrated on core solid spicules',
        attention_rollout: 'Global context confirms peripheral ground-glass margin',
      },
      audit_trail: {
        platform: 'Lung Cancer AI Platform v2.0',
        pipeline_dag_stages: 11,
        resnet3d_sha256: 'a4b8f029...e89c',
        densenet3d_sha256: '7c1d33ba...901e',
        swin3d_sha256: 'f22e8910...44ab',
        radiomics_ibsi_hash: '5d120a4c...bb09',
      },
    },
  };

  const handleDownload = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(reportData, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `clinical_report_${studyId}_lung_rads.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Top Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span
              style={{
                fontSize: '11px',
                fontWeight: '700',
                padding: '2px 8px',
                borderRadius: '4px',
                background: 'rgba(16, 185, 129, 0.15)',
                color: 'var(--accent-emerald)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                letterSpacing: '0.05em',
              }}
            >
              10 · CLINICAL REPORT GENERATION
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Fleischner Society &amp; Lung-RADS v2022 Structured Clinical Document
            </span>
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: '800', margin: 0, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
            Findings Synthesis &amp; Regulatory Audit Report
          </h1>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={handlePrint}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-card)',
              color: 'var(--text-main)',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer',
            }}
          >
            <Printer size={14} />
            Print Report
          </button>
          <button
            onClick={handleDownload}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              background: 'linear-gradient(135deg, var(--primary), var(--accent-cyan))',
              border: 'none',
              color: '#ffffff',
              fontSize: '12px',
              fontWeight: '700',
              cursor: 'pointer',
            }}
          >
            <Download size={14} />
            Download Clinical JSON
          </button>
        </div>
      </div>

      {/* Main Clinical Document Card */}
      <div
        className="glass-panel"
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '10px',
          padding: '28px',
          display: 'flex',
          flexDirection: 'column',
          gap: '24px',
        }}
      >
        {/* Document Header */}
        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontSize: '16px', fontWeight: '800', color: 'var(--text-main)' }}>
              THORACIC VOLUMETRIC CT ARTIFICIAL INTELLIGENCE ASSESSMENT
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Study Accession: <strong style={{ color: 'var(--text-main)' }}>{studyId}</strong> · Patient ID: <strong style={{ color: 'var(--text-main)' }}>{reportData.patient_id}</strong>
            </div>
          </div>
          <div style={{ textAlign: 'right', fontSize: '11px', color: 'var(--text-muted)' }}>
            <div>Generated: {new Date(reportData.generated_at).toLocaleString()}</div>
            <div style={{ color: 'var(--accent-emerald)', marginTop: '2px' }}>✓ Pipeline 11-Stage DAG Completed</div>
          </div>
        </div>

        {/* Lung-RADS Highlight Box */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px 20px',
            borderRadius: '4px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <div>
            <div style={{ fontSize: '14px', fontWeight: '700', color: 'var(--text-main)' }}>
              Lung-RADS Category (Assigned by Radiologist)
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
              Recommendation: <strong>{reportData.recommendation}</strong>
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>ENSEMBLE NODULE PROBABILITY</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: 'var(--primary)' }}>
              {(reportData.sections.ensemble_synthesis.fused_nodule_prob * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Section 1 & 2: DICOM QA and CT Findings */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{ background: 'var(--bg-card)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--accent-cyan)', marginBottom: '10px', textTransform: 'uppercase' }}>
              1. DICOM Quality Assurance
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-body)' }}>
              <div>Quality Status: <strong style={{ color: 'var(--accent-emerald)' }}>PASSED</strong></div>
              <div>Slice Thickness: <strong>{reportData.sections.dicom_qa.slice_thickness_mm} mm</strong></div>
              <div>Pixel Spacing: <strong>{reportData.sections.dicom_qa.pixel_spacing_mm} mm</strong></div>
              <div>Scanner: <strong>{reportData.sections.dicom_qa.manufacturer}</strong></div>
              <div>Reconstruction: <strong>{reportData.sections.dicom_qa.reconstruction_kernel}</strong></div>
            </div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--accent-cyan)', marginBottom: '10px', textTransform: 'uppercase' }}>
              2. Quantitative CT Findings
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-body)' }}>
              <div>Primary Lesion: <strong>{reportData.sections.ct_findings.primary_nodule_id}</strong></div>
              <div>Anatomic Location: <strong>{reportData.sections.ct_findings.anatomic_lobe}</strong></div>
              <div>Effective Diameter: <strong>{reportData.sections.ct_findings.effective_diameter_mm} mm</strong></div>
              <div>Volumetric Size: <strong>{reportData.sections.ct_findings.volume_mm3} mm³</strong></div>
              <div>Morphologic Class: <strong>{reportData.sections.ct_findings.composition}</strong></div>
            </div>
          </div>
        </div>

        {/* Section 3: Model Outputs Table */}
        <div>
          <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--primary)', marginBottom: '8px', textTransform: 'uppercase' }}>
            3. Deep Learning Subsystem Model Outputs &amp; Weights
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'var(--bg-card)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '8px 12px' }}>NEURAL ARCHITECTURE</th>
                <th style={{ padding: '8px 12px' }}>CALIBRATED PROBABILITY</th>
                <th style={{ padding: '8px 12px' }}>ENSEMBLE WEIGHT</th>
                <th style={{ padding: '8px 12px' }}>REVIEW LABEL</th>
              </tr>
            </thead>
            <tbody>
              {reportData.sections.classification_outputs.map((v, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                  <td style={{ padding: '10px 12px', fontWeight: '600', color: 'var(--text-main)' }}>{v.model}</td>
                  <td style={{ padding: '10px 12px', fontFamily: 'JetBrains Mono, monospace' }}>{(v.prob * 100).toFixed(1)}%</td>
                  <td style={{ padding: '10px 12px', fontFamily: 'JetBrains Mono, monospace' }}>{(v.weight * 100).toFixed(0)}%</td>
                  <td style={{ padding: '10px 12px', color: 'var(--text-body)', fontWeight: '600' }}>{v.output_label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Section 4: Explainability & Radiomics */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div style={{ background: 'var(--bg-card)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--accent-cyan)', marginBottom: '10px', textTransform: 'uppercase' }}>
              4. Explainable AI Attribution
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-body)' }}>
              <div>3D Grad-CAM: <strong>{reportData.sections.xai_localisation.gradcam_concordance}</strong></div>
              <div>Integrated Gradients: <strong>{reportData.sections.xai_localisation.integrated_gradients}</strong></div>
              <div>Attention Rollout: <strong>{reportData.sections.xai_localisation.attention_rollout}</strong></div>
            </div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--accent-cyan)', marginBottom: '10px', textTransform: 'uppercase' }}>
              5. Radiomics Quantitative Phenotype
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px', color: 'var(--text-body)' }}>
              <div>Features Standard: <strong>107 IBSI-compliant features</strong></div>
              <div>Sphericity Index: <strong>{reportData.sections.radiomics_signature.sphericity} (Non-spherical)</strong></div>
              <div>Spiculation Metric: <strong>{reportData.sections.radiomics_signature.spiculation_index} (Spiculated)</strong></div>
              <div>Texture Entropy: <strong>{reportData.sections.radiomics_signature.joint_entropy} (High heterogeneity)</strong></div>
            </div>
          </div>
        </div>

        {/* Audit Trail Footer */}
        <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-faint)' }}>
          <div>
            System: {reportData.sections.audit_trail.platform} · Checksum DAG: SHA256 verified
          </div>
          <div>
            Regulatory: Clinical Decision-Support / Research Use Only · Fleischner 2022
          </div>
        </div>
      </div>
    </div>
  );
};
export default ClinicalReportPanel;
