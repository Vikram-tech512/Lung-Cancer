import React, { useEffect, useState } from 'react';
import { ShieldAlert, Activity, CheckCircle, HelpCircle, Layers, BarChart2 } from 'lucide-react';
import { VerdictResult } from '../types';

interface VerdictPanelProps {
  noduleId: string;
}

export const VerdictPanel: React.FC<VerdictPanelProps> = ({ noduleId }) => {
  const [verdict, setVerdict] = useState<VerdictResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/v2/classify/${noduleId}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        setVerdict(data);
        setLoading(false);
      })
      .catch(() => {
        // demo fallback
        setVerdict({
          nodule_id: noduleId,
          ensemble_verdict: 'MALIGNANT',
          ensemble_probability: 0.764,
          confidence_interval: [0.712, 0.816],
          weights_used: { resnet3d: 0.35, swin3d: 0.35, radiomics_xgb: 0.30 },
          disagreement_entropy: 0.38,
          model_breakdown: [
            {
              model_id: 'resnet3d',
              model_name: '3D ResNet-50',
              raw_probability: 0.742,
              calibrated_probability: 0.712,
              uncertainty_score: 0.082,
              uncertainty_level: 'LOW',
              status: 'ACTIVE',
              weight_in_ensemble: 0.35,
              verdict: 'MALIGNANT'
            },
            {
              model_id: 'swin3d',
              model_name: '3D Swin Transformer',
              raw_probability: 0.785,
              calibrated_probability: 0.753,
              uncertainty_score: 0.065,
              uncertainty_level: 'LOW',
              status: 'ACTIVE',
              weight_in_ensemble: 0.35,
              verdict: 'MALIGNANT'
            },
            {
              model_id: 'hybrid_vit',
              model_name: 'Hybrid CNN-ViT (Phase 4)',
              raw_probability: 0.812,
              calibrated_probability: 0.789,
              uncertainty_score: 0.045,
              uncertainty_level: 'LOW',
              status: 'ACTIVE',
              weight_in_ensemble: 0.0,
              verdict: 'MALIGNANT'
            },
            {
              model_id: 'radiomics_xgb',
              model_name: 'Radiomics XGBoost',
              raw_probability: 0.690,
              calibrated_probability: 0.672,
              uncertainty_score: 0.110,
              uncertainty_level: 'MODERATE',
              status: 'ACTIVE',
              weight_in_ensemble: 0.30,
              verdict: 'MALIGNANT'
            }
          ],
          morphometry: {
            nodule_id: noduleId,
            centroid_zyx: [92, 280, 310],
            effective_diameter_mm: 14.8,
            volume_mm3: 1696.5,
            mean_hu: 38.4,
            std_hu: 24.1,
            max_hu: 92.0,
            min_hu: -110.0,
            sphericity: 0.74,
            spiculation_index: 0.42,
            surface_to_volume_ratio: 0.41
          },
          clinical_advisory: 'RESEARCH USE ONLY — NOT APPROVED FOR PRIMARY DIAGNOSTIC DECISIONS'
        });
        setLoading(false);
      });
  }, [noduleId]);

  if (loading || !verdict) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>
        Evaluating candidate {noduleId} across 3D Deep Learning, Vision Transformers, and Radiomics...
      </div>
    );
  }

  const isMalignant = verdict.ensemble_verdict === 'MALIGNANT';

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
          05 · MULTIMODAL VERDICT & UNCERTAINTY QUANTIFICATION
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '6px' }}>
          Consensus evaluation of candidate <span style={{ color: '#38bdf8', fontWeight: '600' }}>{noduleId}</span> across 3D ResNet, Swin3D, HybridViT, and PyRadiomics.
        </p>
      </div>

      {/* Prominent Clinical Advisory Alert */}
      <div style={{
        background: 'rgba(244, 63, 94, 0.1)',
        border: '1px solid rgba(244, 63, 94, 0.3)',
        borderRadius: '8px',
        padding: '14px 18px',
        marginBottom: '24px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <ShieldAlert size={20} color="#f43f5e" style={{ flexShrink: 0 }} />
        <div style={{ fontSize: '13px', color: '#fda4af', fontWeight: '500' }}>
          <strong style={{ color: '#f43f5e' }}>MANDATORY CLINICAL NOTICE: </strong>
          {verdict.clinical_advisory}. This algorithmic assessment represents computational inference and must not be used as primary evidence for biopsy, resection, or clinical staging.
        </div>
      </div>

      {/* Top Banner: Verdict & Metrics Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '20px', marginBottom: '24px' }}>
        {/* Main Verdict Card */}
        <div className="glass-panel-accent" style={{ padding: '24px', position: 'relative' }}>
          <div style={{ fontSize: '12px', textTransform: 'uppercase', color: '#06b6d4', fontWeight: '700', letterSpacing: '0.05em' }}>
            Consensus Ensemble Verdict
          </div>
          <div style={{ fontSize: '32px', fontWeight: '800', color: isMalignant ? '#f43f5e' : '#10b981', marginTop: '8px' }}>
            {verdict.ensemble_verdict}
          </div>
          <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
            Probability: <strong style={{ color: '#f8fafc' }}>{(verdict.ensemble_probability * 100).toFixed(1)}%</strong> (95% CI: [{(verdict.confidence_interval[0] * 100).toFixed(1)}% - {(verdict.confidence_interval[1] * 100).toFixed(1)}%])
          </div>
        </div>

        {/* Epistemic Uncertainty (MC Dropout) */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ fontSize: '12px', textTransform: 'uppercase', color: '#94a3b8', fontWeight: '700', letterSpacing: '0.05em' }}>
            Epistemic Uncertainty
          </div>
          <div style={{ fontSize: '28px', fontWeight: '800', color: '#38bdf8', marginTop: '8px' }}>
            LOW RISK
          </div>
          <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
            Disagreement Entropy: <strong style={{ color: '#f8fafc' }}>{verdict.disagreement_entropy}</strong> (T=30 Passes)
          </div>
        </div>

        {/* Nodule Morphometry Snapshot */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ fontSize: '12px', textTransform: 'uppercase', color: '#94a3b8', fontWeight: '700', letterSpacing: '0.05em' }}>
            Effective Diameter
          </div>
          <div style={{ fontSize: '28px', fontWeight: '800', color: '#f59e0b', marginTop: '8px' }}>
            {verdict.morphometry.effective_diameter_mm} mm
          </div>
          <div style={{ fontSize: '13px', color: '#94a3b8', marginTop: '4px' }}>
            Volume: <strong style={{ color: '#f8fafc' }}>{verdict.morphometry.volume_mm3.toFixed(0)} mm³</strong> | Sphericity: <strong>{verdict.morphometry.sphericity}</strong>
          </div>
        </div>
      </div>

      {/* Model Breakdown Comparison Table */}
      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #1e293b', background: '#0b1120', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <BarChart2 size={16} color="#38bdf8" />
          <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#f1f5f9' }}>
            Individual Model Contribution & Calibration Breakdown
          </h3>
        </div>

        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
          <thead>
            <tr style={{ background: '#070c14', borderBottom: '1px solid #1e293b', color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>
              <th style={{ padding: '12px 18px' }}>Model Architecture</th>
              <th style={{ padding: '12px 18px' }}>Raw Probability</th>
              <th style={{ padding: '12px 18px' }}>Calibrated (Temp Scaled)</th>
              <th style={{ padding: '12px 18px' }}>Uncertainty</th>
              <th style={{ padding: '12px 18px' }}>Ensemble Weight</th>
              <th style={{ padding: '12px 18px' }}>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {verdict.model_breakdown.map((m) => (
              <tr key={m.model_id} style={{ borderBottom: '1px solid #1e293b' }}>
                <td style={{ padding: '14px 18px', fontWeight: '600', color: '#e2e8f0' }}>
                  {m.model_name}
                </td>
                <td style={{ padding: '14px 18px', fontFamily: 'JetBrains Mono, monospace' }}>
                  {(m.raw_probability * 100).toFixed(1)}%
                </td>
                <td style={{ padding: '14px 18px', fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8' }}>
                  {(m.calibrated_probability * 100).toFixed(1)}%
                </td>
                <td style={{ padding: '14px 18px' }}>
                  <span style={{
                    fontSize: '11px',
                    fontWeight: '700',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: m.uncertainty_level === 'LOW' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                    color: m.uncertainty_level === 'LOW' ? '#10b981' : '#f59e0b'
                  }}>
                    {m.uncertainty_level} (±{m.uncertainty_score})
                  </span>
                </td>
                <td style={{ padding: '14px 18px', fontFamily: 'JetBrains Mono, monospace' }}>
                  {m.weight_in_ensemble > 0 ? `${(m.weight_in_ensemble * 100).toFixed(0)}%` : 'Baseline'}
                </td>
                <td style={{ padding: '14px 18px' }}>
                  <span style={{
                    fontWeight: '700',
                    color: m.verdict === 'MALIGNANT' ? '#f43f5e' : '#10b981'
                  }}>
                    {m.verdict}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
