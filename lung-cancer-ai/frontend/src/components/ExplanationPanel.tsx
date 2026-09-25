import React, { useState } from 'react';

interface ExplanationPanelProps {
  noduleData: {
    id: string;
    location: string;
    slice: number;
    diameter_mm: number;
    volume_mm3: number;
    attenuation: string;
    margin: string;
    sphericity: number;
    spiculation_index: number;
    probability: number;
    ci_lower: number;
    ci_upper: number;
    uncertainty_level: string;
    disagreement_entropy: number;
    models: Array<{ name: string; prob: number; calibrated: number; uncertainty: string }>;
  };
  onClose?: () => void;
}

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({ noduleData, onClose }) => {
  const [activeTab, setActiveTab] = useState<'summary' | 'attribution' | 'ensemble'>('summary');
  const [method, setMethod] = useState<'cel_gradcam' | 'integrated_gradients' | 'attention_rollout'>('cel_gradcam');

  return (
    <div style={{
      backgroundColor: '#0d131f',
      border: '1px solid #1e293b',
      borderRadius: '8px',
      padding: '16px',
      color: '#e2e8f0',
      fontFamily: 'Inter, system-ui, sans-serif',
      boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1e293b', paddingBottom: '12px', marginBottom: '14px' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#38bdf8' }} />
            3D Explainable Decision Support — Nodule #{noduleData.id.slice(-4)}
          </h3>
          <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#94a3b8' }}>
            Multi-angle volumetric characterization & contrastive model attribution
          </p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              fontSize: '18px',
              padding: '4px 8px'
            }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Navigation Tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '14px', borderBottom: '1px solid #1e293b' }}>
        {(['summary', 'attribution', 'ensemble'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid #38bdf8' : '2px solid transparent',
              color: activeTab === tab ? '#38bdf8' : '#64748b',
              padding: '6px 12px',
              fontSize: '13px',
              fontWeight: 500,
              cursor: 'pointer',
              textTransform: 'capitalize'
            }}
          >
            {tab === 'summary' ? 'Clinical Overview' : tab === 'attribution' ? 'Feature Attribution' : 'Model Agreement'}
          </button>
        ))}
      </div>

      {/* Tab 1: Clinical Overview */}
      {activeTab === 'summary' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {/* WHERE */}
            <div style={{ backgroundColor: '#131d2e', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 600 }}>Where (Localization)</span>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#f1f5f9', marginTop: '4px' }}>{noduleData.location}</div>
              <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Axial Slice: #{noduleData.slice} • Mid-thoracic</div>
            </div>

            {/* WHAT & HOW LARGE */}
            <div style={{ backgroundColor: '#131d2e', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 600 }}>How Large (Morphometry)</span>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#f1f5f9', marginTop: '4px' }}>
                {noduleData.diameter_mm.toFixed(1)} mm max • {noduleData.volume_mm3.toFixed(0)} mm³
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Sphericity: {noduleData.sphericity.toFixed(2)}</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
            {/* WHAT CHARACTERISTICS */}
            <div style={{ backgroundColor: '#131d2e', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 600 }}>Characteristics (Phenotype)</span>
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#f1f5f9', marginTop: '4px', textTransform: 'capitalize' }}>
                {noduleData.attenuation.replace('_', '-')} • {noduleData.margin} margin
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>Spiculation Index: {noduleData.spiculation_index.toFixed(2)}</div>
            </div>

            {/* HOW CERTAIN & EXPLANATION RELIABILITY */}
            <div style={{ backgroundColor: '#131d2e', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 600 }}>Calibrated Risk & Uncertainty</span>
                <span style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: '4px',
                  backgroundColor: (noduleData.disagreement_entropy > 0.4 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(56, 189, 248, 0.2)'),
                  color: (noduleData.disagreement_entropy > 0.4 ? '#f87171' : '#38bdf8'),
                  border: `1px solid ${noduleData.disagreement_entropy > 0.4 ? '#ef4444' : '#0284c7'}`
                }}>
                  Reliability: {noduleData.disagreement_entropy > 0.4 ? 'LOW' : noduleData.disagreement_entropy > 0.25 ? 'MODERATE' : 'HIGH'}
                </span>
              </div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: noduleData.probability > 0.6 ? '#f87171' : '#34d399', marginTop: '4px' }}>
                {(noduleData.probability * 100).toFixed(1)}% Malignancy Risk
              </div>
              <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                95% CI: [{(noduleData.ci_lower * 100).toFixed(0)}% - {(noduleData.ci_upper * 100).toFixed(0)}%] • {noduleData.uncertainty_level} Uncertainty
              </div>
            </div>
          </div>

          {/* Research Contribution 1: Mandatory Low Reliability Abstention Warning */}
          {noduleData.disagreement_entropy > 0.4 && (
            <div style={{
              backgroundColor: '#450a0a',
              border: '1px solid #dc2626',
              borderRadius: '6px',
              padding: '8px 12px',
              fontSize: '11px',
              color: '#fecaca',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <span style={{ fontSize: '14px' }}>⚠️</span>
              <div>
                <strong>REVIEW REQUIRED (Explanation Reliability: LOW):</strong> Attribution maps from Grad-CAM, Integrated Gradients, and perturbations diverge. Algorithmic outputs are non-deterministic for this lesion.
              </div>
            </div>
          )}

          <div style={{
            fontSize: '11px',
            color: '#64748b',
            backgroundColor: '#090d16',
            padding: '8px 12px',
            borderRadius: '4px',
            borderLeft: '3px solid #38bdf8',
            marginTop: '4px'
          }}>
            <strong>Radiological Note:</strong> Morphological phenotype is characterized according to Fleischner Society criteria. Histopathological subtype cannot and should not be inferred from CT imaging alone.
          </div>
        </div>
      )}

      {/* Tab 2: Feature Attribution */}
      {activeTab === 'attribution' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: '#94a3b8' }}>Attribution Method:</span>
            {(['cel_gradcam', 'integrated_gradients', 'attention_rollout'] as const).map(m => (
              <button
                key={m}
                onClick={() => setMethod(m)}
                style={{
                  backgroundColor: method === m ? '#1e293b' : 'transparent',
                  color: method === m ? '#38bdf8' : '#64748b',
                  border: method === m ? '1px solid #38bdf8' : '1px solid #1e293b',
                  borderRadius: '4px',
                  padding: '4px 8px',
                  fontSize: '11px',
                  cursor: 'pointer'
                }}
              >
                {m === 'cel_gradcam' ? 'CEL Grad-CAM 3D' : m === 'integrated_gradients' ? 'Integrated Gradients' : 'Attention Rollout'}
              </button>
            ))}
          </div>

          {/* Attribution Visual Representation */}
          <div style={{
            position: 'relative',
            height: '180px',
            backgroundColor: '#050811',
            borderRadius: '6px',
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid #1e293b'
          }}>
            {/* Background CT Patch Simulation */}
            <div style={{
              width: '120px',
              height: '120px',
              borderRadius: '50%',
              backgroundColor: '#1e293b',
              boxShadow: 'inset 0 0 20px #0f172a',
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {/* Heatmap overlay */}
              <div style={{
                width: '60px',
                height: '60px',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(239,68,68,0.85) 0%, rgba(245,158,11,0.6) 45%, rgba(59,130,246,0) 80%)',
                filter: 'blur(3px)'
              }} />
            </div>

            {/* Mandatory Attribution Disclaimer Banner */}
            <div style={{
              position: 'absolute',
              bottom: '8px',
              backgroundColor: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid #334155',
              padding: '3px 8px',
              borderRadius: '4px',
              fontSize: '10px',
              fontWeight: 600,
              letterSpacing: '0.06em',
              color: '#f59e0b'
            }}>
              MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE
            </div>
          </div>

          <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: '1.4' }}>
            Contrastive Explainability Learning (CEL) suppresses spurious adjacent vascular bifurcations and concentrates high activation (red) on the irregular, non-calcified nodule margin.
          </div>
        </div>
      )}

      {/* Tab 3: Model Agreement */}
      {activeTab === 'ensemble' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: '#94a3b8' }}>
            <span>Architecture Breakdown</span>
            <span>Disagreement Entropy: {noduleData.disagreement_entropy.toFixed(2)}</span>
          </div>

          {noduleData.models.map((mod, idx) => (
            <div key={idx} style={{ backgroundColor: '#131d2e', padding: '8px 12px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ fontWeight: 600, color: '#f1f5f9' }}>{mod.name}</span>
                <span style={{ color: mod.calibrated > 0.5 ? '#f87171' : '#34d399', fontWeight: 600 }}>
                  {(mod.calibrated * 100).toFixed(1)}% P(Malignant)
                </span>
              </div>
              <div style={{ height: '6px', backgroundColor: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${mod.calibrated * 100}%`,
                  backgroundColor: mod.calibrated > 0.5 ? '#ef4444' : '#10b981',
                  borderRadius: '3px'
                }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', marginTop: '4px' }}>
                <span>Raw: {(mod.prob * 100).toFixed(1)}%</span>
                <span>Uncertainty: {mod.uncertainty}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
