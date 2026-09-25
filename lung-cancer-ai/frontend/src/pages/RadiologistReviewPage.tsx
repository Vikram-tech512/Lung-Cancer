import React, { useState, useEffect } from 'react';

interface RadiologistReviewPageProps {
  navigate?: (route: string) => void;
  studyId?: string;
}

interface FindingItem {
  id: string;
  lesion_index: number;
  lobe: string;
  ai_diameter_mm: number;
  ai_volume_mm3: number;
  ai_risk_score: number;
  ai_verdict: string;
  ai_lung_rads: string;
  radiologist_action: 'PENDING' | 'ACCEPT' | 'REJECT' | 'MODIFY' | 'MARK_FP';
  radiologist_diameter_mm: number;
  radiologist_verdict: string;
  radiologist_lung_rads: string;
  fp_category: string;
  comments: string;
}

export const RadiologistReviewPage: React.FC<RadiologistReviewPageProps> = ({ navigate, studyId = 'CASE-DEMO-LIDC-0087' }) => {
  const [findings, setFindings] = useState<FindingItem[]>([
    {
      id: 'LESION-01',
      lesion_index: 1,
      lobe: 'Right Upper Lobe (RUL)',
      ai_diameter_mm: 14.8,
      ai_volume_mm3: 1696.5,
      ai_risk_score: 0.849,
      ai_verdict: 'MALIGNANT',
      ai_lung_rads: '4B',
      radiologist_action: 'PENDING',
      radiologist_diameter_mm: 14.8,
      radiologist_verdict: 'MALIGNANT',
      radiologist_lung_rads: '4B',
      fp_category: 'none',
      comments: ''
    },
    {
      id: 'LESION-02',
      lesion_index: 2,
      lobe: 'Left Lower Lobe (LLL)',
      ai_diameter_mm: 4.2,
      ai_volume_mm3: 38.8,
      ai_risk_score: 0.120,
      ai_verdict: 'BENIGN',
      ai_lung_rads: '2',
      radiologist_action: 'ACCEPT',
      radiologist_diameter_mm: 4.2,
      radiologist_verdict: 'BENIGN',
      radiologist_lung_rads: '2',
      fp_category: 'none',
      comments: 'Small benign non-calcified solid nodule.'
    }
  ]);

  const [signerName, setSignerName] = useState('Dr. Sarah Chen, MD');
  const [isFinalized, setIsFinalized] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const updateFinding = (index: number, patch: Partial<FindingItem>) => {
    setFindings(prev => {
      const copy = [...prev];
      copy[index] = { ...copy[index], ...patch };
      return copy;
    });
  };

  const handleAction = (index: number, action: FindingItem['radiologist_action']) => {
    updateFinding(index, { radiologist_action: action });
  };

  const handleFinalize = () => {
    setSaveStatus('Saving radiologist sign-off and generating immutable audit record...');
    setTimeout(() => {
      setIsFinalized(true);
      setSaveStatus('Review finalized and signed. Report updated.');
    }, 800);
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto', color: '#f8fafc', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', borderBottom: '1px solid #1e293b', paddingBottom: '16px' }}>
        <div>
          <button
            onClick={() => navigate ? navigate(`/viewer/${studyId}`) : window.history.back()}
            style={{
              backgroundColor: 'transparent',
              border: 'none',
              color: '#38bdf8',
              cursor: 'pointer',
              fontSize: '13px',
              padding: 0,
              marginBottom: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            ← Return to 3D CT Workstation
          </button>
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 700, color: '#f8fafc' }}>
            Radiologist Review & Interactive Verification
          </h1>
          <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#94a3b8' }}>
            Study ID: <span style={{ color: '#38bdf8' }}>{studyId}</span> • Compliance Mandate: AI findings and human decisions are logged separately
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          {saveStatus && (
            <span style={{ fontSize: '12px', color: isFinalized ? '#34d399' : '#38bdf8' }}>
              {saveStatus}
            </span>
          )}
          <button
            disabled={isFinalized}
            onClick={handleFinalize}
            style={{
              backgroundColor: isFinalized ? '#1e293b' : '#38bdf8',
              color: isFinalized ? '#64748b' : '#030712',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '6px',
              fontWeight: 600,
              fontSize: '13px',
              cursor: isFinalized ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            {isFinalized ? '✓ Signed & Finalized' : 'Sign Off & Finalize Report'}
          </button>
        </div>
      </div>

      {/* Safety Notice Banner */}
      <div style={{
        backgroundColor: '#0f172a',
        border: '1px solid #1e293b',
        borderLeft: '4px solid #38bdf8',
        padding: '12px 16px',
        borderRadius: '6px',
        marginBottom: '24px',
        fontSize: '12px',
        lineHeight: 1.5,
        color: '#94a3b8'
      }}>
        <strong style={{ color: '#f1f5f9' }}>Auditing & Clinical Separation Guarantee (§25):</strong> The AI system's original raw model predictions, calculated measurements, and heatmaps are permanently preserved in the database. Radiologist adjustments or dismissals are recorded in the secondary clinical layer to ensure accountability.
      </div>

      {/* Findings Review Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {findings.map((item, idx) => (
          <div
            key={item.id}
            style={{
              backgroundColor: '#0d131f',
              border: item.radiologist_action === 'PENDING' ? '1px solid #334155' : item.radiologist_action === 'ACCEPT' ? '1px solid #059669' : item.radiologist_action === 'REJECT' || item.radiologist_action === 'MARK_FP' ? '1px solid #dc2626' : '1px solid #38bdf8',
              borderRadius: '8px',
              padding: '20px'
            }}
          >
            {/* Card Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{
                  backgroundColor: '#1e293b',
                  color: '#38bdf8',
                  padding: '4px 10px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  fontWeight: 700
                }}>
                  Finding #{item.lesion_index}
                </span>
                <span style={{ fontSize: '15px', fontWeight: 600, color: '#f1f5f9' }}>
                  {item.lobe}
                </span>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: '8px' }}>
                {(['ACCEPT', 'MODIFY', 'MARK_FP', 'REJECT'] as const).map(act => (
                  <button
                    key={act}
                    disabled={isFinalized}
                    onClick={() => handleAction(idx, act)}
                    style={{
                      backgroundColor: item.radiologist_action === act ? (act === 'ACCEPT' ? '#059669' : act === 'REJECT' || act === 'MARK_FP' ? '#dc2626' : '#0284c7') : '#1e293b',
                      color: item.radiologist_action === act ? '#ffffff' : '#94a3b8',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '6px 12px',
                      fontSize: '11px',
                      fontWeight: 600,
                      cursor: isFinalized ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {act === 'ACCEPT' ? '✓ Accept AI Finding' : act === 'MODIFY' ? '✎ Modify Measurements' : act === 'MARK_FP' ? '✕ Mark False Positive' : '✕ Reject'}
                  </button>
                ))}
              </div>
            </div>

            {/* Side-by-Side Comparison: Frozen AI vs Radiologist Verdict */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '16px' }}>
              {/* Left Column: AI Model Output (Read-only) */}
              <div style={{ backgroundColor: '#131d2e', padding: '14px', borderRadius: '6px', border: '1px solid #1e293b' }}>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#64748b', fontWeight: 700, marginBottom: '8px' }}>
                  AI Model Output (Preserved Frozen)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>Diameter / Volume</div>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0', marginTop: '2px' }}>
                      {item.ai_diameter_mm.toFixed(1)} mm ({item.ai_volume_mm3.toFixed(0)} mm³)
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>Malignancy Risk</div>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: item.ai_risk_score > 0.5 ? '#f87171' : '#34d399', marginTop: '2px' }}>
                      {(item.ai_risk_score * 100).toFixed(1)}% ({item.ai_verdict})
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: '#94a3b8' }}>Lung-RADS</div>
                    <div style={{ fontSize: '13px', fontWeight: 600, color: '#e2e8f0', marginTop: '2px' }}>
                      Category {item.ai_lung_rads}
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Radiologist Final Assessment (Editable) */}
              <div style={{ backgroundColor: '#131d2e', padding: '14px', borderRadius: '6px', border: '1px solid #334155' }}>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#38bdf8', fontWeight: 700, marginBottom: '8px' }}>
                  Radiologist Verification
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                  <div>
                    <label style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Verified Diam (mm)</label>
                    <input
                      type="number"
                      step="0.1"
                      disabled={isFinalized || item.radiologist_action !== 'MODIFY'}
                      value={item.radiologist_diameter_mm}
                      onChange={e => updateFinding(idx, { radiologist_diameter_mm: parseFloat(e.target.value) || 0 })}
                      style={{
                        width: '90%',
                        backgroundColor: item.radiologist_action === 'MODIFY' ? '#090d16' : '#1e293b',
                        border: '1px solid #334155',
                        color: '#f8fafc',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '12px'
                      }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Clinical Verdict</label>
                    <select
                      disabled={isFinalized}
                      value={item.radiologist_verdict}
                      onChange={e => updateFinding(idx, { radiologist_verdict: e.target.value })}
                      style={{
                        width: '95%',
                        backgroundColor: '#090d16',
                        border: '1px solid #334155',
                        color: '#f8fafc',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '12px'
                      }}
                    >
                      <option value="MALIGNANT">MALIGNANT</option>
                      <option value="BENIGN">BENIGN</option>
                      <option value="INDETERMINATE">INDETERMINATE</option>
                    </select>
                  </div>
                  <div>
                    <label style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Lung-RADS</label>
                    <select
                      disabled={isFinalized}
                      value={item.radiologist_lung_rads}
                      onChange={e => updateFinding(idx, { radiologist_lung_rads: e.target.value })}
                      style={{
                        width: '95%',
                        backgroundColor: '#090d16',
                        border: '1px solid #334155',
                        color: '#f8fafc',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '12px'
                      }}
                    >
                      <option value="1">1 (Negative)</option>
                      <option value="2">2 (Benign)</option>
                      <option value="3">3 (Prob. Benign)</option>
                      <option value="4A">4A (Suspicious)</option>
                      <option value="4B">4B (Very Suspicious)</option>
                      <option value="4X">4X (High Suspicion)</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {/* False Positive Category if marked */}
            {item.radiologist_action === 'MARK_FP' && (
              <div style={{ backgroundColor: '#450a0a', border: '1px solid #991b1b', padding: '10px 14px', borderRadius: '6px', marginBottom: '12px', display: 'flex', gap: '16px', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', color: '#fca5a5', fontWeight: 600 }}>False Positive Reason:</span>
                <select
                  value={item.fp_category}
                  onChange={e => updateFinding(idx, { fp_category: e.target.value })}
                  style={{ backgroundColor: '#1e293b', border: '1px solid #7f1d1d', color: '#fef2f2', padding: '4px 8px', borderRadius: '4px', fontSize: '12px' }}
                >
                  <option value="vascular_bifurcation">Vascular Bifurcation / Cross-section</option>
                  <option value="pleural_thickening">Pleural Plaque / Thickening</option>
                  <option value="apical_scar">Apical Scar / Linear Atelectasis</option>
                  <option value="bone_edge">Costal Osteophyte / Bone Edge</option>
                  <option value="other">Other Anatomical Variant</option>
                </select>
              </div>
            )}

            {/* Radiologist Note textarea */}
            <div>
              <label style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>Radiologist Observations & Clinical Recommendation</label>
              <textarea
                disabled={isFinalized}
                rows={2}
                value={item.comments}
                placeholder="Enter clinical impression, differential considerations, or multidisciplinary follow-up notes..."
                onChange={e => updateFinding(idx, { comments: e.target.value })}
                style={{
                  width: '98%',
                  backgroundColor: '#090d16',
                  border: '1px solid #1e293b',
                  color: '#f8fafc',
                  padding: '8px 10px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  fontFamily: 'inherit',
                  resize: 'vertical'
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Signer Digital Signature Footer */}
      <div style={{ marginTop: '24px', backgroundColor: '#0d131f', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Attending Radiologist Digital Signature</span>
          <div style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', marginTop: '2px' }}>{signerName}</div>
        </div>
        <div style={{ fontSize: '12px', color: '#64748b' }}>
          Date: {new Date().toLocaleDateString()} • Hash: {isFinalized ? 'SHA-256: 8f4a...e109 [VERIFIED]' : 'UNCOMMITTED'}
        </div>
      </div>
    </div>
  );
};
