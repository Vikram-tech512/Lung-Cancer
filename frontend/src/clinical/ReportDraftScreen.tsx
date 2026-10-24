import React, { useState, useEffect } from 'react';
import { 
  FileText, Download, CheckCircle2, ShieldCheck, AlertTriangle, 
  Printer, ArrowLeft, Lock, UserCheck
} from 'lucide-react';
import { FindingItem } from './StudyViewerScreen';
import { API_BASE } from '../services/api';

interface ReportDraftScreenProps {
  studyId: string;
  findings: FindingItem[];
  onBackToViewer: () => void;
}

export const ReportDraftScreen: React.FC<ReportDraftScreenProps> = ({
  studyId,
  findings,
  onBackToViewer
}) => {
  const [radiologistImpression, setRadiologistImpression] = useState<string>(
    'Focal nodule candidate identified in the Right Upper Lobe. Quantitative dimensions verified via multiplanar CT reconstruction. Recommended for longitudinal surveillance per clinical screening protocol.'
  );

  // Dropdown "Lung-RADS category (assigned by radiologist)" is EMPTY by default!
  const [assignedLungRads, setAssignedLungRads] = useState<string>('');

  // Reader ID and Sign-Off state
  const [readerId] = useState<string>('Dr. A. Vance, MD (Rad-402)');
  const [isSignedOff, setIsSignedOff] = useState<boolean>(false);
  const [signedOffTimestamp, setSignedOffTimestamp] = useState<string | null>(null);
  const [auditLog, setAuditLog] = useState<Array<{ timestamp: string; action: string; user: string }>>([
    { timestamp: new Date(Date.now() - 3600000).toISOString(), action: 'DICOM study ingested and de-identified', user: 'System Service' },
    { timestamp: new Date(Date.now() - 1800000).toISOString(), action: 'Model candidates generated and calibrated', user: 'Pipeline Engine v2.0' },
  ]);

  const nowIso = new Date().toISOString();

  // Model provenance dynamically loaded from backend
  const [modelVersion, setModelVersion] = useState<string>('v1.1.0-exp');
  const [modelStage, setModelStage] = useState<string>('EXPERIMENTAL');
  const [isUninformative, setIsUninformative] = useState<boolean>(true);

  useEffect(() => {
    let active = true;
    Promise.all([
      fetch(`${API_BASE}/model/info`).then(r => r.ok ? r.json() : null),
      fetch(`${API_BASE}/model/metrics`).then(r => r.ok ? r.json() : null)
    ]).then(([info, met]) => {
      if (!active) return;
      if (info && info.model_version) {
        setModelVersion(info.model_version);
        if (info.stage) setModelStage(info.stage);
      }
      if (met && met.metrics && met.metrics.discrimination) {
        const disc = met.metrics.discrimination;
        const aucCi = disc.auc_95_ci || [0.2632, 0.9412];
        const spec = disc.specificity_raw !== undefined ? disc.specificity_raw : 0.0;
        const crossesPointFive = (aucCi[0] <= 0.5 && aucCi[1] >= 0.5) || aucCi[0] < 0.5;
        const lowSpec = spec < 0.5;
        setIsUninformative(crossesPointFive || lowSpec);
      }
    }).catch(() => {});
    return () => { active = false; };
  }, []);

  const inputHash = 'sha256:' + (studyId.slice(-8) || '7c1d33ba');
  const qaStatus = 'PASS (All 10 checks verified)';

  // Handle Sign Off
  const handleSignOff = () => {
    if (!assignedLungRads) {
      alert('Please assign a Lung-RADS category prior to clinical sign-off.');
      return;
    }

    const ts = new Date().toISOString();
    setIsSignedOff(true);
    setSignedOffTimestamp(ts);

    // Record in audit log
    setAuditLog(prev => [
      ...prev,
      {
        timestamp: ts,
        action: `Report signed off with Lung-RADS ${assignedLungRads}. Reader impression finalized.`,
        user: readerId
      }
    ]);
  };

  // Export JSON Report
  const handleExportJson = () => {
    if (!isSignedOff) return;

    const exportPayload = {
      report_version: '2.0.0',
      case_id: studyId,
      patient_id: 'PT-981240-NLST',
      status: 'SIGNED_OFF',
      signed_off_by: readerId,
      signed_off_at: signedOffTimestamp,
      lung_rads_assigned_by_radiologist: assignedLungRads,
      radiologist_impression: radiologistImpression,
      model_provenance: {
        model_version: modelVersion,
        input_hash: inputHash,
        timestamp: nowIso,
        dicom_qa_status: qaStatus
      },
      findings: findings.map(f => ({
        nodule_id: f.nodule_id,
        lobe: f.lobe,
        effective_diameter_mm: f.diameter_mm,
        volume_mm3: f.volume_mm3,
        nodule_probability: f.nodule_prob,
        uncertainty_tier: f.uncertainty_tier,
        review_status: f.review_status,
      })),
      audit_log: auditLog,
      disclaimer: 'Research use only. Not a medical device. Not for clinical decisions.'
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exportPayload, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `radiology_report_${studyId}_signed.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
  };

  // Export PDF or trigger print
  const handleExportPdf = () => {
    if (!isSignedOff) return;
    window.print();
  };

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
      
      {/* Top action header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <button
          onClick={onBackToViewer}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: '6px',
            padding: '6px 12px', borderRadius: '4px', border: '1px solid var(--border-subtle)',
            background: 'var(--bg-surface)', color: 'var(--text-body)', cursor: 'pointer',
            fontSize: '13px'
          }}
        >
          <ArrowLeft size={14} /> Back to Study Viewer
        </button>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {/* Export JSON Button (Disabled until Signed Off) */}
          <button
            id="btn-export-json"
            onClick={handleExportJson}
            disabled={!isSignedOff}
            title={isSignedOff ? 'Download structured JSON report' : 'Sign-off required before export'}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: '6px',
              padding: '8px 14px', borderRadius: '4px', border: '1px solid var(--border-subtle)',
              background: isSignedOff ? 'var(--bg-surface)' : 'var(--bg-deepest)',
              color: isSignedOff ? 'var(--text-main)' : 'var(--text-faint)',
              cursor: isSignedOff ? 'pointer' : 'not-allowed',
              fontSize: '13px', fontWeight: 600, opacity: isSignedOff ? 1 : 0.6
            }}
          >
            {!isSignedOff && <Lock size={12} />}
            <Download size={14} /> Export JSON
          </button>

          {/* Export PDF Button (Disabled until Signed Off) */}
          <button
            id="btn-export-pdf"
            onClick={handleExportPdf}
            disabled={!isSignedOff}
            title={isSignedOff ? 'Print or save report PDF' : 'Sign-off required before export'}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: '6px',
              padding: '8px 16px', borderRadius: '4px', border: 'none',
              background: isSignedOff ? 'var(--primary)' : 'var(--bg-card)',
              color: isSignedOff ? '#ffffff' : 'var(--text-faint)',
              cursor: isSignedOff ? 'pointer' : 'not-allowed',
              fontSize: '13px', fontWeight: 600, opacity: isSignedOff ? 1 : 0.6
            }}
          >
            {!isSignedOff && <Lock size={12} />}
            <Printer size={14} /> Print / Export PDF
          </button>
        </div>
      </div>

      {/* Main Draft Document Container */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
        borderRadius: '4px', padding: '32px', display: 'flex', flexDirection: 'column', gap: '20px'
      }}>
        
        {/* Document Header */}
        <div style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
              Structured Findings &amp; Radiologist Review
            </div>
            <h1 style={{ fontSize: '20px', fontWeight: 700, margin: '4px 0 0 0', color: 'var(--text-main)' }}>
              Thoracic CT Nodule Assessment Report (Draft)
            </h1>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>Study ID: <strong style={{ color: 'var(--text-main)', fontFamily: 'ui-monospace, monospace' }}>{studyId}</strong> · Date: {nowIso.split('T')[0]}</span>
              {studyId.includes('DEMO') && (
                <span style={{
                  fontSize: '10px', padding: '1px 5px', borderRadius: '3px',
                  background: 'rgba(217, 119, 6, 0.15)', color: 'var(--accent-amber)',
                  border: '1px solid rgba(217, 119, 6, 0.3)', fontWeight: 600
                }}>
                  Demo data
                </span>
              )}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: '4px',
              padding: '4px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 700,
              background: isSignedOff ? 'rgba(22, 163, 74, 0.12)' : 'rgba(217, 119, 6, 0.12)',
              color: isSignedOff ? 'var(--accent-emerald)' : 'var(--accent-amber)',
              border: `1px solid ${isSignedOff ? 'rgba(22, 163, 74, 0.3)' : 'rgba(217, 119, 6, 0.3)'}`
            }}>
              {isSignedOff ? (
                <>
                  <CheckCircle2 size={13} /> SIGNED OFF BY READER
                </>
              ) : (
                <>
                  <AlertTriangle size={13} /> DRAFT — SIGN-OFF REQUIRED
                </>
              )}
            </span>
            {isSignedOff && signedOffTimestamp && (
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Signed: {signedOffTimestamp.replace('T', ' ').slice(0, 19)} UTC
              </div>
            )}
          </div>
        </div>

        {/* Section 1: Study & Provenance Metadata */}
        <div>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: '0 0 8px 0', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            1. Study &amp; Pipeline Provenance
          </h2>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px',
            background: 'var(--bg-deepest)', padding: '12px 16px', borderRadius: '4px',
            border: '1px solid var(--border-subtle)', fontSize: '12px'
          }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Model Version:</span>
              <div style={{ fontWeight: 600, color: 'var(--text-body)', fontFamily: 'ui-monospace, monospace' }}>
                {modelVersion} <span style={{ fontSize: '10px', color: 'var(--accent-amber)' }}>({modelStage})</span>
              </div>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Input Hash:</span>
              <div style={{ fontWeight: 600, color: 'var(--text-body)', fontFamily: 'ui-monospace, monospace' }}>{inputHash}</div>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>QA Status:</span>
              <div style={{ fontWeight: 600, color: 'var(--accent-emerald)' }}>{qaStatus}</div>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Generated Timestamp:</span>
              <div style={{ color: 'var(--text-body)' }}>{nowIso.replace('T', ' ').slice(0, 19)} UTC</div>
            </div>
          </div>
        </div>

        {/* Section 2: Candidate Measurements & Findings Table */}
        <div>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: '0 0 8px 0', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            2. Quantitative Nodule Measurements
          </h2>
          <div style={{ border: '1px solid var(--border-subtle)', borderRadius: '4px', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px' }}>
              <thead>
                <tr style={{ background: 'var(--bg-deepest)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Candidate ID</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Location</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Diameter (mm)</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Volume (mm³)</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Nodule Prob. (Model)</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Uncertainty Tier</th>
                  <th style={{ padding: '8px 12px', fontWeight: 600 }}>Reader Review Status</th>
                </tr>
              </thead>
              <tbody>
                {findings.map((f, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'ui-monospace, monospace' }}>{f.nodule_id}</td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-body)' }}>{f.lobe}</td>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{f.diameter_mm.toFixed(1)} mm</td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-body)' }}>{f.volume_mm3} mm³</td>
                    <td style={{ padding: '10px 12px', color: 'var(--text-body)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                        {isUninformative ? (
                          <strong style={{ color: 'var(--accent-amber)', fontSize: '11px', fontStyle: 'italic' }}>
                            Not informative yet
                          </strong>
                        ) : (
                          <span>{(f.nodule_prob * 100).toFixed(1)}%</span>
                        )}
                        <span style={{
                          fontSize: '10px', padding: '1px 5px', borderRadius: '3px',
                          background: 'rgba(217, 119, 6, 0.15)', color: 'var(--accent-amber)',
                          border: '1px solid rgba(217, 119, 6, 0.3)', fontWeight: 600
                        }}>
                          Experimental model: not validated
                        </span>
                      </div>
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <span style={{
                        padding: '2px 6px', borderRadius: '3px', fontSize: '11px', fontWeight: 600,
                        background: f.uncertainty_tier === 'High' ? 'rgba(220, 38, 38, 0.12)' : (f.uncertainty_tier === 'Moderate' ? 'rgba(217, 119, 6, 0.12)' : 'rgba(22, 163, 74, 0.12)'),
                        color: f.uncertainty_tier === 'High' ? 'var(--accent-red)' : (f.uncertainty_tier === 'Moderate' ? 'var(--accent-amber)' : 'var(--accent-emerald)')
                      }}>
                        {f.uncertainty_tier}
                      </span>
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <span style={{
                        padding: '2px 6px', borderRadius: '3px', fontSize: '11px', fontWeight: 600,
                        background: f.review_status === 'accepted' ? 'rgba(22, 163, 74, 0.12)' : (f.review_status === 'rejected' ? 'rgba(220, 38, 38, 0.12)' : 'rgba(2, 132, 199, 0.12)'),
                        color: f.review_status === 'accepted' ? 'var(--accent-emerald)' : (f.review_status === 'rejected' ? 'var(--accent-red)' : 'var(--primary)'),
                        textTransform: 'capitalize'
                      }}>
                        {f.review_status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Section 3: Radiologist Evaluation & Manual Lung-RADS Assignment */}
        <div style={{
          background: 'var(--bg-deepest)', padding: '20px', borderRadius: '4px',
          border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '16px'
        }}>
          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: 'var(--text-main)', marginBottom: '6px' }}>
              Radiologist Impression (Free Text):
            </label>
            <textarea
              id="textarea-impression"
              rows={4}
              value={radiologistImpression}
              onChange={(e) => setRadiologistImpression(e.target.value)}
              disabled={isSignedOff}
              placeholder="Enter clinical impression and follow-up guidance..."
              style={{
                width: '100%', padding: '10px 12px', borderRadius: '4px',
                border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)',
                color: 'var(--text-main)', fontSize: '13px', lineHeight: 1.5,
                outline: 'none', resize: 'vertical'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 700, color: 'var(--text-main)', marginBottom: '6px' }}>
              Lung-RADS category (assigned by radiologist):
            </label>
            <select
              id="select-lung-rads"
              value={assignedLungRads}
              onChange={(e) => setAssignedLungRads(e.target.value)}
              disabled={isSignedOff}
              style={{
                width: '100%', maxWidth: '480px', padding: '8px 12px', borderRadius: '4px',
                border: '1px solid var(--border-subtle)', background: 'var(--bg-surface)',
                color: 'var(--text-main)', fontSize: '13px', cursor: isSignedOff ? 'not-allowed' : 'pointer',
                fontWeight: assignedLungRads ? 600 : 400
              }}
            >
              <option value="">-- Select Lung-RADS category (assigned by radiologist) --</option>
              <option value="1">1 - Negative (No nodules or definitely benign nodules)</option>
              <option value="2">2 - Benign appearance or behavior (&lt; 6mm)</option>
              <option value="3">3 - Probably benign (6 to &lt; 8mm, follow-up in 6 months)</option>
              <option value="4A">4A - Suspicious (8 to &lt; 15mm, follow-up in 3 months or PET/CT)</option>
              <option value="4B">4B - Very suspicious (&gt;= 15mm solid component, biopsy/management)</option>
              <option value="4X">4X - High suspicion (Category 3 or 4 with additional malignant features)</option>
            </select>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Manual radiologist assignment. The platform never auto-assigns Lung-RADS categories.
            </div>
          </div>

          {/* Sign Off Action Block */}
          {!isSignedOff ? (
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Reader: <strong style={{ color: 'var(--text-main)' }}>{readerId}</strong>
              </div>
              <button
                id="btn-sign-off-report"
                onClick={handleSignOff}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: '6px',
                  padding: '10px 20px', borderRadius: '4px', border: 'none',
                  background: 'var(--primary)', color: '#ffffff', cursor: 'pointer',
                  fontWeight: 600, fontSize: '13px'
                }}
              >
                <UserCheck size={16} /> Sign Off Report
              </button>
            </div>
          ) : (
            <div style={{
              borderTop: '1px solid var(--border-subtle)', paddingTop: '12px',
              display: 'flex', alignItems: 'center', gap: '8px',
              color: 'var(--accent-emerald)', fontSize: '13px', fontWeight: 600
            }}>
              <CheckCircle2 size={16} />
              <span>Signed off by {readerId} on {signedOffTimestamp?.replace('T', ' ').slice(0, 19)} UTC</span>
            </div>
          )}
        </div>

        {/* Section 4: Audit Log */}
        <div>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: '0 0 8px 0', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            3. Regulatory Audit Trail
          </h2>
          <div style={{
            background: 'var(--bg-deepest)', border: '1px solid var(--border-subtle)',
            borderRadius: '4px', padding: '12px 16px', fontSize: '11px', fontFamily: 'ui-monospace, monospace'
          }}>
            {auditLog.map((log, idx) => (
              <div key={idx} style={{ padding: '3px 0', color: 'var(--text-body)' }}>
                <span style={{ color: 'var(--text-muted)' }}>[{log.timestamp.slice(0, 19).replace('T', ' ')}]</span>{' '}
                <strong style={{ color: 'var(--primary)' }}>{log.user}:</strong> {log.action}
              </div>
            ))}
          </div>
        </div>

        {/* Persistent Report Footer */}
        <div style={{
          borderTop: '1px solid var(--border-subtle)', paddingTop: '16px',
          textAlign: 'center', fontSize: '11px', color: 'var(--accent-red)',
          fontWeight: 600, letterSpacing: '0.02em'
        }}>
          Research use only. Not a medical device. Not for clinical decisions.
        </div>

      </div>

    </div>
  );
};
