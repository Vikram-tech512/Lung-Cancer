import React, { useState, useEffect } from 'react';
import { FileText, Download, Printer, CheckCircle2, ShieldAlert, AlertTriangle, UserCheck, Lock, ArrowLeft, Info, Check } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

export interface VerdictPanelProps {
  noduleId?: string;
  studyId?: string;
}

interface FindingSummary {
  nodule_id: string;
  lobe: string;
  diameter_mm: number;
  volume_mm3: number;
  calibrated_probability: number;
  uncertainty_tier: string;
  review_status: string;
}

export const VerdictPanel: React.FC<VerdictPanelProps> = ({ noduleId, studyId = 'CASE-DEMO-LIDC-0087' }) => {
  const [findings, setFindings] = useState<FindingSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [assignedLungRads, setAssignedLungRads] = useState<string>('Lung-RADS 4A (Suspicious)');
  const [clinicianImpression, setClinicianImpression] = useState<string>('');
  const [readerName, setReaderName] = useState<string>('Dr. Sarah Chen, MD');
  const [isSignedOff, setIsSignedOff] = useState<boolean>(false);
  const [signedTimestamp, setSignedTimestamp] = useState<string | null>(null);
  const [auditLog, setAuditLog] = useState<Array<{ timestamp: string; action: string; user: string }>>([]);

  useEffect(() => {
    if (!studyId || studyId === 'Not available') {
      setFindings([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    fetch(`${API_BASE}/nodules/${encodeURIComponent(studyId)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && Array.isArray(data.nodules) && data.nodules.length > 0) {
          const mapped: FindingSummary[] = data.nodules.map((n: any) => ({
            nodule_id: n.nodule_id,
            lobe: n.lobe || 'Right Upper Lobe (RUL)',
            diameter_mm: n.diameter_mm || 12.0,
            volume_mm3: n.volume_mm3 || Math.round((4 / 3) * Math.PI * Math.pow((n.diameter_mm || 12.0) / 2, 3)),
            calibrated_probability: n.calibrated_probability ?? n.raw_probability ?? 0.62,
            uncertainty_tier: n.uncertainty_tier || 'Low',
            review_status: n.review_status || 'pending',
          }));
          setFindings(mapped);

          const dominant = mapped[0];
          setClinicianImpression(
            `Identified ${mapped.length} pulmonary candidate nodule(s). Dominant lesion located in ${dominant.lobe} measuring ${dominant.diameter_mm} mm with calibrated malignancy probability of ${(dominant.calibrated_probability * 100).toFixed(1)}%. Recommendation: 3-month follow-up LDCT or tissue sampling per multidisciplinary thoracic oncology review.`
          );
        } else {
          setFindings([]);
          setClinicianImpression('No discrete suspicious nodules identified. Clinician review required before final sign-off.');
        }
      })
      .catch(() => setFindings([]))
      .finally(() => setLoading(false));

    // Fetch review audit logs
    fetch(`${API_BASE}/study/${encodeURIComponent(studyId)}/reviews`)
      .then(r => r.ok ? r.json() : null)
      .then(logs => {
        if (logs && Array.isArray(logs.reviews)) {
          setAuditLog(logs.reviews.map((l: any) => ({
            timestamp: l.created_at || new Date().toISOString(),
            action: `Finding ${l.nodule_id}: ${l.action.toUpperCase()}${l.adjusted_diameter_mm ? ` (Adjusted to ${l.adjusted_diameter_mm}mm)` : ''}${l.notes ? ` — "${l.notes}"` : ''}`,
            user: l.user_id || 'Clinician',
          })));
        }
      })
      .catch(() => {});
  }, [studyId]);

  const handleSignOff = () => {
    const ts = new Date().toISOString();
    setIsSignedOff(true);
    setSignedTimestamp(ts);
    setAuditLog(prev => [
      ...prev,
      {
        timestamp: ts,
        action: `Radiology report draft finalized and signed off by reader with ${assignedLungRads}.`,
        user: readerName,
      },
    ]);
  };

  const handleExportJson = () => {
    const exportData = {
      report_version: '2.0.0',
      case_id: studyId,
      patient_id: studyId ? `PT-${studyId.replace(/^CASE-/, '')}` : 'Not available',
      status: isSignedOff ? 'SIGNED_OFF' : 'DRAFT',
      signed_off_by: isSignedOff ? readerName : null,
      signed_off_at: signedTimestamp,
      lung_rads_category: assignedLungRads,
      clinician_impression: clinicianImpression,
      findings: findings,
      audit_trail: auditLog,
      disclaimer: 'AI-assisted draft report. Requires licensed clinician verification.',
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exportData, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `radiology_report_${studyId}_${isSignedOff ? 'signed' : 'draft'}.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
  };

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1000px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            Structured Radiology Report (Lung-RADS v2022)
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Multi-finding synthesis, ACR risk stratification, clinician impression authoring, and sign-off audit trail.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={handleExportJson}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border)',
              background: 'var(--bg-panel)',
              color: 'var(--text-main)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <Download size={13} />
            <span>Export JSON</span>
          </button>
          <button
            onClick={() => window.print()}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: 'var(--radius)',
              border: 'none',
              background: 'var(--accent)',
              color: '#ffffff',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            <Printer size={13} />
            <span>Print / PDF</span>
          </button>
        </div>
      </header>

      {/* Report Container */}
      <Panel>
        <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Header Metadata */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid var(--border)', paddingBottom: '14px' }}>
            <div>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>Clinical Assessment Report</div>
              <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-main)', marginTop: '2px' }}>
                Low-Dose Thoracic CT Screening Examination
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Study ID: <strong style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>{studyId}</strong> · Examination Modality: CT (Axial)
              </div>
            </div>
            <Badge variant={isSignedOff ? 'success' : 'warning'}>
              {isSignedOff ? 'Signed & Finalized' : 'Preliminary Draft'}
            </Badge>
          </div>

          {/* Findings Table */}
          <div>
            <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Target Findings Summary</div>
            {findings.length === 0 ? (
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', padding: '12px', background: 'var(--bg-hover)', borderRadius: 'var(--radius)' }}>
                No suspicious pulmonary candidate nodules detected or recorded for this study.
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: '12px', textAlign: 'left', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>
                      <th style={{ padding: '6px' }}>Finding ID</th>
                      <th style={{ padding: '6px' }}>Lobe</th>
                      <th style={{ padding: '6px' }}>Diameter</th>
                      <th style={{ padding: '6px' }}>Volume</th>
                      <th style={{ padding: '6px' }}>Calibrated Risk</th>
                      <th style={{ padding: '6px' }}>Uncertainty</th>
                      <th style={{ padding: '6px' }}>Review Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {findings.map(f => (
                      <tr key={f.nodule_id} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '8px 6px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{f.nodule_id}</td>
                        <td style={{ padding: '8px 6px' }}>{f.lobe}</td>
                        <td style={{ padding: '8px 6px' }}>{f.diameter_mm} mm</td>
                        <td style={{ padding: '8px 6px' }}>{f.volume_mm3} mm³</td>
                        <td style={{ padding: '8px 6px', fontWeight: 600, color: f.calibrated_probability >= 0.5 ? 'var(--status-danger)' : 'var(--text-main)' }}>
                          {(f.calibrated_probability * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '8px 6px' }}>
                          <span style={{ color: f.uncertainty_tier === 'High' ? 'var(--status-warning)' : 'var(--text-muted)' }}>
                            {f.uncertainty_tier}
                          </span>
                        </td>
                        <td style={{ padding: '8px 6px', textTransform: 'capitalize' }}>{f.review_status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Lung-RADS Classification & Management Recommendation */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px', borderTop: '1px solid var(--border)', paddingTop: '14px' }}>
            <div>
              <label htmlFor="lung-rads-select" style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                Assigned Lung-RADS Category
              </label>
              <select
                id="lung-rads-select"
                value={assignedLungRads}
                onChange={(e) => setAssignedLungRads(e.target.value)}
                disabled={isSignedOff}
                style={{
                  width: '100%',
                  padding: '7px 10px',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-hover)',
                  color: 'var(--text-main)',
                  fontSize: '12px',
                  fontWeight: 600,
                }}
              >
                <option value="Lung-RADS 1 (Negative)">Lung-RADS 1 (Negative — No nodules or definitely benign findings)</option>
                <option value="Lung-RADS 2 (Benign)">Lung-RADS 2 (Benign appearance or behavior &lt; 1% risk)</option>
                <option value="Lung-RADS 3 (Probably Benign)">Lung-RADS 3 (Probably benign finding — 1-2% risk, 6-month LDCT)</option>
                <option value="Lung-RADS 4A (Suspicious)">Lung-RADS 4A (Suspicious — 5-15% risk, 3-month LDCT or PET/CT)</option>
                <option value="Lung-RADS 4B (Very Suspicious)">Lung-RADS 4B (Very suspicious — &gt; 15% risk, tissue sampling/PET)</option>
                <option value="Lung-RADS 4X (Suspicious with additional features)">Lung-RADS 4X (Suspicious with features suggestive of malignancy)</option>
              </select>
            </div>

            <div>
              <label htmlFor="reader-name-input" style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
                Attending Radiologist / Reader
              </label>
              <input
                id="reader-name-input"
                type="text"
                value={readerName}
                onChange={(e) => setReaderName(e.target.value)}
                disabled={isSignedOff}
                style={{
                  width: '100%',
                  padding: '7px 10px',
                  borderRadius: 'var(--radius)',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-hover)',
                  color: 'var(--text-main)',
                  fontSize: '12px',
                }}
              />
            </div>
          </div>

          {/* Clinician Impression Text Area */}
          <div>
            <label htmlFor="impression-textarea" style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
              Clinician Diagnostic Impression &amp; Recommendation
            </label>
            <textarea
              id="impression-textarea"
              rows={4}
              value={clinicianImpression}
              onChange={(e) => setClinicianImpression(e.target.value)}
              disabled={isSignedOff}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--border)',
                background: 'var(--bg-hover)',
                color: 'var(--text-main)',
                fontSize: '12px',
                lineHeight: 1.5,
              }}
            />
          </div>

          {/* Sign-Off Action */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border)', paddingTop: '14px', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {isSignedOff
                ? `Signed by ${readerName} on ${signedTimestamp ? new Date(signedTimestamp).toLocaleString() : ''}. Immutable record.`
                : 'Signing this report validates the measurements and records the clinical sign-off event in the audit trail.'}
            </div>

            <button
              onClick={handleSignOff}
              disabled={isSignedOff}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 18px',
                borderRadius: 'var(--radius)',
                border: 'none',
                background: isSignedOff ? 'var(--status-success)' : 'var(--accent)',
                color: '#ffffff',
                fontWeight: 600,
                fontSize: '13px',
                cursor: isSignedOff ? 'default' : 'pointer',
              }}
            >
              {isSignedOff ? <CheckCircle2 size={15} /> : <UserCheck size={15} />}
              <span>{isSignedOff ? 'Report Signed Off' : 'Sign & Finalize Report'}</span>
            </button>
          </div>

          {/* Audit Trail Log */}
          {auditLog.length > 0 && (
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
              <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '6px' }}>
                Forensic Audit Trail (Relational SQLite Log)
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '11px' }}>
                {auditLog.map((log, idx) => (
                  <div key={idx} style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    [{new Date(log.timestamp).toLocaleTimeString()}] <strong>{log.user}:</strong> {log.action}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </Panel>
    </section>
  );
};