import React, { useEffect, useState } from 'react';
import { Database, ShieldCheck, CheckCircle2, UserCheck, Clock, ArrowRight, Info, AlertTriangle } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

interface StudyRecord {
  study_id: string;
  patient_id?: string;
  modality?: string;
  qc_status?: string;
  status?: string;
  slice_count?: number;
  study_date?: string;
  timestamp?: string;
  is_demo?: boolean;
}

interface ReviewRecord {
  id?: number;
  study_id: string;
  nodule_id: string;
  user_id: string;
  action: string;
  notes?: string;
  adjusted_diameter_mm?: number;
  created_at: string;
}

export const CaseLogPanel: React.FC<{ onSelectCase?: (caseId: string) => void }> = ({ onSelectCase }) => {
  const [studies, setStudies] = useState<StudyRecord[]>([]);
  const [reviews, setReviews] = useState<ReviewRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<'studies' | 'reviews'>('studies');

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/cases`).then(r => r.ok ? r.json() : { cases: [] }),
      fetch(`${API_BASE}/study/reviews`).then(r => r.ok ? r.json() : { reviews: [] }),
    ])
      .then(([casesData, reviewsData]) => {
        if (casesData && Array.isArray(casesData.cases)) setStudies(casesData.cases);
        if (reviewsData && Array.isArray(reviewsData.reviews)) setReviews(reviewsData.reviews);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            Case Audit Log &amp; Regulatory Tracking
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Forensic relational audit trail: study ingestions, reader evaluations, diameter adjustments, and diagnostic sign-offs.
          </p>
        </div>
        <div style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
          <button
            onClick={() => setActiveTab('studies')}
            style={{
              padding: '6px 14px',
              border: 'none',
              background: activeTab === 'studies' ? 'var(--accent)' : 'var(--bg-panel)',
              color: activeTab === 'studies' ? '#ffffff' : 'var(--text-muted)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Study Workflow ({studies.length})
          </button>
          <button
            onClick={() => setActiveTab('reviews')}
            style={{
              padding: '6px 14px',
              border: 'none',
              background: activeTab === 'reviews' ? 'var(--accent)' : 'var(--bg-panel)',
              color: activeTab === 'reviews' ? '#ffffff' : 'var(--text-muted)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Clinician Action Log ({reviews.length})
          </button>
        </div>
      </header>

      <Panel>
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {loading ? (
            <div style={{ padding: '20px', textAlign: 'center' }}>Loading case audit log…</div>
          ) : activeTab === 'studies' ? (
            studies.length === 0 ? (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                No study workflow records found. Upload a study to create an intake record.
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: '12px', textAlign: 'left', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>
                      <th style={{ padding: '8px' }}>Study ID</th>
                      <th style={{ padding: '8px' }}>Patient ID</th>
                      <th style={{ padding: '8px' }}>Slices</th>
                      <th style={{ padding: '8px' }}>QC Gate</th>
                      <th style={{ padding: '8px' }}>Review Status</th>
                      <th style={{ padding: '8px' }}>Timestamp</th>
                      <th style={{ padding: '8px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {studies.map((item, idx) => (
                      <tr key={`${item.study_id}-${idx}`} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                          {item.study_id}
                          {item.is_demo && (
                            <span style={{ marginLeft: '6px', color: 'var(--status-warning)', fontSize: '10px', fontWeight: 700 }}>
                              DEMO
                            </span>
                          )}
                        </td>
                        <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>{item.patient_id || 'ANON'}</td>
                        <td style={{ padding: '8px' }}>{item.slice_count ?? 'Not available'}</td>
                        <td style={{ padding: '8px' }}>
                          <Badge variant={item.qc_status === 'PASS' ? 'success' : 'warning'}>
                            {item.qc_status || 'PASS'}
                          </Badge>
                        </td>
                        <td style={{ padding: '8px' }}>{item.status || 'Ready for review'}</td>
                        <td style={{ padding: '8px', color: 'var(--text-muted)', fontSize: '11px' }}>
                          {item.study_date || item.timestamp || 'Not available'}
                        </td>
                        <td style={{ padding: '8px' }}>
                          <button
                            onClick={() => onSelectCase?.(item.study_id)}
                            style={{
                              padding: '4px 10px',
                              borderRadius: 'var(--radius)',
                              border: '1px solid var(--border)',
                              background: 'var(--bg-hover)',
                              color: 'var(--accent)',
                              fontSize: '11px',
                              fontWeight: 600,
                              cursor: 'pointer',
                            }}
                          >
                            Open in Viewer
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          ) : (
            reviews.length === 0 ? (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                No clinician review actions recorded yet. Accept, reject, or adjust findings to generate immutable audit log entries.
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: '12px', textAlign: 'left', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>
                      <th style={{ padding: '8px' }}>Timestamp</th>
                      <th style={{ padding: '8px' }}>Finding</th>
                      <th style={{ padding: '8px' }}>Clinician / User</th>
                      <th style={{ padding: '8px' }}>Action</th>
                      <th style={{ padding: '8px' }}>Adjusted Diameter</th>
                      <th style={{ padding: '8px' }}>Clinical Rationale / Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reviews.map((rev, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '8px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                          {new Date(rev.created_at).toLocaleString()}
                        </td>
                        <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{rev.nodule_id}</td>
                        <td style={{ padding: '8px' }}>{rev.user_id}</td>
                        <td style={{ padding: '8px' }}>
                          <Badge variant={rev.action === 'accept' ? 'success' : rev.action === 'reject' ? 'danger' : 'warning'}>
                            {rev.action.toUpperCase()}
                          </Badge>
                        </td>
                        <td style={{ padding: '8px' }}>
                          {rev.adjusted_diameter_mm ? `${rev.adjusted_diameter_mm} mm` : '—'}
                        </td>
                        <td style={{ padding: '8px', color: 'var(--text-main)', fontSize: '11px', maxWidth: '300px' }}>
                          {rev.notes || 'No note recorded'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          )}
        </div>
      </Panel>
    </section>
  );
};