import React, { useEffect, useState } from 'react';
import { Database, Eye, CheckCircle2, FileText } from 'lucide-react';
import { StudyCase } from '../types';
import { API_BASE } from '../services/api';

interface CaseLogPanelProps {
  onSelectCase: (studyId: string) => void;
}

export const CaseLogPanel: React.FC<CaseLogPanelProps> = ({ onSelectCase }) => {
  const [cases, setCases] = useState<StudyCase[]>([]);

  useEffect(() => {
    fetch(`${API_BASE}/cases`)
      .then(res => res.json())
      .then(data => setCases(data.cases || []))
      .catch(() => {
        setCases([
          {
            study_id: 'CASE-DEMO-LIDC-0087',
            patient_id: 'LIDC-IDRI-0087',
            modality: 'CT',
            slice_count: 184,
            qc_status: 'PASS',
            nodule_count: 1,
            timestamp: '2026-09-22 12:30:00'
          },
          {
            study_id: 'CASE-LIDC-0124',
            patient_id: 'LIDC-IDRI-0124',
            modality: 'CT',
            slice_count: 210,
            qc_status: 'PASS',
            nodule_count: 2,
            timestamp: '2026-09-21 16:45:10'
          },
          {
            study_id: 'CASE-NLST-4402',
            patient_id: 'NLST-44029',
            modality: 'CT',
            slice_count: 196,
            qc_status: 'PASS',
            nodule_count: 1,
            timestamp: '2026-09-20 09:15:22'
          }
        ]);
      });
  }, []);

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
          02 · CLINICAL CASE LOG & COHORT ARCHIVE
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '6px' }}>
          Track, reload, and review thoracic CT studies and their corresponding multi-model AI inferences.
        </p>
      </div>

      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
          <thead>
            <tr style={{ background: '#0b1120', borderBottom: '1px solid #1e293b', color: '#64748b', textTransform: 'uppercase', fontSize: '11px', letterSpacing: '0.05em' }}>
              <th style={{ padding: '14px 16px' }}>Study ID</th>
              <th style={{ padding: '14px 16px' }}>Patient Identifier</th>
              <th style={{ padding: '14px 16px' }}>Modality</th>
              <th style={{ padding: '14px 16px' }}>Slices</th>
              <th style={{ padding: '14px 16px' }}>QC Audit</th>
              <th style={{ padding: '14px 16px' }}>Nodules Detected</th>
              <th style={{ padding: '14px 16px', textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.study_id} style={{ borderBottom: '1px solid #1e293b', color: '#cbd5e1' }}>
                <td style={{ padding: '14px 16px', fontWeight: '600', color: '#38bdf8' }}>
                  {c.study_id}
                </td>
                <td style={{ padding: '14px 16px', fontFamily: 'JetBrains Mono, monospace' }}>
                  {c.patient_id}
                </td>
                <td style={{ padding: '14px 16px' }}>
                  <span style={{ background: '#1e293b', padding: '2px 6px', borderRadius: '4px', fontSize: '11px' }}>
                    {c.modality}
                  </span>
                </td>
                <td style={{ padding: '14px 16px' }}>{c.slice_count}</td>
                <td style={{ padding: '14px 16px' }}>
                  <span style={{
                    color: '#10b981',
                    background: 'rgba(16, 185, 129, 0.15)',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: '600',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    <CheckCircle2 size={12} />
                    {c.qc_status}
                  </span>
                </td>
                <td style={{ padding: '14px 16px', fontWeight: '600' }}>
                  {c.nodule_count} candidate(s)
                </td>
                <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                  <button
                    onClick={() => onSelectCase(c.study_id)}
                    style={{
                      background: 'rgba(56, 189, 248, 0.15)',
                      border: '1px solid rgba(56, 189, 248, 0.3)',
                      color: '#38bdf8',
                      padding: '6px 12px',
                      borderRadius: '5px',
                      fontSize: '12px',
                      fontWeight: '600',
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                  >
                    <Eye size={14} />
                    Load Study
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
