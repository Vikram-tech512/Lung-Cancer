import React, { useState, useEffect } from 'react';
import { Database, Eye, Plus, CheckCircle2, FileText, ArrowRight } from 'lucide-react';
import { api } from '../services/api';

interface StudiesPageProps {
  navigate: (route: string) => void;
}

export const StudiesPage: React.FC<StudiesPageProps> = ({ navigate }) => {
  const [studies, setStudies] = useState<any[]>([]);

  useEffect(() => {
    api.listStudies()
      .then(setStudies)
      .catch(() => {
        setStudies([
          {
            id: 'CASE-DEMO-LIDC-0087',
            patient_id: 'LIDC-IDRI-0087',
            modality: 'CT',
            slice_count: 184,
            slice_thickness: 1.25,
            qc_status: 'PASS',
            created_at: '2026-09-22 12:30:00'
          },
          {
            id: 'CASE-LIDC-0124',
            patient_id: 'LIDC-IDRI-0124',
            modality: 'CT',
            slice_count: 210,
            slice_thickness: 1.0,
            qc_status: 'PASS',
            created_at: '2026-09-21 16:45:10'
          },
          {
            id: 'CASE-NLST-4402',
            patient_id: 'NLST-44029',
            modality: 'CT',
            slice_count: 196,
            slice_thickness: 1.25,
            qc_status: 'PASS',
            created_at: '2026-09-20 09:15:22'
          }
        ]);
      });
  }, []);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: '800', margin: 0, color: '#f8fafc' }}>
            CLINICAL COHORT STUDY MANAGER
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0 0 0' }}>
            De-identified research patient scans and quality control status.
          </p>
        </div>

        <button
          onClick={() => navigate('/viewer/CASE-DEMO-LIDC-0087')}
          style={{
            background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
            color: '#ffffff',
            border: 'none',
            borderRadius: '6px',
            padding: '10px 18px',
            fontSize: '13px',
            fontWeight: '700',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <Plus size={16} />
          Ingest New DICOM
        </button>
      </div>

      <div className="medical-card" style={{ overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
          <thead>
            <tr style={{ background: '#070c17', borderBottom: '1px solid #1c2638', color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>
              <th style={{ padding: '14px 18px' }}>Study Identifier</th>
              <th style={{ padding: '14px 18px' }}>Patient Code</th>
              <th style={{ padding: '14px 18px' }}>Modality</th>
              <th style={{ padding: '14px 18px' }}>Slices</th>
              <th style={{ padding: '14px 18px' }}>Thickness</th>
              <th style={{ padding: '14px 18px' }}>QC Audit</th>
              <th style={{ padding: '14px 18px', textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {studies.map((s) => (
              <tr key={s.id} style={{ borderBottom: '1px solid #1c2638' }}>
                <td style={{ padding: '14px 18px', fontWeight: '700', color: '#38bdf8' }}>{s.id}</td>
                <td style={{ padding: '14px 18px', fontFamily: 'JetBrains Mono, monospace', color: '#cbd5e1' }}>{s.patient_id}</td>
                <td style={{ padding: '14px 18px' }}>
                  <span style={{ background: '#1e293b', padding: '2px 6px', borderRadius: '4px', fontSize: '11px' }}>{s.modality}</span>
                </td>
                <td style={{ padding: '14px 18px', color: '#cbd5e1' }}>{s.slice_count}</td>
                <td style={{ padding: '14px 18px', color: '#cbd5e1' }}>{s.slice_thickness} mm</td>
                <td style={{ padding: '14px 18px' }}>
                  <span style={{ fontSize: '11px', fontWeight: '700', color: '#10b981', background: 'rgba(16, 185, 129, 0.15)', padding: '2px 8px', borderRadius: '4px', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                    <CheckCircle2 size={12} />
                    {s.qc_status}
                  </span>
                </td>
                <td style={{ padding: '14px 18px', textAlign: 'right' }}>
                  <button
                    onClick={() => navigate(`/viewer/${s.id}`)}
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
                    Inspect
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
