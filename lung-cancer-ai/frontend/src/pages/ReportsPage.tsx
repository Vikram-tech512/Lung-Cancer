import React, { useState, useEffect } from 'react';
import { FileText, Download, CheckCircle2, ShieldCheck, Clock } from 'lucide-react';
import { api } from '../services/api';

export const ReportsPage: React.FC = () => {
  const [reports, setReports] = useState<any[]>([]);

  useEffect(() => {
    api.listReports()
      .then(setReports)
      .catch(() => {});
  }, []);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '800', margin: 0, color: '#f8fafc' }}>
          CLINICAL AI PDF REPORT ARCHIVE
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0 0 0' }}>
          Download structured clinical PDF summaries compiling DICOM QC, 3D morphometry, and multimodal ensemble verdicts.
        </p>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {reports.map((r, i) => (
          <div key={i} className="medical-card" style={{ padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: 'rgba(2, 132, 199, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <FileText size={20} color="#38bdf8" />
              </div>
              <div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#f8fafc' }}>
                  Clinical Assessment Report — {r.study_id}
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
                  {r.summary}
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
                  Generated: {r.created_at}
                </div>
              </div>
            </div>

            <a
              href={r.download_url}
              target="_blank"
              rel="noreferrer"
              style={{
                background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
                color: '#ffffff',
                textDecoration: 'none',
                padding: '9px 16px',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: '700',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <Download size={14} />
              Download PDF
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};
