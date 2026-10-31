import React from 'react';
import { Badge } from '../components/ui/Badge';
import { Panel } from '../components/ui/Panel';

export const ClinicalReportPanel: React.FC<{ studyId?: string; noduleId?: string }> = () => (
  <section className="report-screen" style={{ display: 'flex', flexDirection: 'column', gap: '12px', color: '#20252a' }} aria-labelledby="clinical-report-title">
    <header>
      <h2 id="clinical-report-title" style={{ margin: 0, fontSize: '15px', fontWeight: 600 }}>Report draft</h2>
      <p style={{ margin: '4px 0 0', color: '#5b6570', fontSize: '12px' }}>A report cannot be populated until a verified study result is available.</p>
    </header>
    <Panel>
      <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <Badge variant="warning">Research use only · Not signed</Badge>
        <div>Findings, measurements, probability, uncertainty, prior comparison, and category: Not available.</div>
        <div>Export is disabled because there is no verified report content. No patient or accession identifiers are shown.</div>
        <div>Any future draft requires clinician review and must never be auto-signed.</div>
      </div>
    </Panel>
  </section>
);
