import React from 'react';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

export const RadiomicsPanel: React.FC<{ studyId?: string; noduleId?: string }> = () => (
  <section style={{ display: 'flex', flexDirection: 'column', gap: '12px' }} aria-labelledby="radiomics-title">
    <header>
      <h2 id="radiomics-title" style={{ margin: 0, fontSize: '15px', fontWeight: 600 }}>Radiomics features</h2>
      <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '12px' }}>Feature extraction and model scoring are not available without an image volume, verified mask, and recorded run.</p>
    </header>
    <Panel>
      <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <Badge variant="neutral">Not available</Badge>
        <div>Feature values, reference distributions, feature importance, and malignancy scoring are not available.</div>
        <div>No sample feature table is shown as patient or model output.</div>
      </div>
    </Panel>
  </section>
);
