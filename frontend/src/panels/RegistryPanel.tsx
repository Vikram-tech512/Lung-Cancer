import React, { useEffect, useState } from 'react';
import { Box, Layers, ShieldCheck, AlertTriangle, CheckCircle2, HardDrive, Info } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

interface ModelVersion {
  model_name: string;
  version: string;
  stage: string;
  architecture: string;
  input_dimension?: string;
  checkpoint_path?: string | null;
  patient_split_hash?: string | null;
  notes?: string;
  metrics?: Record<string, any>;
}

export const RegistryPanel: React.FC = () => {
  const [registry, setRegistry] = useState<Record<string, ModelVersion[]>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    fetch(`${API_BASE}/registry`)
      .then(res => {
        if (!res.ok) throw new Error('Registry fetch failed');
        return res.json();
      })
      .then(data => setRegistry(data && typeof data === 'object' ? data : {}))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  const allModels: ModelVersion[] = Object.values(registry).flat();

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            Model Architecture &amp; Checkpoint Registry
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Versioned 3D deep learning architectures, training stages, input contracts, and weights provenance.
          </p>
        </div>
        <Badge variant="neutral">Models Registry v2.0</Badge>
      </header>

      <Panel>
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 600, fontSize: '14px' }}>Registered 3D Models &amp; Pipelines</span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Source: <code>checkpoints/models_registry.json</code>
            </span>
          </div>

          {loading ? (
            <div style={{ padding: '20px', textAlign: 'center' }}>Loading model registry…</div>
          ) : error || allModels.length === 0 ? (
            <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No registered model architectures found.
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: '12px', textAlign: 'left', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>
                    <th style={{ padding: '8px' }}>Model Name</th>
                    <th style={{ padding: '8px' }}>Architecture</th>
                    <th style={{ padding: '8px' }}>Lifecycle Stage</th>
                    <th style={{ padding: '8px' }}>Input Shape</th>
                    <th style={{ padding: '8px' }}>Checkpoint File</th>
                    <th style={{ padding: '8px' }}>Verification Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {allModels.map((m, idx) => (
                    <tr key={`${m.model_name}-${idx}`} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '10px 8px', fontWeight: 600 }}>
                        <div style={{ color: 'var(--text-main)' }}>{m.model_name}</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{m.version || 'v1.0.0'}</div>
                      </td>
                      <td style={{ padding: '10px 8px' }}>{m.architecture}</td>
                      <td style={{ padding: '10px 8px' }}>
                        {m.stage === 'STAGING' || m.stage === 'PRODUCTION' ? (
                          <Badge variant="success">{m.stage}</Badge>
                        ) : m.stage === 'NOT_TRAINED' ? (
                          <Badge variant="neutral">NOT TRAINED</Badge>
                        ) : (
                          <Badge variant="warning">{m.stage}</Badge>
                        )}
                      </td>
                      <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                        {m.input_dimension || 'Patch (32³ or 64³)'}
                      </td>
                      <td style={{ padding: '10px 8px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                        {m.checkpoint_path ? (
                          <span style={{ color: 'var(--status-success)' }}>{m.checkpoint_path}</span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)' }}>None (Uninitialized)</span>
                        )}
                      </td>
                      <td style={{ padding: '10px 8px', color: 'var(--text-muted)', fontSize: '11px', maxWidth: '280px' }}>
                        {m.notes || 'Experimental baseline.'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Panel>
    </section>
  );
};