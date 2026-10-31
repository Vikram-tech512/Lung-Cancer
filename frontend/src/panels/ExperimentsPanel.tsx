import React, { useEffect, useState } from 'react';
import { GitBranch, Activity, Clock, Layers, ShieldCheck, Terminal, Info } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

interface ExperimentRun {
  run_id: string;
  run_name: string;
  model_name: string;
  timestamp: string;
  params?: Record<string, any>;
  metrics?: Record<string, any>;
  tags?: Record<string, any>;
  artifacts?: string[];
}

export const ExperimentsPanel: React.FC = () => {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    fetch(`${API_BASE}/experiments`)
      .then(r => {
        if (!r.ok) throw new Error('Failed to fetch runs');
        return r.json();
      })
      .then(data => setRuns(Array.isArray(data.runs) ? data.runs : []))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            Scientific Experiment Tracking &amp; Runs
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Reproducible training configurations, hyperparameters, evaluation metrics, and artifact lineage.
          </p>
        </div>
        <Badge variant="neutral">MLflow Tracker</Badge>
      </header>

      <Panel>
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 600, fontSize: '14px' }}>Experiment Run History</span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Storage: <code>experiments_log/</code></span>
          </div>

          {loading ? (
            <div style={{ padding: '20px', textAlign: 'center' }}>Loading experiment history…</div>
          ) : error || runs.length === 0 ? (
            <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
              No recorded experiment runs found.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {runs.map((run) => (
                <div
                  key={run.run_id}
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius)',
                    background: 'var(--bg-hover)',
                    padding: '16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '12px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
                    <div>
                      <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-main)' }}>
                        {run.run_name || run.run_id}
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                        ID: {run.run_id} · Timestamp: {run.timestamp}
                      </div>
                    </div>
                    <Badge variant="neutral">{run.tags?.dataset || 'LUNA16'}</Badge>
                  </div>

                  {/* Hyperparameters Grid */}
                  {run.params && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px', fontSize: '11px' }}>
                      <div style={{ background: 'var(--bg-panel)', padding: '6px 8px', borderRadius: 'var(--radius)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Model: </span>
                        <strong>{run.params.model}</strong>
                      </div>
                      <div style={{ background: 'var(--bg-panel)', padding: '6px 8px', borderRadius: 'var(--radius)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Batch Size: </span>
                        <strong>{run.params.batch_size}</strong>
                      </div>
                      <div style={{ background: 'var(--bg-panel)', padding: '6px 8px', borderRadius: 'var(--radius)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Crop Size: </span>
                        <strong>{run.params.crop_size}³</strong>
                      </div>
                      <div style={{ background: 'var(--bg-panel)', padding: '6px 8px', borderRadius: 'var(--radius)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Learning Rate: </span>
                        <strong>{run.params.lr}</strong>
                      </div>
                      <div style={{ background: 'var(--bg-panel)', padding: '6px 8px', borderRadius: 'var(--radius)' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Seed: </span>
                        <strong>{run.params.seed}</strong>
                      </div>
                    </div>
                  )}

                  {/* Metrics Bar */}
                  {run.metrics && (
                    <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px', display: 'flex', gap: '16px', flexWrap: 'wrap', fontSize: '12px' }}>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Test AUC: </span>
                        <strong style={{ color: 'var(--accent)' }}>{(run.metrics.test_auc * 100).toFixed(1)}%</strong>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Sensitivity: </span>
                        <strong>{(run.metrics.test_sensitivity * 100).toFixed(1)}%</strong>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-muted)' }}>Specificity: </span>
                        <strong>{(run.metrics.test_specificity * 100).toFixed(1)}%</strong>
                      </div>
                      {run.metrics.test_ece !== undefined && (
                        <div>
                          <span style={{ color: 'var(--text-muted)' }}>Test ECE: </span>
                          <strong>{run.metrics.test_ece.toFixed(4)}</strong>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Artifacts */}
                  {run.artifacts && run.artifacts.length > 0 && (
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                      <strong>Artifact: </strong>
                      <code style={{ fontFamily: 'var(--font-mono)' }}>{run.artifacts.join(', ')}</code>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </Panel>
    </section>
  );
};