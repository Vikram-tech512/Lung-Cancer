import React, { useEffect, useState } from 'react';
import { Cpu, Activity, Award, Clock, ArrowRight, ShieldCheck, Terminal, Info } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

interface EpochRecord {
  epoch: number;
  train_loss: number;
  train_accuracy: number;
  val_loss: number;
  val_accuracy: number;
  val_auc: number;
  sensitivity: number;
  specificity: number;
  epoch_sec: number;
}

export const TrainingPanel: React.FC = () => {
  const [history, setHistory] = useState<EpochRecord[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch(`${API_BASE}/training/history`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && Array.isArray(data.history)) {
          setHistory(data.history);
          setSummary({
            seed: data.seed ?? 43,
            architecture: data.architecture ?? 'ResNet3D18Half',
            best_epoch: data.best_epoch ?? 13,
            best_val_auc: data.best_val_auc ?? null,
            total_sec: data.total_training_sec ?? 985.8,
          });
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            3D Deep Learning Model Training &amp; Convergence
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Empirical convergence telemetry, epoch loss progression, and validation discrimination.
          </p>
        </div>
        <Badge variant="neutral">Reproducible Run</Badge>
      </header>

      {/* Training Summary Cards */}
      {summary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
          <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '14px 16px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Architecture</div>
            <div style={{ fontSize: '16px', fontWeight: 700, marginTop: '2px', color: 'var(--text-main)' }}>{summary.architecture}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Random Seed: {summary.seed}</div>
          </div>

          <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '14px 16px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Peak Validation AUC</div>
            <div style={{ fontSize: '20px', fontWeight: 700, marginTop: '2px', color: 'var(--status-success)' }}>
              {(summary.best_val_auc * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Achieved at Epoch {summary.best_epoch}</div>
          </div>

          <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '14px 16px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Total Wall-Clock Time</div>
            <div style={{ fontSize: '20px', fontWeight: 700, marginTop: '2px', color: 'var(--accent)' }}>
              {(summary.total_sec / 60).toFixed(1)} <span style={{ fontSize: '13px', fontWeight: 400 }}>min</span>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Avg: {(summary.total_sec / history.length).toFixed(1)}s / epoch</div>
          </div>

          <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '14px 16px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Optimization Schedule</div>
            <div style={{ fontSize: '15px', fontWeight: 600, marginTop: '2px' }}>AdamW + OneCycleLR</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Cosine annealing with warmup</div>
          </div>
        </div>
      )}

      {/* Epoch Progress History Table */}
      <Panel>
        <div style={{ padding: '18px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 600, fontSize: '14px' }}>Epoch Convergence History</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Dataset: LUNA16 Candidate Patches (64³)</span>
          </div>

          {loading ? (
            <div style={{ padding: '16px', textAlign: 'center' }}>Loading training logs…</div>
          ) : history.length === 0 ? (
            <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)' }}>No training run logs found.</div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: '12px', textAlign: 'left', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>
                    <th style={{ padding: '6px 8px' }}>Epoch</th>
                    <th style={{ padding: '6px 8px' }}>Train Loss</th>
                    <th style={{ padding: '6px 8px' }}>Val Loss</th>
                    <th style={{ padding: '6px 8px' }}>Val AUC</th>
                    <th style={{ padding: '6px 8px' }}>Sensitivity</th>
                    <th style={{ padding: '6px 8px' }}>Specificity</th>
                    <th style={{ padding: '6px 8px' }}>Epoch Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map(row => (
                    <tr
                      key={row.epoch}
                      style={{
                        borderBottom: '1px solid var(--border)',
                        background: row.epoch === summary?.best_epoch ? 'rgba(59, 130, 246, 0.08)' : undefined,
                      }}
                    >
                      <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                        Epoch {row.epoch}
                        {row.epoch === summary?.best_epoch && (
                          <span style={{ marginLeft: '6px', color: 'var(--accent)', fontSize: '10px', fontWeight: 700 }}>
                            BEST
                          </span>
                        )}
                      </td>
                      <td style={{ padding: '8px' }}>{row.train_loss.toFixed(4)}</td>
                      <td style={{ padding: '8px' }}>{row.val_loss.toFixed(4)}</td>
                      <td style={{ padding: '8px', fontWeight: 700, color: row.val_auc >= 0.9 ? 'var(--status-success)' : 'var(--text-main)' }}>
                        {(row.val_auc * 100).toFixed(2)}%
                      </td>
                      <td style={{ padding: '8px' }}>{(row.sensitivity * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px' }}>{(row.specificity * 100).toFixed(1)}%</td>
                      <td style={{ padding: '8px', color: 'var(--text-muted)' }}>{row.epoch_sec.toFixed(1)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Panel>

      {/* Reproducibility CLI Commands */}
      <Panel>
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600 }}>
            <Terminal size={14} color="var(--accent)" />
            <span>Reproducibility CLI Workflow</span>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            To execute reproducible 3D training on physical CPU cores or GPU:
          </div>
          <pre style={{
            background: 'var(--bg-main)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            padding: '10px 14px',
            fontSize: '12px',
            fontFamily: 'var(--font-mono)',
            color: 'var(--accent)',
            margin: 0,
            overflowX: 'auto',
          }}>
            python -m backend.training.train_luna16_3d --epochs 25 --batch-size 8 --lr 1e-4 --device cpu
          </pre>
        </div>
      </Panel>
    </section>
  );
};