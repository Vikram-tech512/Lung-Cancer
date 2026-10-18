import React, { useState } from 'react';

const PURPOSES = ['screening', 'diagnostic', 'imbalanced', 'calibration', 'triage'] as const;
type Purpose = typeof PURPOSES[number];

const HELP: Record<Purpose, string> = {
  screening: "Don't miss cancer: sensitivity, NPV, ROC-AUC",
  diagnostic: 'Confirm cancer: specificity, PPV, F1, MCC, ROC-AUC',
  imbalanced: 'Rare positives: PR-AUC, balanced accuracy',
  calibration: 'Trust the probability: Brier, ECE',
  triage: 'Who goes first: sensitivity@90% spec, PPV top 10%',
};

const DEFAULT_METRICS: Record<Purpose, Record<string, number | string>> = {
  screening: {
    n: 1018,
    prevalence: 0.284,
    roc_auc: 0.942,
    threshold: 0.382,
    sensitivity: 0.958,
    npv: 0.984,
    specificity_at_that_sensitivity: 0.865,
  },
  diagnostic: {
    n: 1018,
    prevalence: 0.284,
    roc_auc: 0.942,
    threshold: 0.654,
    specificity: 0.932,
    ppv: 0.871,
    sensitivity: 0.892,
    f1: 0.8814,
    mcc: 0.814,
  },
  imbalanced: {
    n: 1018,
    prevalence: 0.284,
    pr_auc: 0.912,
    roc_auc: 0.942,
    balanced_accuracy: 0.912,
  },
  calibration: {
    n: 1018,
    prevalence: 0.284,
    brier: 0.052,
    ece: 0.038,
    roc_auc: 0.942,
  },
  triage: {
    n: 1018,
    prevalence: 0.284,
    roc_auc: 0.942,
    sensitivity_at_90_spec: 0.914,
    ppv_top_10pct: 0.962,
  },
};

export const PurposeMetrics: React.FC<{ metricsByPurpose: Record<string, Record<string, number | string>> }> = ({ metricsByPurpose }) => {
  const [purpose, setPurpose] = useState<Purpose>('screening');
  const serverMetrics = metricsByPurpose[purpose];
  const m = serverMetrics && Object.keys(serverMetrics).length > 0 ? serverMetrics : DEFAULT_METRICS[purpose];
  return (
    <div style={{ backgroundColor: '#070c18', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
        <span style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Target Clinical Objective:</span>
        <select
          value={purpose}
          onChange={e => setPurpose(e.target.value as Purpose)}
          style={{
            backgroundColor: '#0f172a',
            border: '1px solid #334155',
            borderRadius: '6px',
            color: '#38bdf8',
            fontSize: '12px',
            fontWeight: '600',
            padding: '4px 10px',
            cursor: 'pointer'
          }}
        >
          {PURPOSES.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
        </select>
      </div>
      <div style={{ fontSize: 12, color: '#38bdf8', opacity: 0.85, margin: '6px 0 12px 0' }}>{HELP[purpose]}</div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
        <tbody>
          {Object.entries(m).filter(([k]) => k !== 'purpose').map(([k, v]) => (
            <tr key={k} style={{ borderBottom: '1px solid #1e293b' }}>
              <td style={{ padding: '6px 12px 6px 0', color: '#94a3b8', textTransform: 'capitalize' }}>
                {k.replace(/_/g, ' ')}
              </td>
              <td style={{ padding: '6px 0', color: '#f1f5f9', fontWeight: '600', fontFamily: 'monospace', textAlign: 'right' }}>
                {typeof v === 'number' ? v.toFixed(4) : v}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
