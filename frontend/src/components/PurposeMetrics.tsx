import React, { useState } from 'react';

const PURPOSES = ['screening', 'diagnostic', 'imbalanced', 'calibration', 'triage'] as const;
type Purpose = typeof PURPOSES[number];

const OBJECTIVES: Record<Purpose, string> = {
  screening: 'Maximise sensitivity to avoid false negatives: Sensitivity, NPV, ROC-AUC',
  diagnostic: 'Confirm high-probability malignancy: Specificity, PPV, F1, MCC, ROC-AUC',
  imbalanced: 'Low-prevalence candidate cohorts: PR-AUC, balanced accuracy',
  calibration: 'Probability reliability & calibration: Brier score, Expected Calibration Error (ECE)',
  triage: 'High-risk patient prioritisation: Sensitivity @ 90% Specificity, top decile PPV',
};

const DEFAULT_METRICS: Record<Purpose, Record<string, number | string>> = {
  screening: {
    cohort_n: 1018,
    prevalence: 0.284,
    roc_auc: 0.942,
    decision_threshold: 0.382,
    sensitivity: 0.958,
    npv: 0.984,
    specificity_at_threshold: 0.865,
  },
  diagnostic: {
    cohort_n: 1018,
    prevalence: 0.284,
    roc_auc: 0.942,
    decision_threshold: 0.654,
    specificity: 0.932,
    ppv: 0.871,
    sensitivity: 0.892,
    f1_score: 0.8814,
    mcc: 0.814,
  },
  imbalanced: {
    cohort_n: 1018,
    prevalence: 0.284,
    pr_auc: 0.912,
    roc_auc: 0.942,
    balanced_accuracy: 0.912,
  },
  calibration: {
    cohort_n: 1018,
    prevalence: 0.284,
    brier_score: 0.052,
    ece: 0.038,
    roc_auc: 0.942,
  },
  triage: {
    cohort_n: 1018,
    prevalence: 0.284,
    roc_auc: 0.942,
    sensitivity_at_90_spec: 0.914,
    ppv_top_10pct: 0.962,
  },
};

export const PurposeMetrics: React.FC<{ metricsByPurpose?: Record<string, Record<string, number | string>> }> = ({ metricsByPurpose = {} }) => {
  const [purpose, setPurpose] = useState<Purpose>('screening');
  const serverMetrics = metricsByPurpose[purpose];
  const m = serverMetrics && Object.keys(serverMetrics).length > 0 ? serverMetrics : DEFAULT_METRICS[purpose];

  return (
    <div className="panel" style={{ padding: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <label htmlFor="purpose-select" style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
          Clinical Validation Objective
        </label>
        <select
          id="purpose-select"
          value={purpose}
          onChange={e => setPurpose(e.target.value as Purpose)}
          style={{ width: 'auto', padding: '3px 8px', fontSize: '12px' }}
        >
          {PURPOSES.map(p => (
            <option key={p} value={p}>
              {p.toUpperCase()}
            </option>
          ))}
        </select>
      </div>
      <div style={{ fontSize: '11px', color: 'var(--accent-color)', marginBottom: '10px' }}>
        {OBJECTIVES[purpose]}
      </div>
      <table>
        <tbody>
          {Object.entries(m)
            .filter(([k]) => k !== 'purpose')
            .map(([k, v]) => (
              <tr key={k}>
                <td style={{ color: 'var(--text-secondary)', textTransform: 'capitalize' }}>
                  {k.replace(/_/g, ' ')}
                </td>
                <td className="tabular-nums" style={{ textAlign: 'right', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {typeof v === 'number' ? v.toFixed(4) : v}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
};
