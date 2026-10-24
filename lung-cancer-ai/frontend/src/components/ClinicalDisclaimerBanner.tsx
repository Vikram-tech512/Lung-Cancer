import React from 'react';

export const CLINICAL_DISCLAIMER_TEXT =
  "AI-assisted preliminary analysis. Radiologist review is required. This system is intended for research/clinical decision-support evaluation and is not a standalone diagnostic device.";

export const ClinicalDisclaimerBanner: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  return (
    <div
      style={{
        backgroundColor: 'var(--bg-surface)',
        borderTop: '1px solid var(--border)',
        padding: compact ? '6px 16px' : '8px 24px',
        color: 'var(--accent-amber)',
        fontSize: compact ? '11px' : '12px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
        textAlign: 'center',
        fontWeight: 500,
        letterSpacing: '0.01em',
        transition: 'background-color 0.3s ease, border-color 0.3s ease',
      }}
    >
      <span style={{ fontSize: '13px' }}>⚠️</span>
      <span>{CLINICAL_DISCLAIMER_TEXT}</span>
    </div>
  );
};
