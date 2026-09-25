import React from 'react';

export const CLINICAL_DISCLAIMER_TEXT =
  "AI-assisted preliminary analysis. Radiologist review is required. This system is intended for research/clinical decision-support evaluation and is not a standalone diagnostic device.";

export const ClinicalDisclaimerBanner: React.FC<{ compact?: boolean }> = ({ compact = false }) => {
  return (
    <div
      style={{
        backgroundColor: '#0a0f1d',
        borderTop: '1px solid #1c2638',
        padding: compact ? '6px 16px' : '10px 24px',
        color: '#f59e0b',
        fontSize: compact ? '11px' : '12px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
        textAlign: 'center',
        fontWeight: 500,
        letterSpacing: '0.01em'
      }}
    >
      <span style={{ fontSize: '13px' }}>⚠️</span>
      <span>{CLINICAL_DISCLAIMER_TEXT}</span>
    </div>
  );
};
