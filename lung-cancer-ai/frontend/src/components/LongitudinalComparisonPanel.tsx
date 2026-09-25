import React from 'react';

interface LongitudinalProps {
  currentStudyId: string;
  priorStudyId: string;
  daysBetween: number;
  overallAssessment: string;
  isReliable: boolean;
  failSafeMessage?: string;
  matchedNodules: Array<{
    current_nodule_id: string;
    prior_nodule_id: string;
    prior_diameter_mm: number;
    current_diameter_mm: number;
    diameter_change_pct: number;
    prior_volume_mm3: number;
    current_volume_mm3: number;
    volume_change_pct: number;
    volume_doubling_time_days?: number | null;
    growth_category: string;
    match_confidence: number;
    clinical_interpretation: string;
  }>;
  onClose?: () => void;
}

export const LongitudinalComparisonPanel: React.FC<LongitudinalProps> = ({
  currentStudyId,
  priorStudyId,
  daysBetween,
  overallAssessment,
  isReliable,
  failSafeMessage,
  matchedNodules,
  onClose
}) => {
  return (
    <div style={{
      backgroundColor: '#0d131f',
      border: '1px solid #1e293b',
      borderRadius: '8px',
      padding: '16px',
      color: '#e2e8f0',
      fontFamily: 'Inter, system-ui, sans-serif'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1e293b', paddingBottom: '10px', marginBottom: '12px' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 600, color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#a855f7' }} />
            Longitudinal Volumetric Tracking
          </h3>
          <p style={{ margin: '3px 0 0', fontSize: '11px', color: '#94a3b8' }}>
            Interval: {daysBetween} days ({priorStudyId.slice(0, 10)} → {currentStudyId.slice(0, 10)})
          </p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '16px' }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Safety Abstention Alert if unreliable */}
      {!isReliable && failSafeMessage && (
        <div style={{
          backgroundColor: '#451a03',
          border: '1px solid #b45309',
          color: '#fef3c7',
          padding: '10px 12px',
          borderRadius: '6px',
          fontSize: '12px',
          marginBottom: '14px',
          display: 'flex',
          gap: '8px',
          alignItems: 'center'
        }}>
          <span>⚠️</span>
          <div>
            <strong>Fail-Safe Triggered:</strong> {failSafeMessage}
          </div>
        </div>
      )}

      {/* Trajectory Status Banner */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: '#131d2e',
        padding: '10px 14px',
        borderRadius: '6px',
        border: '1px solid #1e293b',
        marginBottom: '14px'
      }}>
        <span style={{ fontSize: '12px', color: '#94a3b8' }}>Study-Level Trajectory</span>
        <span style={{
          fontSize: '13px',
          fontWeight: 700,
          color: overallAssessment.includes('Progress') ? '#f87171' : overallAssessment.includes('Regress') ? '#34d399' : '#38bdf8'
        }}>
          {overallAssessment}
        </span>
      </div>

      {/* Matched Nodules List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {matchedNodules.map((nod, idx) => (
          <div key={idx} style={{ backgroundColor: '#131d2e', padding: '12px', borderRadius: '6px', border: '1px solid #1e293b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: '#f1f5f9' }}>
                Nodule #{nod.current_nodule_id.slice(-4)} (Match Conf: {(nod.match_confidence * 100).toFixed(0)}%)
              </span>
              <span style={{
                fontSize: '11px',
                fontWeight: 600,
                padding: '2px 8px',
                borderRadius: '12px',
                backgroundColor: nod.growth_category.includes('Rapid') ? '#7f1d1d' : nod.growth_category === 'Stable' ? '#064e3b' : '#1e293b',
                color: nod.growth_category.includes('Rapid') ? '#fca5a5' : nod.growth_category === 'Stable' ? '#6ee7b7' : '#94a3b8'
              }}>
                {nod.growth_category}
              </span>
            </div>

            {/* Metrics comparison */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginBottom: '8px' }}>
              <div style={{ backgroundColor: '#090d16', padding: '8px', borderRadius: '4px' }}>
                <div style={{ fontSize: '10px', color: '#64748b' }}>Prior → Current Diam</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#e2e8f0', marginTop: '2px' }}>
                  {nod.prior_diameter_mm.toFixed(1)} → {nod.current_diameter_mm.toFixed(1)} mm
                </div>
                <div style={{ fontSize: '10px', color: nod.diameter_change_pct > 0 ? '#f87171' : '#34d399' }}>
                  {nod.diameter_change_pct > 0 ? '+' : ''}{nod.diameter_change_pct.toFixed(1)}%
                </div>
              </div>

              <div style={{ backgroundColor: '#090d16', padding: '8px', borderRadius: '4px' }}>
                <div style={{ fontSize: '10px', color: '#64748b' }}>Prior → Current Vol</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: '#e2e8f0', marginTop: '2px' }}>
                  {nod.prior_volume_mm3.toFixed(0)} → {nod.current_volume_mm3.toFixed(0)} mm³
                </div>
                <div style={{ fontSize: '10px', color: nod.volume_change_pct > 0 ? '#f87171' : '#34d399' }}>
                  {nod.volume_change_pct > 0 ? '+' : ''}{nod.volume_change_pct.toFixed(1)}%
                </div>
              </div>

              <div style={{ backgroundColor: '#090d16', padding: '8px', borderRadius: '4px' }}>
                <div style={{ fontSize: '10px', color: '#64748b' }}>Volume Doubling Time</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: nod.volume_doubling_time_days && nod.volume_doubling_time_days < 400 ? '#f87171' : '#e2e8f0', marginTop: '2px' }}>
                  {nod.volume_doubling_time_days ? `${nod.volume_doubling_time_days.toFixed(0)} days` : 'N/A (Stable)'}
                </div>
                <div style={{ fontSize: '10px', color: '#64748b' }}>
                  {nod.volume_doubling_time_days && nod.volume_doubling_time_days < 400 ? '<400d High Risk' : 'Standard kinetics'}
                </div>
              </div>
            </div>

            <div style={{ fontSize: '11px', color: '#94a3b8', fontStyle: 'italic' }}>
              {nod.clinical_interpretation}
            </div>
          </div>
        ))}

        {matchedNodules.length === 0 && (
          <div style={{ padding: '20px', textAlign: 'center', color: '#64748b', fontSize: '12px' }}>
            No prior lesions registered for longitudinal comparison.
          </div>
        )}
      </div>
    </div>
  );
};
