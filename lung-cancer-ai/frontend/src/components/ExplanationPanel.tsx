import React, { useState } from 'react';
import { X, MapPin, Ruler, Layers, Activity, AlertTriangle } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

interface ExplanationPanelProps {
  noduleData: {
    id: string;
    location: string;
    slice: number;
    diameter_mm: number;
    volume_mm3: number;
    attenuation: string;
    margin: string;
    sphericity: number;
    spiculation_index: number;
    probability: number;
    ci_lower: number;
    ci_upper: number;
    uncertainty_level: string;
    disagreement_entropy: number;
    models: Array<{ name: string; prob: number; calibrated: number; uncertainty: string }>;
  };
  onClose?: () => void;
}

type Tab = 'summary' | 'ensemble';

const tabLabels: Record<Tab, string> = {
  summary: 'Overview',
  ensemble: 'Model Agreement',
};

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({ noduleData, onClose }) => {
  const [activeTab, setActiveTab] = useState<Tab>('summary');
  const { theme } = useTheme();

  const isDark = theme === 'dark';
  const risk = noduleData.probability;
  const isHighRisk = risk > 0.6;
  const riskColor = isHighRisk ? 'var(--risk-high)' : 'var(--risk-low)';
  const riskLabel = isHighRisk ? 'High Risk' : 'Low Risk';

  // ── shared styles ────────────────────────────────────────────
  const panel: React.CSSProperties = {
    backgroundColor: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: '12px',
    overflow: 'hidden',
    color: 'var(--text-main)',
    fontFamily: 'Inter, system-ui, sans-serif',
    boxShadow: 'var(--shadow-elevated)',
  };

  const header: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '14px 18px',
    borderBottom: '1px solid var(--border-subtle)',
    background: isDark
      ? 'linear-gradient(135deg, rgba(37,99,235,0.08), rgba(14,165,233,0.05))'
      : 'linear-gradient(135deg, rgba(37,99,235,0.04), rgba(14,165,233,0.02))',
  };

  const stat: React.CSSProperties = {
    backgroundColor: 'var(--bg-surface)',
    border: '1px solid var(--border-subtle)',
    borderRadius: '8px',
    padding: '12px 14px',
  };

  const statLabel: React.CSSProperties = {
    fontSize: '10px',
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
    color: 'var(--text-muted)',
    marginBottom: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  };

  const statValue: React.CSSProperties = {
    fontSize: '15px',
    fontWeight: 700,
    color: 'var(--text-main)',
    lineHeight: 1.2,
  };

  const statSub: React.CSSProperties = {
    fontSize: '11px',
    color: 'var(--text-secondary)',
    marginTop: '2px',
  };

  return (
    <div style={panel}>
      {/* ── Header ── */}
      <div style={header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '8px',
            background: 'linear-gradient(135deg, var(--primary), var(--accent-cyan))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Activity size={16} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-main)' }}>
              Nodule #{noduleData.id.slice(-4)}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              {noduleData.location} · Slice {noduleData.slice}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Risk badge */}
          <span style={{
            padding: '3px 10px',
            borderRadius: '20px',
            fontSize: '11px',
            fontWeight: 700,
            background: isHighRisk ? 'rgba(239,68,68,0.12)' : 'rgba(34,197,94,0.12)',
            color: riskColor,
            border: `1px solid ${isHighRisk ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`,
          }}>
            {riskLabel}
          </span>

          {onClose && (
            <button
              onClick={onClose}
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                padding: '4px',
              }}
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* ── Tabs ── */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '0 18px',
        gap: '4px',
      }}>
        {(Object.keys(tabLabels) as Tab[]).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid var(--accent-cyan)' : '2px solid transparent',
              color: activeTab === tab ? 'var(--accent-cyan)' : 'var(--text-muted)',
              padding: '10px 12px',
              fontSize: '12px',
              fontWeight: activeTab === tab ? 700 : 500,
              cursor: 'pointer',
              transition: 'color 0.15s ease',
            }}
          >
            {tabLabels[tab]}
          </button>
        ))}
      </div>

      {/* ── Content ── */}
      <div style={{ padding: '16px 18px' }}>

        {/* ── Tab: Overview ── */}
        {activeTab === 'summary' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

            {/* Risk meter row */}
            <div style={{
              ...stat,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div>
                <div style={{ ...statLabel }}><Activity size={10} /> Malignancy Risk</div>
                <div style={{ ...statValue, fontSize: '22px', color: riskColor }}>
                  {(risk * 100).toFixed(1)}%
                </div>
                <div style={statSub}>
                  95% CI: {(noduleData.ci_lower * 100).toFixed(0)}% – {(noduleData.ci_upper * 100).toFixed(0)}%
                  &nbsp;·&nbsp;{noduleData.uncertainty_level} uncertainty
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ ...statLabel, justifyContent: 'flex-end' }}>Reliability</div>
                <div style={{
                  fontSize: '13px', fontWeight: 700,
                  color: noduleData.disagreement_entropy > 0.4
                    ? 'var(--risk-high)'
                    : noduleData.disagreement_entropy > 0.25
                    ? 'var(--accent-amber)'
                    : 'var(--risk-low)',
                }}>
                  {noduleData.disagreement_entropy > 0.4 ? 'Low' : noduleData.disagreement_entropy > 0.25 ? 'Moderate' : 'High'}
                </div>
              </div>
            </div>

            {/* Stats grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <div style={stat}>
                <div style={statLabel}><MapPin size={10} /> Location</div>
                <div style={statValue}>{noduleData.location}</div>
                <div style={statSub}>Axial slice #{noduleData.slice}</div>
              </div>

              <div style={stat}>
                <div style={statLabel}><Ruler size={10} /> Size</div>
                <div style={statValue}>{noduleData.diameter_mm.toFixed(1)} mm</div>
                <div style={statSub}>{noduleData.volume_mm3.toFixed(0)} mm³ volume</div>
              </div>

              <div style={stat}>
                <div style={statLabel}><Layers size={10} /> Attenuation</div>
                <div style={statValue} className="capitalize">
                  {noduleData.attenuation.replace('_', '-')}
                </div>
                <div style={statSub}>{noduleData.margin} margin</div>
              </div>

              <div style={stat}>
                <div style={statLabel}><Activity size={10} /> Morphology</div>
                <div style={statValue}>SI {noduleData.spiculation_index.toFixed(2)}</div>
                <div style={statSub}>Sphericity {noduleData.sphericity.toFixed(2)}</div>
              </div>
            </div>

            {/* Low reliability warning */}
            {noduleData.disagreement_entropy > 0.4 && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: 'rgba(239,68,68,0.08)',
                border: '1px solid rgba(239,68,68,0.3)',
                borderRadius: '8px',
                padding: '10px 12px',
                fontSize: '12px',
                color: 'var(--risk-high)',
              }}>
                <AlertTriangle size={14} style={{ flexShrink: 0 }} />
                <span>Model outputs diverge significantly — radiologist review required.</span>
              </div>
            )}
          </div>
        )}

        {/* ── Tab: Model Agreement ── */}
        {activeTab === 'ensemble' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px',
            }}>
              <span>Architecture breakdown</span>
              <span>Entropy: {noduleData.disagreement_entropy.toFixed(2)}</span>
            </div>

            {noduleData.models.map((mod, idx) => (
              <div key={idx} style={{
                ...stat,
                display: 'flex',
                flexDirection: 'column',
                gap: '6px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-main)' }}>{mod.name}</span>
                  <span style={{
                    fontSize: '13px', fontWeight: 700,
                    color: mod.calibrated > 0.5 ? 'var(--risk-high)' : 'var(--risk-low)',
                  }}>
                    {(mod.calibrated * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-bar"
                    style={{
                      width: `${mod.calibrated * 100}%`,
                      background: mod.calibrated > 0.5
                        ? 'linear-gradient(90deg, var(--accent-amber), var(--risk-high))'
                        : 'linear-gradient(90deg, var(--accent-cyan), var(--risk-low))',
                    }}
                  />
                </div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>Uncertainty: {mod.uncertainty}</span>
                  <span>Raw: {(mod.prob * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
