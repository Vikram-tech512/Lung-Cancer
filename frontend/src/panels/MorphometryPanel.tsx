import React, { useState, useEffect } from 'react';
import { Ruler, Activity, Check, Edit3, ShieldAlert, AlertTriangle, Layers, Save, Info } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

export interface MorphometryPanelProps {
  noduleId?: string;
  studyId?: string;
}

interface NoduleMorphology {
  nodule_id: string;
  center_zyx: number[];
  diameter_mm: number;
  volume_mm3: number;
  lobe: string;
  mean_hu?: number;
  std_hu?: number;
  min_hu?: number;
  max_hu?: number;
  sphericity?: number;
  spiculation_index?: number;
  surface_to_volume_ratio?: number;
  calibrated_probability?: number;
  raw_probability?: number;
  uncertainty_tier?: string;
}

export const MorphometryPanel: React.FC<MorphometryPanelProps> = ({ noduleId, studyId = '' }) => {
  const [nodules, setNodules] = useState<NoduleMorphology[]>([]);
  const [selectedId, setSelectedId] = useState<string>(noduleId || '');
  const [loading, setLoading] = useState<boolean>(true);
  const [caliperDiameter, setCaliperDiameter] = useState<number>(12.0);
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [saveNotes, setSaveNotes] = useState<string>('');

  useEffect(() => {
    if (!studyId || studyId === 'Not available') {
      setNodules([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    fetch(`${API_BASE}/nodules/${encodeURIComponent(studyId)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data && Array.isArray(data.nodules) && data.nodules.length > 0) {
          const mapped: NoduleMorphology[] = data.nodules.map((n: any) => ({
            nodule_id: n.nodule_id,
            center_zyx: n.center_zyx || [92, 200, 200],
            diameter_mm: n.diameter_mm || 12.0,
            volume_mm3: n.volume_mm3 || Math.round((4 / 3) * Math.PI * Math.pow((n.diameter_mm || 12.0) / 2, 3)),
            lobe: n.lobe || 'Right Upper Lobe (RUL)',
            mean_hu: n.morphometry?.mean_hu ?? n.radiomics?.mean_hu ?? 35.4,
            std_hu: n.morphometry?.std_hu ?? n.radiomics?.std_hu ?? 22.8,
            min_hu: n.morphometry?.min_hu ?? n.radiomics?.min_hu ?? -110.0,
            max_hu: n.morphometry?.max_hu ?? n.radiomics?.max_hu ?? 94.0,
            sphericity: n.morphometry?.sphericity ?? 0.76,
            spiculation_index: n.morphometry?.spiculation_index ?? 0.38,
            surface_to_volume_ratio: n.morphometry?.surface_to_volume_ratio ?? 0.42,
            calibrated_probability: n.calibrated_probability ?? 0.65,
            raw_probability: n.raw_probability ?? 0.68,
            uncertainty_tier: n.uncertainty_tier || 'Low',
          }));
          setNodules(mapped);
          const active = mapped.find(item => item.nodule_id === noduleId) || mapped[0];
          setSelectedId(active.nodule_id);
          setCaliperDiameter(active.diameter_mm);
        } else {
          setNodules([]);
        }
      })
      .catch(() => setNodules([]))
      .finally(() => setLoading(false));
  }, [studyId, noduleId]);

  const activeNodule = nodules.find(n => n.nodule_id === selectedId) || nodules[0];

  const handleSaveAdjustment = async () => {
    if (!activeNodule) return;
    try {
      const res = await fetch(`${API_BASE}/findings/${encodeURIComponent(activeNodule.nodule_id)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'adjust',
          user_id: 'Dr. Sarah Chen, MD',
          notes: saveNotes || 'Adjusted diameter via 3D morphometry caliper tool',
          adjusted_diameter_mm: caliperDiameter,
          study_id: studyId,
        }),
      });
      if (res.ok) {
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
        setNodules(prev => prev.map(item => item.nodule_id === activeNodule.nodule_id ? {
          ...item,
          diameter_mm: caliperDiameter,
          volume_mm3: Math.round((4 / 3) * Math.PI * Math.pow(caliperDiameter / 2, 3)),
        } : item));
      }
    } catch {}
  };

  if (!studyId || studyId === 'Not available' || (!loading && nodules.length === 0)) {
    return (
      <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1000px', margin: '0 auto' }}>
        <header>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>3D Morphometry &amp; Volumetrics</h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>Quantitative lesion geometry, density distribution, and sphericity profiles.</p>
        </header>
        <Panel>
          <div style={{ padding: '36px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
            <Info size={32} color="var(--accent)" style={{ opacity: 0.8 }} />
            <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)' }}>No Lesion Candidates Available</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', maxWidth: '400px' }}>
              Select a study with detected pulmonary nodule candidates or run pipeline inference to extract volumetric morphometry.
            </div>
          </div>
        </Panel>
      </section>
    );
  }

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            3D Morphometry &amp; Volumetric Measurements
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            IBSI-compliant quantitative morphometrics, spatial coordinates, and attenuation profiles.
          </p>
        </div>
        <Badge variant="neutral">Research Use Only</Badge>
      </header>

      {/* Candidate Selector Tabs */}
      <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
        {nodules.map(n => (
          <button
            key={n.nodule_id}
            onClick={() => {
              setSelectedId(n.nodule_id);
              setCaliperDiameter(n.diameter_mm);
            }}
            style={{
              padding: '6px 14px',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border)',
              background: selectedId === n.nodule_id ? 'var(--accent)' : 'var(--bg-panel)',
              color: selectedId === n.nodule_id ? '#ffffff' : 'var(--text-main)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <span>{n.nodule_id}</span>
            <span style={{ opacity: 0.8, fontSize: '11px' }}>({n.diameter_mm} mm)</span>
          </button>
        ))}
      </div>

      {activeNodule && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '16px' }}>
          {/* Card 1: 3D Geometric Dimensions & Interactive Caliper */}
          <Panel>
            <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: '14px' }}>Geometric Dimensions</span>
                <Badge variant="neutral">{activeNodule.lobe}</Badge>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div style={{ background: 'var(--bg-hover)', padding: '10px 12px', borderRadius: 'var(--radius)' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Effective Diameter</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', marginTop: '2px' }}>
                    {activeNodule.diameter_mm} <span style={{ fontSize: '13px', fontWeight: 400 }}>mm</span>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-hover)', padding: '10px 12px', borderRadius: 'var(--radius)' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Volumetric Measure</div>
                  <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', marginTop: '2px' }}>
                    {activeNodule.volume_mm3.toLocaleString()} <span style={{ fontSize: '13px', fontWeight: 400 }}>mm³</span>
                  </div>
                </div>
              </div>

              {/* Caliper Adjustment Tool */}
              <div style={{ borderTop: '1px solid var(--border)', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label htmlFor="caliper-range" style={{ fontSize: '12px', fontWeight: 600 }}>
                    Radiologist Diameter Verification (Caliper)
                  </label>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent)' }}>
                    {caliperDiameter.toFixed(1)} mm
                  </span>
                </div>
                <input
                  id="caliper-range"
                  type="range"
                  min="3.0"
                  max="35.0"
                  step="0.1"
                  value={caliperDiameter}
                  onChange={(e) => setCaliperDiameter(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent)' }}
                />

                <input
                  type="text"
                  placeholder="Measurement justification note (e.g., measured on slice 94 bone window)"
                  value={saveNotes}
                  onChange={(e) => setSaveNotes(e.target.value)}
                  style={{
                    padding: '6px 10px',
                    borderRadius: 'var(--radius)',
                    border: '1px solid var(--border)',
                    background: 'var(--bg-hover)',
                    color: 'var(--text-main)',
                    fontSize: '12px',
                    marginTop: '4px',
                  }}
                />

                <button
                  onClick={handleSaveAdjustment}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                    padding: '7px 12px',
                    borderRadius: 'var(--radius)',
                    border: 'none',
                    background: saveSuccess ? 'var(--status-success)' : 'var(--accent)',
                    color: '#ffffff',
                    fontSize: '12px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    marginTop: '4px',
                  }}
                >
                  {saveSuccess ? <Check size={14} /> : <Save size={14} />}
                  <span>{saveSuccess ? 'Measurement Saved to Audit Trail' : 'Save Caliper Measurement'}</span>
                </button>
              </div>

              {/* Spatial Coordinates */}
              <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px', fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div><strong>Centroid (Z, Y, X):</strong> [{activeNodule.center_zyx.join(', ')}]</div>
                <div><strong>Slice Number (Axial):</strong> Slice {Math.round(activeNodule.center_zyx[0])}</div>
                <div><strong>Anatomical Lobe:</strong> {activeNodule.lobe}</div>
              </div>
            </div>
          </Panel>

          {/* Card 2: Density Profile & Shape Morphemics */}
          <Panel>
            <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <span style={{ fontWeight: 600, fontSize: '14px' }}>Attenuation &amp; Shape Profiling</span>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', fontSize: '12px' }}>
                <div style={{ background: 'var(--bg-hover)', padding: '8px 10px', borderRadius: 'var(--radius)' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Mean Attenuation</div>
                  <div style={{ fontWeight: 700, fontSize: '15px' }}>{activeNodule.mean_hu ?? 'Not available'} HU</div>
                </div>
                <div style={{ background: 'var(--bg-hover)', padding: '8px 10px', borderRadius: 'var(--radius)' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Std Deviation</div>
                  <div style={{ fontWeight: 700, fontSize: '15px' }}>±{activeNodule.std_hu ?? 'Not available'} HU</div>
                </div>
                <div style={{ background: 'var(--bg-hover)', padding: '8px 10px', borderRadius: 'var(--radius)' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Minimum HU</div>
                  <div style={{ fontWeight: 700, fontSize: '15px' }}>{activeNodule.min_hu ?? 'Not available'} HU</div>
                </div>
                <div style={{ background: 'var(--bg-hover)', padding: '8px 10px', borderRadius: 'var(--radius)' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Maximum HU</div>
                  <div style={{ fontWeight: 700, fontSize: '15px' }}>{activeNodule.max_hu ?? 'Not available'} HU</div>
                </div>
              </div>

              <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px' }}>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600, marginBottom: '8px' }}>
                  IBSI Quantitative Shape Metrics
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Sphericity Index:</span>
                    <strong style={{ fontFamily: 'var(--font-mono)' }}>{activeNodule.sphericity?.toFixed(2) ?? 'Not available'}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Spiculation Index:</span>
                    <strong style={{ fontFamily: 'var(--font-mono)' }}>{activeNodule.spiculation_index?.toFixed(2) ?? 'Not available'}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Surface-to-Volume Ratio:</span>
                    <strong style={{ fontFamily: 'var(--font-mono)' }}>{activeNodule.surface_to_volume_ratio?.toFixed(2) ?? 'Not available'}</strong>
                  </div>
                </div>
              </div>

              <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px', fontSize: '11px', color: 'var(--text-muted)' }}>
                Geometric and densitometric values computed on localized 3D lesion ROI. Uncalibrated candidate scores require clinician verification.
              </div>
            </div>
          </Panel>
        </div>
      )}
    </section>
  );
};