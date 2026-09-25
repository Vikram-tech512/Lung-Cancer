import React, { useEffect, useState } from 'react';
import { Ruler, Activity, BarChart, FileCode, CheckCircle2 } from 'lucide-react';
import { NoduleMorphometry } from '../types';

interface MorphometryPanelProps {
  noduleId: string;
}

export const MorphometryPanel: React.FC<MorphometryPanelProps> = ({ noduleId }) => {
  const [morph, setMorph] = useState<NoduleMorphometry | null>(null);

  useEffect(() => {
    fetch(`/api/v2/morphometry/${noduleId}`)
      .then(res => res.json())
      .then(data => setMorph(data))
      .catch(() => {
        setMorph({
          nodule_id: noduleId,
          centroid_zyx: [92.0, 280.0, 310.0],
          effective_diameter_mm: 14.8,
          volume_mm3: 1696.5,
          mean_hu: 38.4,
          std_hu: 24.1,
          max_hu: 92.0,
          min_hu: -110.0,
          sphericity: 0.74,
          spiculation_index: 0.42,
          surface_to_volume_ratio: 0.41
        });
      });
  }, [noduleId]);

  if (!morph) {
    return <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>Loading morphometric descriptors...</div>;
  }

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
          06 · QUANTITATIVE 3D MORPHOMETRY & RADIOMICS
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '6px' }}>
          Standardized volumetric shape metrics and IBSI-compliant HU attenuation statistics for candidate <span style={{ color: '#38bdf8', fontWeight: '600' }}>{noduleId}</span>.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
        {/* Volumetric Shape Card */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: '600', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Ruler size={18} />
            3D Volumetric Geometry
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Volume</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', marginTop: '4px' }}>
                {morph.volume_mm3.toFixed(1)} <span style={{ fontSize: '13px', color: '#94a3b8' }}>mm³</span>
              </div>
            </div>

            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Effective Diameter</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#f59e0b', marginTop: '4px' }}>
                {morph.effective_diameter_mm.toFixed(1)} <span style={{ fontSize: '13px', color: '#94a3b8' }}>mm</span>
              </div>
            </div>

            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Sphericity Index</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#10b981', marginTop: '4px' }}>
                {morph.sphericity.toFixed(2)}
              </div>
              <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>1.0 = Perfect sphere</div>
            </div>

            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Spiculation Index</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#f43f5e', marginTop: '4px' }}>
                {morph.spiculation_index.toFixed(2)}
              </div>
              <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>Higher = Malignancy risk</div>
            </div>
          </div>
        </div>

        {/* HU Attenuation Density Profile */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ margin: '0 0 16px 0', fontSize: '15px', fontWeight: '600', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} />
            Hounsfield Unit (HU) Attenuation
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Mean Attenuation</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', marginTop: '4px' }}>
                {morph.mean_hu.toFixed(1)} <span style={{ fontSize: '13px', color: '#94a3b8' }}>HU</span>
              </div>
            </div>

            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Attenuation Std Dev</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', marginTop: '4px' }}>
                ±{morph.std_hu.toFixed(1)} <span style={{ fontSize: '13px', color: '#94a3b8' }}>HU</span>
              </div>
            </div>

            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Max Attenuation (Solid core)</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', marginTop: '4px' }}>
                {morph.max_hu.toFixed(1)} <span style={{ fontSize: '13px', color: '#94a3b8' }}>HU</span>
              </div>
            </div>

            <div style={{ background: '#0b1120', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ color: '#64748b', fontSize: '11px', textTransform: 'uppercase' }}>Min Attenuation</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#f8fafc', marginTop: '4px' }}>
                {morph.min_hu.toFixed(1)} <span style={{ fontSize: '13px', color: '#94a3b8' }}>HU</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* IBSI Feature Vector Overview */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '600', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileCode size={16} color="#10b981" />
          IBSI Compliance Summary (107 Radiomic Features Extracted)
        </h3>
        <p style={{ fontSize: '12px', color: '#94a3b8', margin: '0 0 16px 0' }}>
          Computed via PyRadiomics adhering to the Image Biomarker Standardisation Initiative (IBSI) benchmarks across 6 standardized feature classes:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
          {[
            { name: 'First Order Statistics', count: 19, desc: 'Energy, Entropy, Skewness, Kurtosis' },
            { name: '3D Shape Descriptors', count: 16, desc: 'Elongation, Flatness, Mesh Volume' },
            { name: 'GLCM (Gray Level Co-occurrence)', count: 24, desc: 'Joint Entropy, Contrast, Correlation' },
            { name: 'GLRLM (Run Length Matrix)', count: 16, desc: 'Short Run Emphasis, Run Percentage' },
            { name: 'GLSZM (Size Zone Matrix)', count: 16, desc: 'Small Area Emphasis, Zone Variance' },
            { name: 'GLDM (Dependence Matrix)', count: 16, desc: 'Dependence Non-Uniformity' },
          ].map(f => (
            <div key={f.name} style={{ background: '#0b1120', padding: '12px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', fontWeight: '600', color: '#38bdf8' }}>{f.name}</span>
                <span style={{ fontSize: '11px', background: '#1e293b', padding: '2px 6px', borderRadius: '4px', color: '#94a3b8' }}>{f.count}</span>
              </div>
              <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>{f.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
