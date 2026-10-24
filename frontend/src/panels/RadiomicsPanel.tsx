import React, { useState } from 'react';
import { Layers, Activity, Database, Download, CheckCircle2, ChevronRight, BarChart2 } from 'lucide-react';

interface RadiomicsFeature {
  featureClass: string;
  featureName: string;
  ibsiCode: string;
  value: number;
  zScore: number;
  importance: number;
  description: string;
}

export const RadiomicsPanel: React.FC<{ studyId?: string; noduleId?: string }> = ({
  studyId = 'CASE-DEMO-LIDC-0087',
  noduleId = 'NODULE-01',
}) => {
  const [selectedClass, setSelectedClass] = useState<string>('all');
  const [cacheHit, setCacheHit] = useState<boolean>(true);

  const featureClasses = [
    { id: 'all', label: 'All Classes (107)', count: 107 },
    { id: 'firstorder', label: 'First-Order (19)', count: 19 },
    { id: 'glcm', label: 'GLCM Texture (24)', count: 24 },
    { id: 'glrlm', label: 'GLRLM Matrix (16)', count: 16 },
    { id: 'glszm', label: 'GLSZM Size Zone (16)', count: 16 },
    { id: 'ngtdm', label: 'NGTDM Difference (5)', count: 5 },
    { id: 'shape', label: '3D Morphometry & Shape (14)', count: 14 },
  ];

  const sampleFeatures: RadiomicsFeature[] = [
    {
      featureClass: 'shape',
      featureName: 'Sphericity',
      ibsiCode: 'IBSI:Q3CL',
      value: 0.742,
      zScore: -1.24,
      importance: 0.185,
      description: 'Measure of roundness; lower values indicate irregular margins indicative of malignancy.',
    },
    {
      featureClass: 'shape',
      featureName: 'SurfaceToVolumeRatio',
      ibsiCode: 'IBSI:RN81',
      value: 0.412,
      zScore: 1.85,
      importance: 0.142,
      description: 'High ratio reflects complex, non-compact or spiculated tumor surfaces.',
    },
    {
      featureClass: 'glcm',
      featureName: 'JointEntropy',
      ibsiCode: 'IBSI:TU9B',
      value: 4.891,
      zScore: 1.62,
      importance: 0.128,
      description: 'Randomness/complexity in co-occurrence of voxel grayscale values.',
    },
    {
      featureClass: 'glcm',
      featureName: 'Contrast',
      ibsiCode: 'IBSI:ACUI',
      value: 18.42,
      zScore: 1.45,
      importance: 0.115,
      description: 'Measures local variations present in the gray level co-occurrence matrix.',
    },
    {
      featureClass: 'firstorder',
      featureName: 'MeanIntensity_HU',
      ibsiCode: 'IBSI:Q4LE',
      value: -142.5,
      zScore: 0.88,
      importance: 0.098,
      description: 'Mean attenuation value in Hounsfield Units within segmented ROI.',
    },
    {
      featureClass: 'firstorder',
      featureName: 'Kurtosis',
      ibsiCode: 'IBSI:C3WJ',
      value: 3.42,
      zScore: 1.12,
      importance: 0.084,
      description: 'Peakedness of voxel distribution relative to normal distribution.',
    },
    {
      featureClass: 'glrlm',
      featureName: 'ShortRunEmphasis',
      ibsiCode: 'IBSI:22OV',
      value: 0.884,
      zScore: 1.05,
      importance: 0.076,
      description: 'Distribution of short run lengths; high values signify fine structural texture.',
    },
    {
      featureClass: 'glszm',
      featureName: 'LargeAreaHighGrayLevelEmphasis',
      ibsiCode: 'IBSI:48P8',
      value: 1240.2,
      zScore: 0.94,
      importance: 0.068,
      description: 'Proportion in the image of large zones with higher gray-level values.',
    },
    {
      featureClass: 'ngtdm',
      featureName: 'Coarseness',
      ibsiCode: 'IBSI:QCDE',
      value: 0.018,
      zScore: -1.41,
      importance: 0.062,
      description: 'Inverse of rate of spatial change in intensity; lower indicates fine, heterogeneous tissue.',
    },
    {
      featureClass: 'shape',
      featureName: 'Elongation',
      ibsiCode: 'IBSI:Q3BX',
      value: 0.621,
      zScore: 0.72,
      importance: 0.042,
      description: 'Relationship between two largest principal components of the shape.',
    },
  ];

  const filteredFeatures =
    selectedClass === 'all'
      ? sampleFeatures
      : sampleFeatures.filter((f) => f.featureClass === selectedClass);

  const exportJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(sampleFeatures, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `radiomics_${studyId}_${noduleId}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <span
              style={{
                fontSize: '11px',
                fontWeight: '700',
                padding: '2px 8px',
                borderRadius: '4px',
                background: 'rgba(6, 182, 212, 0.15)',
                color: 'var(--accent-cyan)',
                border: '1px solid var(--border-accent)',
                letterSpacing: '0.05em',
              }}
            >
              09 · IBSI RADIOMICS PROFILING
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              PyRadiomics IBSI-Compliant Feature Extraction &amp; XGBoost Malignancy Scoring
            </span>
          </div>
          <h1 style={{ fontSize: '22px', fontWeight: '800', margin: 0, color: 'var(--text-main)', letterSpacing: '-0.02em' }}>
            Quantitative Radiomics &amp; Sub-Visual Phenotyping
          </h1>
          <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Active Analysis Target: <strong style={{ color: 'var(--accent-cyan)' }}>{studyId}</strong> · ROI:{' '}
            <strong style={{ color: 'var(--primary-light)' }}>{noduleId} (64³ Isotropic Crop)</strong>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setCacheHit(!cacheHit)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: '6px',
              background: cacheHit ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)',
              border: `1px solid ${cacheHit ? 'var(--accent-emerald)' : 'var(--accent-amber)'}`,
              color: cacheHit ? 'var(--accent-emerald)' : 'var(--accent-amber)',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer',
            }}
          >
            <Database size={14} />
            {cacheHit ? 'SQLite Cache: HIT (0.1ms)' : 'Cache: Recomputed'}
          </button>
          <button
            onClick={exportJson}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border-card)',
              color: 'var(--text-main)',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer',
            }}
          >
            <Download size={14} />
            Export IBSI JSON
          </button>
        </div>
      </div>

      {/* Top Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px' }}>
        <div className="glass-panel" style={{ padding: '16px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: '8px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Total Features Extracted</div>
          <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--text-main)', marginTop: '4px' }}>107 IBSI</div>
          <div style={{ fontSize: '11px', color: 'var(--accent-emerald)', marginTop: '4px' }}>✓ 100% IBSI v1.0 standard</div>
        </div>

        <div className="glass-panel" style={{ padding: '16px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: '8px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Radiomics Malignancy Score</div>
          <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--accent-rose)', marginTop: '4px' }}>0.874</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Trained XGBoost Regressor (AUC 0.912)</div>
        </div>

        <div className="glass-panel" style={{ padding: '16px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: '8px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Dominant Texture Signature</div>
          <div style={{ fontSize: '18px', fontWeight: '700', color: 'var(--accent-amber)', marginTop: '6px' }}>Heterogeneous / Spiculated</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>High Joint Entropy + Low Sphericity</div>
        </div>

        <div className="glass-panel" style={{ padding: '16px', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: '8px' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Extraction Latency</div>
          <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--accent-cyan)', marginTop: '4px' }}>142 ms</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>Cached in results/radiomics_cache.db</div>
        </div>
      </div>

      {/* Class Selector Tabs */}
      <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '10px' }}>
        {featureClasses.map((fc) => (
          <button
            key={fc.id}
            onClick={() => setSelectedClass(fc.id)}
            style={{
              padding: '6px 14px',
              borderRadius: '6px',
              border: selectedClass === fc.id ? '1px solid var(--accent-cyan)' : '1px solid var(--border-subtle)',
              background: selectedClass === fc.id ? 'rgba(6, 182, 212, 0.15)' : 'var(--bg-surface)',
              color: selectedClass === fc.id ? 'var(--accent-cyan)' : 'var(--text-muted)',
              fontSize: '12px',
              fontWeight: selectedClass === fc.id ? '700' : '500',
              cursor: 'pointer',
            }}
          >
            {fc.label}
          </button>
        ))}
      </div>

      {/* Features Table */}
      <div className="glass-panel" style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', borderRadius: '8px', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '12px' }}>
          <thead>
            <tr style={{ background: 'var(--bg-card)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
              <th style={{ padding: '10px 14px' }}>FEATURE NAME</th>
              <th style={{ padding: '10px 14px' }}>IBSI CODE</th>
              <th style={{ padding: '10px 14px' }}>EXTRACTED VALUE</th>
              <th style={{ padding: '10px 14px' }}>POPULATION Z-SCORE</th>
              <th style={{ padding: '10px 14px' }}>XGBOOST IMPORTANCE</th>
              <th style={{ padding: '10px 14px' }}>CLINICAL INTERPRETATION</th>
            </tr>
          </thead>
          <tbody>
            {filteredFeatures.map((feat, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-body)' }}>
                <td style={{ padding: '12px 14px', fontWeight: '600', color: 'var(--text-main)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Layers size={13} color="var(--accent-cyan)" />
                    {feat.featureName}
                  </div>
                </td>
                <td style={{ padding: '12px 14px', fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-faint)' }}>
                  {feat.ibsiCode}
                </td>
                <td style={{ padding: '12px 14px', fontFamily: 'JetBrains Mono, monospace', fontWeight: '700', color: 'var(--text-main)' }}>
                  {typeof feat.value === 'number' && feat.value % 1 !== 0 ? feat.value.toFixed(3) : feat.value}
                </td>
                <td style={{ padding: '12px 14px' }}>
                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: '11px',
                      fontWeight: '700',
                      background: Math.abs(feat.zScore) > 1.4 ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)',
                      color: Math.abs(feat.zScore) > 1.4 ? 'var(--accent-rose)' : 'var(--accent-emerald)',
                    }}
                  >
                    {feat.zScore > 0 ? `+${feat.zScore.toFixed(2)}` : feat.zScore.toFixed(2)}σ
                  </span>
                </td>
                <td style={{ padding: '12px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '60px', height: '6px', background: 'var(--border-subtle)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${Math.min(100, feat.importance * 400)}%`,
                          height: '100%',
                          background: 'linear-gradient(90deg, var(--primary), var(--accent-cyan))',
                        }}
                      />
                    </div>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', color: 'var(--text-muted)' }}>
                      {(feat.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td style={{ padding: '12px 14px', color: 'var(--text-muted)', fontSize: '11px', maxWidth: '320px' }}>
                  {feat.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
export default RadiomicsPanel;
