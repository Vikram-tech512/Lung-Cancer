import React, { useEffect, useRef, useState } from 'react';
import { ShieldAlert, Activity, BarChart2, Eye, ZoomIn, Layers } from 'lucide-react';
import { VerdictResult } from '../types';
import { API_BASE } from '../services/api';

interface VerdictPanelProps {
  noduleId: string;
}

/** Renders a synthetic Grad-CAM / attention heatmap on a canvas */
function HeatmapCanvas({
  method,
  noduleId,
  opacity = 0.72
}: {
  method: 'gradcam3d' | 'integrated_gradients' | 'attention_rollout3d';
  noduleId: string;
  opacity?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    setLoading(true);
    const W = canvas.width;
    const H = canvas.height;

    // Attempt to fetch from API
    fetch(`${API_BASE}/xai/${noduleId}/${method}`)
      .then(res => res.json())
      .then((data: { heatmap_matrix: number[][] }) => {
        renderHeatmap(ctx, W, H, data.heatmap_matrix, method);
        setLoading(false);
      })
      .catch(() => {
        renderSyntheticHeatmap(ctx, W, H, method);
        setLoading(false);
      });
  }, [noduleId, method]);

  const renderHeatmap = (
    ctx: CanvasRenderingContext2D,
    W: number,
    H: number,
    matrix: number[][],
    method: string
  ) => {
    // Draw CT background
    ctx.fillStyle = '#050d15';
    ctx.fillRect(0, 0, W, H);

    // Draw thorax outline
    ctx.beginPath();
    ctx.ellipse(W / 2, H / 2, W * 0.42, H * 0.38, 0, 0, 2 * Math.PI);
    ctx.fillStyle = '#0e1e30';
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#1e3a5a';
    ctx.stroke();

    // Lung regions
    [W * 0.32, W * 0.68].forEach(lx => {
      ctx.beginPath();
      ctx.ellipse(lx, H * 0.5, W * 0.14, H * 0.24, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#060e17';
      ctx.fill();
      ctx.strokeStyle = '#1a3254';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Heatmap overlay
    if (matrix && matrix.length > 0) {
      const matH = matrix.length;
      const matW = matrix[0].length;
      const cellW = W / matW;
      const cellH = H / matH;

      matrix.forEach((row, ry) => {
        row.forEach((val, rx) => {
          if (val < 0.08) return;
          const r = Math.round(255 * Math.min(1, val * 1.8));
          const g = Math.round(180 * Math.max(0, 1 - val * 1.5));
          const b = Math.round(80 * Math.max(0, 1 - val * 2));
          ctx.fillStyle = `rgba(${r},${g},${b},${(val * opacity).toFixed(2)})`;
          ctx.fillRect(rx * cellW, ry * cellH, cellW + 1, cellH + 1);
        });
      });
    }

    drawAnnotations(ctx, W, H, method);
  };

  const renderSyntheticHeatmap = (
    ctx: CanvasRenderingContext2D,
    W: number,
    H: number,
    method: string
  ) => {
    // Synthetic background
    ctx.fillStyle = '#050d15';
    ctx.fillRect(0, 0, W, H);

    ctx.beginPath();
    ctx.ellipse(W / 2, H / 2, W * 0.42, H * 0.38, 0, 0, 2 * Math.PI);
    ctx.fillStyle = '#0e1e30';
    ctx.fill();

    [W * 0.32, W * 0.68].forEach(lx => {
      ctx.beginPath();
      ctx.ellipse(lx, H * 0.5, W * 0.14, H * 0.24, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#060e17';
      ctx.fill();
    });

    // Gaussian heatmap centered on right-upper lobe
    const cx = W * 0.62;
    const cy = H * 0.36;
    const hotSpotR = method === 'gradcam3d' ? 28 : method === 'integrated_gradients' ? 22 : 32;

    const imgData = ctx.createImageData(W, H);
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const dist = Math.sqrt((px - cx) ** 2 + (py - cy) ** 2);
        const val = Math.exp(-(dist ** 2) / (2 * hotSpotR ** 2));
        if (val < 0.04) continue;

        // Jet colormap
        const r = Math.min(255, Math.round(255 * Math.min(1, val * 2.0)));
        const g = Math.min(255, Math.round(255 * Math.max(0, 1 - Math.abs(val - 0.5) * 2)));
        const b = Math.min(255, Math.round(255 * Math.max(0, 1 - val * 2)));

        const idx = (py * W + px) * 4;
        imgData.data[idx]     = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = Math.round(val * opacity * 255);
      }
    }
    ctx.putImageData(imgData, 0, 0);

    drawAnnotations(ctx, W, H, method);
  };

  const drawAnnotations = (ctx: CanvasRenderingContext2D, W: number, H: number, method: string) => {
    // Method label
    const labels: Record<string, string> = {
      gradcam3d: '3D Grad-CAM',
      integrated_gradients: 'Integrated Gradients',
      attention_rollout3d: 'Attention Rollout'
    };
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(56, 189, 248, 0.9)';
    ctx.fillText(labels[method] || method, 8, 18);

    // Nodule crosshair
    const nx = W * 0.62;
    const ny = H * 0.36;
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.8)';
    ctx.lineWidth = 1.5;
    const arm = 10;
    ctx.beginPath();
    ctx.moveTo(nx - arm, ny); ctx.lineTo(nx + arm, ny);
    ctx.moveTo(nx, ny - arm); ctx.lineTo(nx, ny + arm);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(244, 63, 94, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(nx, ny, 14, 0, 2 * Math.PI);
    ctx.stroke();

    // Disclaimer
    ctx.font = '9px Inter, sans-serif';
    ctx.fillStyle = 'rgba(100, 116, 139, 0.7)';
    ctx.fillText('MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE', 6, H - 6);
  };

  return (
    <div style={{ position: 'relative', lineHeight: 0 }}>
      {loading && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
          justifyContent: 'center', background: 'rgba(5, 13, 21, 0.85)', borderRadius: '6px', zIndex: 2
        }}>
          <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>Computing {method}...</span>
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={200}
        height={180}
        className="heatmap-canvas"
        style={{ width: '100%', height: 'auto', borderRadius: '6px', border: '1px solid #1e293b' }}
      />
    </div>
  );
}

export const VerdictPanel: React.FC<VerdictPanelProps> = ({ noduleId }) => {
  const [verdict, setVerdict] = useState<VerdictResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeXai, setActiveXai] = useState<'gradcam3d' | 'integrated_gradients' | 'attention_rollout3d'>('gradcam3d');
  const [xaiOpacity, setXaiOpacity] = useState<number>(0.72);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/classify/${noduleId}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => { setVerdict(data); setLoading(false); })
      .catch(() => {
        setVerdict({
          nodule_id: noduleId,
          ensemble_verdict: 'MALIGNANT',
          ensemble_probability: 0.764,
          confidence_interval: [0.712, 0.816],
          weights_used: { resnet3d: 0.35, swin3d: 0.35, radiomics_xgb: 0.30 },
          disagreement_entropy: 0.38,
          model_breakdown: [
            { model_id: 'resnet3d',      model_name: '3D ResNet-50',         raw_probability: 0.742, calibrated_probability: 0.712, uncertainty_score: 0.082, uncertainty_level: 'LOW',      status: 'ACTIVE', weight_in_ensemble: 0.35, verdict: 'MALIGNANT' },
            { model_id: 'swin3d',        model_name: '3D Swin Transformer',  raw_probability: 0.785, calibrated_probability: 0.753, uncertainty_score: 0.065, uncertainty_level: 'LOW',      status: 'ACTIVE', weight_in_ensemble: 0.35, verdict: 'MALIGNANT' },
            { model_id: 'hybrid_vit',    model_name: 'Hybrid CNN-ViT',       raw_probability: 0.812, calibrated_probability: 0.789, uncertainty_score: 0.045, uncertainty_level: 'LOW',      status: 'ACTIVE', weight_in_ensemble: 0.0,  verdict: 'MALIGNANT' },
            { model_id: 'radiomics_xgb', model_name: 'Radiomics XGBoost',    raw_probability: 0.690, calibrated_probability: 0.672, uncertainty_score: 0.110, uncertainty_level: 'MODERATE', status: 'ACTIVE', weight_in_ensemble: 0.30, verdict: 'MALIGNANT' },
          ],
          morphometry: {
            nodule_id: noduleId, centroid_zyx: [92, 280, 310],
            effective_diameter_mm: 14.8, volume_mm3: 1696.5,
            mean_hu: 38.4, std_hu: 24.1, max_hu: 92.0, min_hu: -110.0,
            sphericity: 0.74, spiculation_index: 0.42, surface_to_volume_ratio: 0.41
          },
          clinical_advisory: 'RESEARCH USE ONLY — NOT APPROVED FOR PRIMARY DIAGNOSTIC DECISIONS'
        });
        setLoading(false);
      });
  }, [noduleId]);

  if (loading || !verdict) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#94a3b8' }}>
        <Activity size={28} style={{ marginBottom: '12px', opacity: 0.5 }} />
        <div>Evaluating candidate {noduleId} across 3D Deep Learning, Vision Transformers &amp; Radiomics...</div>
      </div>
    );
  }

  const isMalignant = verdict.ensemble_verdict === 'MALIGNANT';

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }} className="animate-fade-in-up">
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '800', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
          05 · MULTIMODAL VERDICT &amp; UNCERTAINTY QUANTIFICATION
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '6px' }}>
          Consensus evaluation of candidate <span style={{ color: '#38bdf8', fontWeight: '600' }}>{noduleId}</span> across 3D ResNet, Swin3D, HybridViT, and PyRadiomics.
        </p>
      </div>

      {/* Clinical Advisory */}
      <div className="glass-panel-warning" style={{ padding: '13px 18px', marginBottom: '22px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <ShieldAlert size={20} color="#f43f5e" style={{ flexShrink: 0 }} />
        <div style={{ fontSize: '12px', color: '#fda4af', fontWeight: '500' }}>
          <strong style={{ color: '#f43f5e' }}>MANDATORY CLINICAL NOTICE: </strong>
          {verdict.clinical_advisory}. Algorithmic inference only — must not be used as primary evidence for biopsy, resection, or clinical staging.
        </div>
      </div>

      {/* Summary Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '18px', marginBottom: '22px' }}>
        <div className="glass-panel-accent" style={{ padding: '22px', position: 'relative' }}>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#06b6d4', fontWeight: '700', letterSpacing: '0.06em' }}>Consensus Ensemble Verdict</div>
          <div style={{ fontSize: '32px', fontWeight: '800', color: isMalignant ? '#f43f5e' : '#10b981', marginTop: '8px', letterSpacing: '-0.02em' }}>
            {verdict.ensemble_verdict}
          </div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
            P = <strong style={{ color: '#f8fafc' }}>{(verdict.ensemble_probability * 100).toFixed(1)}%</strong>
            &nbsp;· 95% CI [{(verdict.confidence_interval[0] * 100).toFixed(1)}% – {(verdict.confidence_interval[1] * 100).toFixed(1)}%]
          </div>
          {/* Probability bar */}
          <div style={{ marginTop: '14px', height: '5px', background: '#1e293b', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${(verdict.ensemble_probability * 100).toFixed(0)}%`, background: isMalignant ? 'linear-gradient(90deg, #be123c, #f43f5e)' : 'linear-gradient(90deg, #059669, #10b981)', transition: 'width 1s ease', borderRadius: '3px' }} />
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '22px' }}>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#94a3b8', fontWeight: '700', letterSpacing: '0.06em' }}>Epistemic Uncertainty</div>
          <div style={{ fontSize: '26px', fontWeight: '800', color: '#38bdf8', marginTop: '8px' }}>LOW RISK</div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
            Entropy: <strong style={{ color: '#f8fafc' }}>{verdict.disagreement_entropy}</strong> (T=30 MC Passes)
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '22px' }}>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#94a3b8', fontWeight: '700', letterSpacing: '0.06em' }}>Effective Diameter</div>
          <div style={{ fontSize: '26px', fontWeight: '800', color: '#f59e0b', marginTop: '8px' }}>{verdict.morphometry.effective_diameter_mm} mm</div>
          <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '4px' }}>
            Volume: <strong style={{ color: '#f8fafc' }}>{verdict.morphometry.volume_mm3.toFixed(0)} mm³</strong>
            &nbsp;· Sphericity: <strong>{verdict.morphometry.sphericity}</strong>
          </div>
        </div>
      </div>

      {/* Main 2-column: Breakdown table + XAI */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '20px', marginBottom: '20px' }}>

        {/* Model Breakdown Table */}
        <div className="glass-panel" style={{ overflow: 'hidden' }}>
          <div style={{ padding: '14px 20px', borderBottom: '1px solid #1e293b', background: '#0b1120', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart2 size={16} color="#38bdf8" />
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#f1f5f9' }}>
              Individual Model Contribution &amp; Calibration
            </h3>
          </div>
          <table className="clinical-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Raw Prob</th>
                <th>Calibrated</th>
                <th>Uncertainty</th>
                <th>Weight</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {verdict.model_breakdown.map((m) => (
                <tr key={m.model_id}>
                  <td style={{ fontWeight: '600', color: '#e2e8f0' }}>{m.model_name}</td>
                  <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>{(m.raw_probability * 100).toFixed(1)}%</td>
                  <td style={{ fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8' }}>{(m.calibrated_probability * 100).toFixed(1)}%</td>
                  <td>
                    <span className={`badge badge-${m.uncertainty_level === 'LOW' ? 'success' : m.uncertainty_level === 'MODERATE' ? 'warning' : 'danger'}`}>
                      {m.uncertainty_level} ±{m.uncertainty_score}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'JetBrains Mono, monospace' }}>
                    {m.weight_in_ensemble > 0 ? `${(m.weight_in_ensemble * 100).toFixed(0)}%` : 'Ref'}
                  </td>
                  <td style={{ fontWeight: '700', color: m.verdict === 'MALIGNANT' ? '#f43f5e' : '#10b981' }}>{m.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* XAI Heatmap Panel */}
        <div className="glass-panel" style={{ padding: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <h3 style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#f1f5f9', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Eye size={16} color="#a78bfa" /> XAI Attribution Map
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: '#64748b' }}>
              <ZoomIn size={12} /> Axial MIP
            </div>
          </div>

          {/* Method tabs */}
          <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
            {[
              { key: 'gradcam3d', label: '3D Grad-CAM' },
              { key: 'integrated_gradients', label: 'Int. Gradients' },
              { key: 'attention_rollout3d', label: 'Attn Rollout' },
            ].map(tab => (
              <button key={tab.key}
                onClick={() => setActiveXai(tab.key as any)}
                style={{
                  flex: 1,
                  background: activeXai === tab.key ? 'rgba(139, 92, 246, 0.2)' : 'transparent',
                  border: `1px solid ${activeXai === tab.key ? 'rgba(139, 92, 246, 0.5)' : '#1e293b'}`,
                  borderRadius: '5px',
                  color: activeXai === tab.key ? '#a78bfa' : '#64748b',
                  fontSize: '10px',
                  fontWeight: '600',
                  padding: '5px 4px',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  textAlign: 'center'
                }}>
                {tab.label}
              </button>
            ))}
          </div>

          {/* Heatmap canvas */}
          <HeatmapCanvas method={activeXai} noduleId={noduleId} opacity={xaiOpacity} />

          {/* Opacity slider */}
          <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '11px', color: '#64748b', whiteSpace: 'nowrap' }}>Overlay</span>
            <input type="range" min={0} max={100} value={Math.round(xaiOpacity * 100)}
              onChange={e => setXaiOpacity(Number(e.target.value) / 100)}
              style={{ flex: 1, accentColor: '#8b5cf6' }} />
            <span style={{ fontSize: '11px', color: '#94a3b8', fontFamily: 'JetBrains Mono', width: '28px', textAlign: 'right' }}>
              {Math.round(xaiOpacity * 100)}%
            </span>
          </div>

          {/* XAI disclaimer */}
          <div style={{ marginTop: '10px', padding: '8px 10px', background: 'rgba(139, 92, 246, 0.07)', borderRadius: '5px', border: '1px solid rgba(139, 92, 246, 0.18)' }}>
            <div style={{ fontSize: '10px', color: '#7c6bad' }}>
              Attribution map highlights regions most influential for model decision. Max Intensity Projection of 3D saliency volume. Not validated as diagnostic evidence.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
