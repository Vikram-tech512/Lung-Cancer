import React, { useState, useRef, useEffect } from 'react';
import { Layers, Crosshair, ZoomIn, SunMedium, Eye, Zap, AlertCircle } from 'lucide-react';
import { NoduleItem } from '../types';
import { API_BASE } from '../services/api';

interface CTViewerPanelProps {
  studyId: string;
  onInspectNodule: (noduleId: string) => void;
}

export const CTViewerPanel: React.FC<CTViewerPanelProps> = ({ studyId, onInspectNodule }) => {
  const [activeLayer, setActiveLayer] = useState<'none' | 'segmentation' | 'gradcam3d' | 'attention_rollout3d' | 'integrated_gradients'>('segmentation');
  const [windowPreset, setWindowPreset] = useState<'lung' | 'mediastinum' | 'bone'>('lung');
  const [sliceIndex, setSliceIndex] = useState<number>(92);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [nodules, setNodules] = useState<NoduleItem[]>([]);
  const [selectedNodule, setSelectedNodule] = useState<string>('NODULE-01');

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/nodules/${studyId}`)
      .then(res => res.json())
      .then(data => {
        if (data.nodules && data.nodules.length > 0) {
          setNodules(data.nodules);
          setSelectedNodule(data.nodules[0].nodule_id);
          setSliceIndex(data.nodules[0].center_zyx[0]);
        }
      })
      .catch(() => {
        // demo nodule
        const demo: NoduleItem = {
          nodule_id: 'NODULE-01',
          center_zyx: [92, 280, 310],
          diameter_mm: 14.8,
          volume_mm3: 1696.5,
          lobe: 'Right Upper Lobe (RUL)',
          preliminary_risk: 'High Suspicion (Lung-RADS 4B)',
          morphometry: {
            nodule_id: 'NODULE-01',
            centroid_zyx: [92, 280, 310],
            effective_diameter_mm: 14.8,
            volume_mm3: 1696.5,
            mean_hu: 38.4,
            std_hu: 24.1,
            max_hu: 92.0,
            min_hu: -110.0,
            sphericity: 0.74,
            spiculation_index: 0.42,
            surface_to_volume_ratio: 0.41
          }
        };
        setNodules([demo]);
      });
  }, [studyId]);

  // Render high-fidelity canvas simulation representing 3D CT slice & overlays
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Background CT representation
    ctx.fillStyle = windowPreset === 'lung' ? '#0d1520' : '#141414';
    ctx.fillRect(0, 0, width, height);

    // Anatomical thoracic contour simulation
    ctx.beginPath();
    ctx.ellipse(width / 2, height / 2, width * 0.42, height * 0.38, 0, 0, 2 * Math.PI);
    ctx.fillStyle = '#1e293b';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#334155';
    ctx.stroke();

    // Left & Right Lung Parenchyma (low attenuation / darker)
    ctx.beginPath();
    ctx.ellipse(width * 0.33, height / 2, width * 0.16, height * 0.28, 0, 0, 2 * Math.PI);
    ctx.fillStyle = '#070c14';
    ctx.fill();

    ctx.beginPath();
    ctx.ellipse(width * 0.67, height / 2, width * 0.16, height * 0.28, 0, 0, 2 * Math.PI);
    ctx.fillStyle = '#070c14';
    ctx.fill();

    // Thoracic spine & ribs
    ctx.beginPath();
    ctx.arc(width / 2, height * 0.78, 16, 0, 2 * Math.PI);
    ctx.fillStyle = windowPreset === 'bone' ? '#f8fafc' : '#94a3b8';
    ctx.fill();

    // Nodule rendering at candidate coordinates
    const nx = width * 0.62;
    const ny = height * 0.45;
    const nRadius = 16;

    ctx.beginPath();
    ctx.arc(nx, ny, nRadius, 0, 2 * Math.PI);
    ctx.fillStyle = '#cbd5e1';
    ctx.fill();

    // Overlays
    if (activeLayer === 'segmentation') {
      // 3D U-Net / nnU-Net segmentation contour
      ctx.beginPath();
      ctx.arc(nx, ny, nRadius + 2, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(16, 185, 129, ${overlayOpacity * 0.5})`;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#10b981';
      ctx.stroke();
    } else if (activeLayer === 'gradcam3d' || activeLayer === 'attention_rollout3d' || activeLayer === 'integrated_gradients') {
      // XAI Heatmap radial gradient
      const grad = ctx.createRadialGradient(nx, ny, 2, nx, ny, 36);
      if (activeLayer === 'gradcam3d') {
        grad.addColorStop(0, `rgba(239, 68, 68, ${overlayOpacity})`);
        grad.addColorStop(0.5, `rgba(245, 158, 11, ${overlayOpacity * 0.7})`);
        grad.addColorStop(1, 'rgba(59, 130, 246, 0)');
      } else if (activeLayer === 'attention_rollout3d') {
        grad.addColorStop(0, `rgba(168, 85, 247, ${overlayOpacity})`);
        grad.addColorStop(0.6, `rgba(6, 182, 212, ${overlayOpacity * 0.6})`);
        grad.addColorStop(1, 'rgba(15, 23, 42, 0)');
      } else {
        // Integrated Gradients
        grad.addColorStop(0, `rgba(6, 182, 212, ${overlayOpacity})`);
        grad.addColorStop(0.6, `rgba(16, 185, 129, ${overlayOpacity * 0.5})`);
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      }
      ctx.beginPath();
      ctx.arc(nx, ny, 36, 0, 2 * Math.PI);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // Candidate Nodule Bounding Box and Crosshair
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#f59e0b';
    ctx.strokeRect(nx - 24, ny - 24, 48, 48);

    // Crosshair lines
    ctx.beginPath();
    ctx.moveTo(nx - 30, ny);
    ctx.lineTo(nx + 30, ny);
    ctx.moveTo(nx, ny - 30);
    ctx.lineTo(nx, ny + 30);
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    ctx.stroke();

  }, [sliceIndex, activeLayer, overlayOpacity, windowPreset]);

  return (
    <div style={{ padding: '24px', maxWidth: '1280px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
            03 · MULTIPLANAR CT VIEWER & XAI EXPLORER
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
            Study: <span style={{ color: '#38bdf8', fontWeight: '600' }}>{studyId}</span> | 1mm³ Isotropic Volume | Interactive Multiplanar Reconstruction
          </p>
        </div>

        <button
          onClick={() => onInspectNodule(selectedNodule)}
          style={{
            background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
            color: 'white',
            fontWeight: '700',
            fontSize: '13px',
            border: 'none',
            borderRadius: '6px',
            padding: '10px 18px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            boxShadow: '0 4px 14px rgba(2, 132, 199, 0.35)'
          }}
        >
          <Zap size={16} />
          Launch Verdict & Morphometry ({selectedNodule})
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
        {/* Main Viewport Container */}
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          {/* Overlay Selector Bar */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', width: '100%', flexWrap: 'wrap' }}>
            {[
              { id: 'none', label: 'Native CT' },
              { id: 'segmentation', label: '3D Segmentation Mask' },
              { id: 'gradcam3d', label: '3D Grad-CAM' },
              { id: 'attention_rollout3d', label: 'Swin Attention Rollout' },
              { id: 'integrated_gradients', label: 'Integrated Gradients' },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveLayer(tab.id as any)}
                style={{
                  background: activeLayer === tab.id ? '#0284c7' : '#1e293b',
                  color: activeLayer === tab.id ? '#ffffff' : '#94a3b8',
                  border: '1px solid',
                  borderColor: activeLayer === tab.id ? '#38bdf8' : '#334155',
                  padding: '6px 14px',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Interactive Canvas Viewport */}
          <div style={{ position: 'relative', border: '1px solid #1e293b', borderRadius: '8px', overflow: 'hidden' }}>
            <canvas ref={canvasRef} width={512} height={512} style={{ display: 'block', maxWidth: '100%', height: 'auto' }} />

            {/* Medical Info Overlay */}
            <div style={{ position: 'absolute', top: 12, left: 12, fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8', textShadow: '0 1px 2px #000' }}>
              <div>STUDY: {studyId}</div>
              <div>SERIES: 3D AXIAL CT</div>
              <div>SLICE: {sliceIndex} / 196</div>
            </div>

            <div style={{ position: 'absolute', bottom: 12, left: 12, fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', color: '#94a3b8', textShadow: '0 1px 2px #000' }}>
              <div>W: {windowPreset === 'lung' ? 1400 : 400} L: {windowPreset === 'lung' ? -500 : 40}</div>
              <div>THICKNESS: 1.0mm</div>
            </div>

            <div style={{ position: 'absolute', top: 12, right: 12, fontSize: '11px', color: '#f59e0b', background: 'rgba(0,0,0,0.7)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(245, 158, 11, 0.4)' }}>
              NODULE ROI: [92, 280, 310]
            </div>

            {/* Strict Regulatory Disclaimer Watermark */}
            {activeLayer !== 'none' && activeLayer !== 'segmentation' && (
              <div style={{ position: 'absolute', bottom: 12, right: 12, fontSize: '10px', color: '#f43f5e', background: 'rgba(15, 23, 42, 0.85)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(244, 63, 94, 0.4)', fontWeight: '600' }}>
                MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE
              </div>
            )}
          </div>

          {/* Slice Scrubber */}
          <div style={{ width: '100%', maxWidth: '512px', marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '12px', color: '#94a3b8', width: '70px' }}>Axial Slice:</span>
            <input
              type="range"
              min={1}
              max={196}
              value={sliceIndex}
              onChange={(e) => setSliceIndex(Number(e.target.value))}
              style={{ flex: 1, accentColor: '#0284c7' }}
            />
            <span style={{ fontSize: '12px', fontFamily: 'JetBrains Mono, monospace', color: '#f8fafc', width: '40px', textAlign: 'right' }}>
              {sliceIndex}
            </span>
          </div>
        </div>

        {/* Viewport Controls & Nodule Drawer */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Window Presets */}
          <div className="glass-panel" style={{ padding: '16px' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '13px', fontWeight: '600', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <SunMedium size={16} color="#06b6d4" />
              Window / Level Presets
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
              {(['lung', 'mediastinum', 'bone'] as const).map(w => (
                <button
                  key={w}
                  onClick={() => setWindowPreset(w)}
                  style={{
                    padding: '8px 4px',
                    borderRadius: '5px',
                    border: '1px solid',
                    borderColor: windowPreset === w ? '#06b6d4' : '#334155',
                    background: windowPreset === w ? 'rgba(6, 182, 212, 0.15)' : '#0f172a',
                    color: windowPreset === w ? '#38bdf8' : '#94a3b8',
                    fontSize: '11px',
                    fontWeight: '600',
                    textTransform: 'capitalize',
                    cursor: 'pointer'
                  }}
                >
                  {w}
                </button>
              ))}
            </div>

            {/* Opacity Control */}
            <div style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8', marginBottom: '6px' }}>
                <span>Overlay Opacity:</span>
                <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>{Math.round(overlayOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                style={{ width: '100%', accentColor: '#06b6d4' }}
              />
            </div>
          </div>

          {/* Detected Candidates */}
          <div className="glass-panel" style={{ padding: '16px', flex: 1 }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '13px', fontWeight: '600', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Crosshair size={16} color="#f59e0b" />
              Detected Nodule Candidates ({nodules.length})
            </h4>

            {nodules.map(nodule => (
              <div
                key={nodule.nodule_id}
                onClick={() => {
                  setSelectedNodule(nodule.nodule_id);
                  setSliceIndex(nodule.center_zyx[0]);
                }}
                style={{
                  padding: '12px',
                  borderRadius: '6px',
                  border: '1px solid',
                  borderColor: selectedNodule === nodule.nodule_id ? '#f59e0b' : '#1e293b',
                  background: selectedNodule === nodule.nodule_id ? 'rgba(245, 158, 11, 0.1)' : '#0b1120',
                  cursor: 'pointer',
                  marginBottom: '10px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <span style={{ fontWeight: '700', fontSize: '13px', color: '#f8fafc' }}>{nodule.nodule_id}</span>
                  <span style={{ fontSize: '11px', color: '#f59e0b', fontWeight: '600' }}>Ø {nodule.diameter_mm} mm</span>
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>{nodule.lobe}</div>
                <div style={{ fontSize: '11px', color: '#f43f5e', fontWeight: '600', marginTop: '4px' }}>
                  {nodule.preliminary_risk}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
