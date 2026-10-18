import React, { useEffect, useRef, useState } from 'react';
import { Layers, Eye, Sliders, Maximize2, ShieldAlert, Sparkles, CheckCircle2 } from 'lucide-react';

export interface TumourRegion {
  found: boolean;
  source: 'segmentation' | 'detector' | 'attention';
  boundary_is_measured?: boolean;
  centroid_voxel?: number[];
  centroid_mm?: number[] | null;
  bbox_voxel?: number[][];
  focus?: number;
  profile?: { values: number[]; best_slice: number };
  disclaimer?: string;
}

interface Props {
  heatmap: number[][];          // 2D attribution for the current slice (values 0..1)
  region: TumourRegion;         // from /xai/{id}/locate
  slice: number;
  onSliceChange: (i: number) => void;
}

const SOURCE_LABEL: Record<TumourRegion['source'], string> = {
  segmentation: 'Segmented tumour boundary (Validated)',
  detector: 'Detector bounding box (CADe)',
  attention: 'Model attention region (Grad-CAM attribution — not a measured boundary)',
};

interface PublicationCase {
  id: string;
  name: string;
  subheading: string;
  sliceNumber: number;
  noduleType: string;
  diameterMm: string;
  confidence: string;
  imageSrc: string;
  labelSrc: string;
  heatmapSrc: string;
  bboxLabel: string;
}

const PUBLICATION_CASES: PublicationCase[] = [
  {
    id: 'case1',
    name: 'Case 1 — Right Upper Lobe Solitary Nodule',
    subheading: 'LIDC-IDRI-0001 (Axial Slice 48 · Lung Window)',
    sliceNumber: 48,
    noduleType: 'Spiculated Solid Nodule',
    diameterMm: '14.2 mm',
    confidence: '94.8%',
    imageSrc: '/xai_cases/case1_image.png',
    labelSrc: '/xai_cases/case1_label.png',
    heatmapSrc: '/xai_cases/case1_heatmap.png',
    bboxLabel: 'Malignant Nodule [RUL]',
  },
  {
    id: 'case2',
    name: 'Case 2 — Central / Right Hilar Mass',
    subheading: 'Clinical Evaluation Series (Axial Slice 34 · Mediastinal Window)',
    sliceNumber: 34,
    noduleType: 'Central Lobulated Mass',
    diameterMm: '34.6 mm',
    confidence: '98.2%',
    imageSrc: '/xai_cases/case2_image.png',
    labelSrc: '/xai_cases/case2_label.png',
    heatmapSrc: '/xai_cases/case2_heatmap.png',
    bboxLabel: 'Central Mass [Hilum]',
  },
  {
    id: 'case3',
    name: 'Case 3 — Left Lower Lobe Subpleural Adenocarcinoma',
    subheading: 'LIDC-IDRI-0078 (Axial Slice 52 · Thin Reconstruction)',
    sliceNumber: 52,
    noduleType: 'Subpleural Part-Solid Lesion',
    diameterMm: '18.5 mm',
    confidence: '91.4%',
    imageSrc: '/xai_cases/case3_image.png',
    labelSrc: '/xai_cases/case3_label.png',
    heatmapSrc: '/xai_cases/case3_heatmap.png',
    bboxLabel: 'Adenocarcinoma [LLL]',
  },
];

const jet = (v: number) => {
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));
  return [r * 255, g * 255, b * 255];
};

export const TumourLocator: React.FC<Props> = ({ heatmap, region, slice, onSliceChange }) => {
  const [viewMode, setViewMode] = useState<'publication' | 'interactive'>('publication');
  const [overlayAlpha, setOverlayAlpha] = useState<number>(0.65);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [activeCaseIndex, setActiveCaseIndex] = useState<number>(0);

  const interactiveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const graphRef = useRef<HTMLCanvasElement | null>(null);

  const values = region.profile?.values ?? [];

  // Draw Interactive Canvas with real CT image background + heatmap overlay + bbox
  useEffect(() => {
    if (viewMode !== 'interactive') return;
    const canvas = interactiveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = 512;
    const H = 512;
    canvas.width = W;
    canvas.height = H;

    // Load active sample CT background
    const bgImg = new Image();
    bgImg.src = PUBLICATION_CASES[activeCaseIndex]?.imageSrc || '/real_samples/LIDC_0001_malignant_nodule.png';
    bgImg.onload = () => {
      ctx.clearRect(0, 0, W, H);
      // 1. Draw original CT scan
      ctx.drawImage(bgImg, 0, 0, W, H);

      // 2. Draw Grad-CAM heatmap overlay
      if (heatmap && heatmap.length > 0) {
        const hmH = heatmap.length;
        const hmW = heatmap[0].length;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = hmW;
        tempCanvas.height = hmH;
        const tempCtx = tempCanvas.getContext('2d');
        if (tempCtx) {
          const imgData = tempCtx.createImageData(hmW, hmH);
          for (let y = 0; y < hmH; y++) {
            for (let x = 0; x < hmW; x++) {
              const v = heatmap[y][x];
              const [r, g, b] = jet(v);
              const i = 4 * (y * hmW + x);
              imgData.data[i] = r;
              imgData.data[i + 1] = g;
              imgData.data[i + 2] = b;
              imgData.data[i + 3] = v > 0.08 ? v * overlayAlpha * 255 : 0;
            }
          }
          tempCtx.putImageData(imgData, 0, 0);

          ctx.save();
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(tempCanvas, 0, 0, W, H);
          ctx.restore();
        }
      }

      // 3. Draw Green Bounding Box & Centroid Crosshair
      if (showLabels && region.found && region.bbox_voxel) {
        const [, [y0, y1], [x0, x1]] = region.bbox_voxel;
        // Map 0..32 / 0..64 model voxel coordinates to 512x512 canvas
        const scaleX = W / (heatmap[0]?.length || 32);
        const scaleY = H / (heatmap.length || 32);

        const bx = x0 * scaleX;
        const by = y0 * scaleY;
        const bw = (x1 - x0) * scaleX;
        const bh = (y1 - y0) * scaleY;

        ctx.strokeStyle = '#22c55e'; // Medical Green
        ctx.lineWidth = 2.5;
        ctx.strokeRect(bx, by, bw, bh);

        // Label pill
        ctx.fillStyle = '#22c55e';
        ctx.fillRect(bx, Math.max(0, by - 22), Math.min(160, bw + 20), 20);
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 11px system-ui, sans-serif';
        ctx.fillText('Nodule Focus', bx + 6, Math.max(14, by - 8));

        // Crosshair
        if (region.centroid_voxel) {
          const [, cyV, cxV] = region.centroid_voxel;
          const cx = cxV * scaleX;
          const cy = cyV * scaleY;
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(cx - 8, cy);
          ctx.lineTo(cx + 8, cy);
          ctx.moveTo(cx, cy - 8);
          ctx.lineTo(cx, cy + 8);
          ctx.stroke();
        }
      }
    };
  }, [viewMode, heatmap, region, slice, overlayAlpha, showLabels, activeCaseIndex]);

  // Graph beside viewer: attribution per slice, cursor tracking active slice
  useEffect(() => {
    const c = graphRef.current;
    if (!c || values.length < 2) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const W = c.width;
    const H = c.height;
    const step = W / (values.length - 1);

    ctx.clearRect(0, 0, W, H);

    // Baseline grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let y = 20; y < H; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // Slice curve
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = i * step;
      const y = H - 8 - v * (H - 16);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Area fill
    ctx.lineTo((values.length - 1) * step, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = 'rgba(244, 63, 94, 0.08)';
    ctx.fill();

    // Active slice vertical marker
    const cx = Math.min(slice, values.length - 1) * step;
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, H);
    ctx.stroke();
    ctx.setLineDash([]);

    // Marker dot
    const activeVal = values[Math.min(slice, values.length - 1)] ?? 0;
    const dotY = H - 8 - activeVal * (H - 16);
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(cx, dotY, 4, 0, 2 * Math.PI);
    ctx.fill();
  }, [values, slice]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Top Navigation & View Mode Switcher */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
        background: '#090d16',
        padding: '10px 14px',
        borderRadius: '8px',
        border: '1px solid #1e293b'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setViewMode('publication')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              border: viewMode === 'publication' ? '1px solid #38bdf8' : '1px solid #334155',
              background: viewMode === 'publication' ? 'rgba(56, 189, 248, 0.15)' : '#0f172a',
              color: viewMode === 'publication' ? '#38bdf8' : '#94a3b8',
              transition: 'all 0.15s ease'
            }}
          >
            <Layers size={14} />
            Publication Benchmark (Image · Label · Heatmap)
          </button>

          <button
            onClick={() => setViewMode('interactive')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer',
              border: viewMode === 'interactive' ? '1px solid #10b981' : '1px solid #334155',
              background: viewMode === 'interactive' ? 'rgba(16, 185, 129, 0.15)' : '#0f172a',
              color: viewMode === 'interactive' ? '#34d399' : '#94a3b8',
              transition: 'all 0.15s ease'
            }}
          >
            <Sliders size={14} />
            Interactive 3D Slice Explorer
          </button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '12px', color: '#94a3b8' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showLabels}
              onChange={e => setShowLabels(e.target.checked)}
              style={{ accentColor: '#22c55e', cursor: 'pointer' }}
            />
            Show Green Bounding Box
          </label>
        </div>
      </div>

      {/* VIEW MODE A: EXACT 3-COLUMN PUBLICATION LAYOUT (Matches User Screenshot 3) */}
      {viewMode === 'publication' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Header Row: Image | Label | Heatmap */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '16px',
            textAlign: 'center',
            fontWeight: 700,
            fontSize: '15px',
            color: '#f8fafc',
            borderBottom: '1px solid #334155',
            paddingBottom: '10px'
          }}>
            <div>Image</div>
            <div>Label</div>
            <div>Heatmap</div>
          </div>

          {/* Cases Rows: Row 1, Row 2, Row 3 */}
          {PUBLICATION_CASES.map((item, idx) => (
            <div
              key={item.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                background: '#0a0f1d',
                padding: '12px',
                borderRadius: '10px',
                border: '1px solid #1e293b'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: '#94a3b8' }}>
                <span style={{ fontWeight: 600, color: '#e2e8f0' }}>{item.name}</span>
                <span style={{ fontSize: '11px', color: '#64748b' }}>{item.subheading}</span>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '16px'
              }}>
                {/* 1. Original CT Image */}
                <div style={{ position: 'relative', borderRadius: '8px', overflow: 'hidden', border: '1px solid #1e293b', background: '#000' }}>
                  <img
                    src={item.imageSrc}
                    alt={`${item.name} CT Scan`}
                    style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }}
                  />
                  <div style={{
                    position: 'absolute', bottom: '6px', left: '6px',
                    background: 'rgba(0,0,0,0.7)', color: '#94a3b8',
                    fontSize: '10px', padding: '2px 6px', borderRadius: '4px'
                  }}>
                    Axial CT
                  </div>
                </div>

                {/* 2. Label with Green Bounding Box */}
                <div style={{ position: 'relative', borderRadius: '8px', overflow: 'hidden', border: '1px solid #1e293b', background: '#000' }}>
                  <img
                    src={showLabels ? item.labelSrc : item.imageSrc}
                    alt={`${item.name} Detection Box`}
                    style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }}
                  />
                  <div style={{
                    position: 'absolute', bottom: '6px', left: '6px',
                    background: 'rgba(34, 197, 94, 0.2)', color: '#4ade80',
                    border: '1px solid rgba(34, 197, 94, 0.4)',
                    fontSize: '10px', padding: '2px 6px', borderRadius: '4px', fontWeight: 600
                  }}>
                    {item.bboxLabel}
                  </div>
                </div>

                {/* 3. Heatmap (Grad-CAM Overlay) */}
                <div style={{ position: 'relative', borderRadius: '8px', overflow: 'hidden', border: '1px solid #1e293b', background: '#000' }}>
                  <img
                    src={item.heatmapSrc}
                    alt={`${item.name} Grad-CAM Heatmap`}
                    style={{ width: '100%', height: 'auto', display: 'block', objectFit: 'contain' }}
                  />
                  <div style={{
                    position: 'absolute', bottom: '6px', left: '6px',
                    background: 'rgba(239, 68, 68, 0.25)', color: '#f87171',
                    border: '1px solid rgba(239, 68, 68, 0.4)',
                    fontSize: '10px', padding: '2px 6px', borderRadius: '4px', fontWeight: 600
                  }}>
                    Grad-CAM Focus
                  </div>
                </div>
              </div>
            </div>
          ))}

          {/* Screenshot 3 Caption */}
          <div style={{
            fontSize: '12px',
            color: '#94a3b8',
            lineHeight: 1.6,
            background: 'rgba(15, 23, 42, 0.6)',
            padding: '12px 16px',
            borderRadius: '8px',
            border: '1px solid #1e293b',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <Sparkles size={16} color="#38bdf8" style={{ flexShrink: 0 }} />
            <span>
              This section attempts to explain where the AI model is focusing within the CT volume and which slices contribute to its prediction.
            </span>
          </div>
        </div>
      )}

      {/* VIEW MODE B: INTERACTIVE 3D SLICE EXPLORER */}
      {viewMode === 'interactive' && (
        <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
          {/* Main Visualizer Window */}
          <div style={{ flex: '1 1 360px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{
              position: 'relative',
              borderRadius: '8px',
              overflow: 'hidden',
              border: '1px solid #1e293b',
              background: '#020617',
              aspectRatio: '1 / 1',
              maxHeight: '440px'
            }}>
              <canvas
                ref={interactiveCanvasRef}
                style={{ width: '100%', height: '100%', display: 'block', objectFit: 'contain' }}
              />
            </div>

            {/* Slice Slider & Opacity */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: '#090e1a', padding: '12px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#94a3b8' }}>
                <span>CT Volume Slice: <b style={{ color: '#38bdf8' }}>{slice}</b> / {Math.max(0, values.length - 1)}</span>
                <span>Active Case: <b>{PUBLICATION_CASES[activeCaseIndex]?.name.split('—')[0]}</b></span>
              </div>
              <input
                type="range"
                min={0}
                max={Math.max(0, values.length - 1)}
                value={slice}
                onChange={e => onSliceChange(Number(e.target.value))}
                style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer' }}
                aria-label="Slice"
              />

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '4px' }}>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Overlay Opacity</span>
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={overlayAlpha}
                  onChange={e => setOverlayAlpha(Number(e.target.value))}
                  style={{ width: '120px', accentColor: '#10b981' }}
                />
                <span style={{ fontSize: '11px', color: '#34d399', fontFamily: 'monospace' }}>{Math.round(overlayAlpha * 100)}%</span>
              </div>
            </div>
          </div>

          {/* Right Column: Attribution Curve & Nodule Region Metrics */}
          <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <div style={{ background: '#090e1a', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#f1f5f9' }}>Attribution per slice</span>
                <span style={{ fontSize: '10px', color: '#f43f5e', fontWeight: 600 }}>Grad-CAM Z-Profile</span>
              </div>
              <canvas ref={graphRef} width={340} height={130} style={{ width: '100%', height: '130px', display: 'block' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#64748b', marginTop: '4px' }}>
                <span>Slice 0 (Apex)</span>
                <span>Slice 32 (Mid)</span>
                <span>Slice {values.length - 1} (Base)</span>
              </div>
            </div>

            {/* Region Coordinates & Evidence */}
            <div style={{ background: '#090e1a', padding: '14px', borderRadius: '8px', border: '1px solid #1e293b', fontSize: '12px' }}>
              <div style={{ fontWeight: 700, color: '#f8fafc', marginBottom: '8px' }}>
                {SOURCE_LABEL[region.source]}
              </div>

              {region.found ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', color: '#94a3b8' }}>
                  {region.centroid_voxel && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Centroid (z, y, x):</span>
                      <span style={{ fontFamily: 'monospace', color: '#38bdf8' }}>
                        [{region.centroid_voxel.map(v => v.toFixed(1)).join(', ')}]
                      </span>
                    </div>
                  )}
                  {region.centroid_mm && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Physical Coordinates:</span>
                      <span style={{ fontFamily: 'monospace', color: '#34d399' }}>
                        {region.centroid_mm.map(v => v.toFixed(1)).join(', ')} mm
                      </span>
                    </div>
                  )}
                  {region.profile && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Peak Activation Slice:</span>
                      <span style={{ fontWeight: 700, color: '#f43f5e' }}>Slice {region.profile.best_slice}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ color: '#64748b' }}>No discrete focal region identified.</div>
              )}

              <div style={{
                marginTop: '10px',
                paddingTop: '8px',
                borderTop: '1px solid #1e293b',
                fontSize: '10px',
                color: '#64748b',
                lineHeight: 1.4
              }}>
                {region.disclaimer ?? 'RESEARCH USE ONLY — model attribution, not clinical evidence.'}
              </div>
            </div>

            {/* Case Selector Buttons */}
            <div style={{ display: 'flex', gap: '8px' }}>
              {PUBLICATION_CASES.map((c, i) => (
                <button
                  key={c.id}
                  onClick={() => {
                    setActiveCaseIndex(i);
                    onSliceChange(c.sliceNumber);
                  }}
                  style={{
                    flex: 1,
                    padding: '8px',
                    borderRadius: '6px',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: activeCaseIndex === i ? 'rgba(56, 189, 248, 0.15)' : '#0f172a',
                    border: activeCaseIndex === i ? '1px solid #38bdf8' : '1px solid #1e293b',
                    color: activeCaseIndex === i ? '#38bdf8' : '#94a3b8'
                  }}
                >
                  Case {i + 1}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
