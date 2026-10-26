import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Maximize2, RotateCcw, Move, ZoomIn, Sun, Ruler, Square,
  Layers, ChevronLeft, ChevronRight, Eye, Play, Pause,
  CheckCircle, AlertCircle, XCircle, Edit3, Shield, Activity
} from 'lucide-react';
import { NoduleItem } from '../types';
import { API_BASE } from '../services/api';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

interface CTViewerPanelProps {
  studyId: string;
  onInspectNodule: (noduleId: string) => void;
}

type WindowPreset = 'lung' | 'mediastinum' | 'bone';
type ActiveTool = 'pan' | 'zoom' | 'wl' | 'measure' | 'roi';
type LayoutMode = '1-up' | '2x2';

interface SeriesThumb {
  id: string;
  description: string;
  slices: number;
  thickness: string;
  modality: string;
}

interface FindingState {
  id: string;
  center_zyx: [number, number, number];
  diameter_mm: number;
  volume_mm3: number;
  lobe: string;
  density: 'Solid' | 'Part-Solid' | 'Ground-Glass';
  calibrated_prob?: number;
  uncertainty_std?: number;
  confidence_band?: 'High confidence' | 'Moderate confidence' | 'Low confidence, review closely';
  needs_review?: boolean;
  status: 'PENDING' | 'ACCEPTED' | 'EDITED' | 'REJECTED';
  actionReason?: string;
  clinicianImpression?: string;
}

const PRESET_VALUES: Record<WindowPreset, { w: number; l: number; label: string }> = {
  lung: { w: 1500, l: -600, label: 'Lung (W:1500 L:-600)' },
  mediastinum: { w: 350, l: 40, label: 'Mediastinum (W:350 L:40)' },
  bone: { w: 2000, l: 400, label: 'Bone (W:2000 L:400)' },
};

const SERIES_LIST: SeriesThumb[] = [
  { id: 'SERIES-01', description: 'Axial 1.0mm Lung Kernel', slices: 196, thickness: '1.0 mm', modality: 'CT' },
  { id: 'SERIES-02', description: 'Axial Mediastinal Window', slices: 196, thickness: '1.0 mm', modality: 'CT' },
  { id: 'SERIES-03', description: 'Coronal MPR Reformat', slices: 160, thickness: '1.0 mm', modality: 'CT' },
  { id: 'SERIES-04', description: 'Sagittal MPR Reformat', slices: 160, thickness: '1.0 mm', modality: 'CT' },
];

export const CTViewerPanel: React.FC<CTViewerPanelProps> = ({ studyId, onInspectNodule }) => {
  const [sliceIndex, setSliceIndex] = useState<number>(92);
  const [maxSlices, setMaxSlices] = useState<number>(196);
  const [windowPreset, setWindowPreset] = useState<WindowPreset>('lung');
  const [activeTool, setActiveTool] = useState<ActiveTool>('pan');
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('2x2');
  const [activeSeries, setActiveSeries] = useState<string>('SERIES-01');
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Cine Playback
  const [isCinePlaying, setIsCinePlaying] = useState<boolean>(false);
  const [cineFps, setCineFps] = useState<number>(15);
  const cineTimerRef = useRef<any>(null);

  // Task 3: Blinded-First Reading Workflow
  const [blindedMode, setBlindedMode] = useState<boolean>(true);
  const [modelAssessmentRevealed, setModelAssessmentRevealed] = useState<boolean>(false);
  const [initialImpression, setInitialImpression] = useState<string>('');

  // Findings list
  const [findings, setFindings] = useState<FindingState[]>([
    {
      id: 'FINDING-01',
      center_zyx: [92, 280, 310],
      diameter_mm: 14.8,
      volume_mm3: 1696.5,
      lobe: 'Right Upper Lobe',
      density: 'Solid',
      calibrated_prob: 0.849,
      uncertainty_std: 0.042,
      confidence_band: 'High confidence',
      needs_review: false,
      status: 'PENDING',
    },
    {
      id: 'FINDING-02',
      center_zyx: [134, 190, 240],
      diameter_mm: 5.4,
      volume_mm3: 82.4,
      lobe: 'Left Lower Lobe',
      density: 'Part-Solid',
      calibrated_prob: 0.321,
      uncertainty_std: 0.185,
      confidence_band: 'Low confidence, review closely',
      needs_review: true,
      status: 'PENDING',
    },
  ]);
  const [selectedFindingId, setSelectedFindingId] = useState<string>('FINDING-01');
  const [showOverlays, setShowOverlays] = useState<boolean>(true);
  const [showCrosshairs, setShowCrosshairs] = useState<boolean>(true);

  // Measurement State
  const [rulerLengthMm, setRulerLengthMm] = useState<number | null>(null);

  const axialCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const coronalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sagittalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mipCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Cine loop
  useEffect(() => {
    if (isCinePlaying) {
      cineTimerRef.current = setInterval(() => {
        setSliceIndex(prev => (prev >= maxSlices ? 1 : prev + 1));
      }, 1000 / cineFps);
    } else {
      if (cineTimerRef.current) clearInterval(cineTimerRef.current);
    }
    return () => {
      if (cineTimerRef.current) clearInterval(cineTimerRef.current);
    };
  }, [isCinePlaying, cineFps, maxSlices]);

  // Jump to slice on finding select
  const handleSelectFinding = (finding: FindingState) => {
    setSelectedFindingId(finding.id);
    setSliceIndex(finding.center_zyx[0]);
  };

  // Action on finding
  const handleFindingAction = (id: string, action: 'ACCEPTED' | 'EDITED' | 'REJECTED', reason?: string) => {
    setFindings(prev => prev.map(f => f.id === id ? { ...f, status: action, actionReason: reason } : f));
  };

  const handleReset = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    setWindowPreset('lung');
  };

  // Canvas drawing with true PACS corner overlays
  const drawViewport = useCallback((canvas: HTMLCanvasElement | null, plane: 'axial' | 'coronal' | 'sagittal' | 'mip') => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    ctx.fillStyle = '#0a0c0e';
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.translate(W / 2 + pan.x, H / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-W / 2, -H / 2);

    // Anatomical thoracic boundary simulation
    ctx.beginPath();
    ctx.ellipse(W / 2, H / 2, W * 0.42, H * 0.38, 0, 0, 2 * Math.PI);
    ctx.fillStyle = '#14171a';
    ctx.fill();
    ctx.strokeStyle = '#23282e';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Bilateral lung fields
    [W * 0.32, W * 0.68].forEach(lx => {
      ctx.beginPath();
      ctx.ellipse(lx, H * 0.5, W * 0.14, H * 0.24, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#0b0d0f';
      ctx.fill();
      ctx.strokeStyle = '#1e2227';
      ctx.stroke();
    });

    // Render finding ROI if on slice
    if (showOverlays) {
      findings.forEach(f => {
        const sliceDiff = Math.abs(f.center_zyx[0] - sliceIndex);
        if (sliceDiff <= 6) {
          const nx = plane === 'axial' ? W * 0.64 : W * 0.58;
          const ny = plane === 'axial' ? H * 0.38 : H * 0.44;
          const r = Math.max(8, 20 - sliceDiff * 2);

          ctx.strokeStyle = f.status === 'REJECTED' ? '#64748b' : f.needs_review ? '#f59e0b' : '#ef4444';
          ctx.lineWidth = f.id === selectedFindingId ? 2 : 1.2;
          ctx.setLineDash(f.status === 'EDITED' ? [4, 4] : []);
          ctx.strokeRect(nx - r, ny - r, r * 2, r * 2);
          ctx.setLineDash([]);

          ctx.fillStyle = f.needs_review ? 'rgba(245, 158, 11, 0.15)' : 'rgba(239, 68, 68, 0.15)';
          ctx.fillRect(nx - r, ny - r, r * 2, r * 2);

          ctx.font = '10px monospace';
          ctx.fillStyle = '#e2e8f0';
          ctx.fillText(`${f.id} (${f.diameter_mm.toFixed(1)} mm)`, nx - r, ny - r - 4);
        }
      });
    }

    // Crosshairs
    if (showCrosshairs) {
      const cx = W / 2;
      const cy = H / 2;
      ctx.strokeStyle = 'rgba(59, 130, 196, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
      ctx.moveTo(0, cy); ctx.lineTo(W, cy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();

    // Orientation Labels
    ctx.font = '11px monospace';
    ctx.fillStyle = '#94a3b8';
    if (plane === 'axial') {
      ctx.textAlign = 'center';
      ctx.fillText('A', W / 2, 16);
      ctx.fillText('P', W / 2, H - 8);
      ctx.textAlign = 'left';
      ctx.fillText('R', 8, H / 2);
      ctx.textAlign = 'right';
      ctx.fillText('L', W - 8, H / 2);
    }

    // Professional DICOM Corner Overlays
    ctx.font = '11px monospace';
    ctx.fillStyle = '#cbd5e1';

    // Top-Left: Patient & Acquisition
    ctx.textAlign = 'left';
    ctx.fillText(studyId, 12, 18);
    ctx.fillText('CT CHEST (1.0 mm)', 12, 32);

    // Top-Right: Series & Geometry
    ctx.textAlign = 'right';
    ctx.fillText(`${plane.toUpperCase()} MPR`, W - 12, 18);
    ctx.fillText(`Slice ${sliceIndex}/${maxSlices}`, W - 12, 32);

    // Bottom-Left: Window / Level & Preset
    ctx.textAlign = 'left';
    ctx.fillText(`W: ${PRESET_VALUES[windowPreset].w} L: ${PRESET_VALUES[windowPreset].l}`, 12, H - 24);
    ctx.fillText(`Preset: ${windowPreset.toUpperCase()}`, 12, H - 10);

    // Bottom-Right: Zoom, Table Position & Scale bar
    ctx.textAlign = 'right';
    ctx.fillText(`Zoom: ${(zoom * 100).toFixed(0)}%`, W - 12, H - 24);
    ctx.fillText(`Z: ${(-128.0 + sliceIndex * 1.0).toFixed(1)} mm`, W - 12, H - 10);

    // Scale bar (5 cm)
    const barWidth = 50 * zoom;
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(W - 12 - barWidth, H - 38);
    ctx.lineTo(W - 12, H - 38);
    ctx.stroke();
    ctx.font = '9px monospace';
    ctx.fillText('5 cm', W - 12, H - 42);
  }, [pan, zoom, showOverlays, showCrosshairs, studyId, sliceIndex, maxSlices, windowPreset, findings, selectedFindingId]);

  useEffect(() => {
    drawViewport(axialCanvasRef.current, 'axial');
    if (layoutMode === '2x2') {
      drawViewport(coronalCanvasRef.current, 'coronal');
      drawViewport(sagittalCanvasRef.current, 'sagittal');
      drawViewport(mipCanvasRef.current, 'mip');
    }
  }, [drawViewport, layoutMode]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY > 0) {
      setSliceIndex(prev => Math.max(1, prev - 1));
    } else {
      setSliceIndex(prev => Math.min(maxSlices, prev + 1));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (activeTool === 'pan') {
      setIsDragging(true);
      dragStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && activeTool === 'pan') {
      setPan({ x: e.clientX - dragStartRef.current.x, y: e.clientY - dragStartRef.current.y });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const selectedFinding = findings.find(f => f.id === selectedFindingId);

  return (
    <div style={{ display: 'flex', gap: '8px', height: 'calc(100vh - 110px)', overflow: 'hidden' }}>
      {/* 1. Left Series Browser */}
      <aside
        style={{
          width: '180px',
          background: 'var(--bg-panel)',
          border: '1px solid var(--border)',
          borderRadius: '2px',
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          overflowY: 'auto',
          padding: '8px',
          gap: '8px',
        }}
      >
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          Series ({SERIES_LIST.length})
        </div>
        {SERIES_LIST.map(s => {
          const isCurrent = activeSeries === s.id;
          return (
            <button
              key={s.id}
              onClick={() => setActiveSeries(s.id)}
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '4px',
                padding: '6px 8px',
                background: isCurrent ? 'var(--bg-active)' : 'var(--bg-surface)',
                border: isCurrent ? '1px solid var(--accent)' : '1px solid var(--border)',
                borderRadius: '2px',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', fontWeight: 600, color: isCurrent ? 'var(--accent)' : 'var(--text-main)' }}>
                  {s.id}
                </span>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{s.modality}</span>
              </div>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: '1.2' }}>{s.description}</div>
              <div style={{ fontSize: '9px', color: 'var(--text-muted)' }} className="tabular-nums">
                {s.slices} slices · {s.thickness}
              </div>
            </button>
          );
        })}
      </aside>

      {/* 2. Center Viewport Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '6px', overflow: 'hidden' }}>
        {/* Top Viewport Toolbar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'var(--bg-panel)',
            padding: '6px 10px',
            border: '1px solid var(--border)',
            borderRadius: '2px',
            fontSize: '12px',
          }}
        >
          {/* Tool Group */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Button size="sm" variant={activeTool === 'pan' ? 'primary' : 'ghost'} onClick={() => setActiveTool('pan')}>
              <Move size={12} style={{ marginRight: '4px' }} /> Pan
            </Button>
            <Button size="sm" variant={activeTool === 'zoom' ? 'primary' : 'ghost'} onClick={() => { setActiveTool('zoom'); setZoom(z => Math.min(4.0, z + 0.25)); }}>
              <ZoomIn size={12} style={{ marginRight: '4px' }} /> Zoom
            </Button>
            <Button size="sm" variant={activeTool === 'wl' ? 'primary' : 'ghost'} onClick={() => setActiveTool('wl')}>
              <Sun size={12} style={{ marginRight: '4px' }} /> W/L
            </Button>
            <Button size="sm" variant={activeTool === 'measure' ? 'primary' : 'ghost'} onClick={() => setActiveTool('measure')}>
              <Ruler size={12} style={{ marginRight: '4px' }} /> Caliper
            </Button>
            <Button size="sm" variant="ghost" onClick={handleReset}>
              <RotateCcw size={12} style={{ marginRight: '4px' }} /> Reset
            </Button>
          </div>

          {/* W/L Presets */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Window:</span>
            {(['lung', 'mediastinum', 'bone'] as const).map(p => (
              <Button key={p} size="sm" variant={windowPreset === p ? 'secondary' : 'ghost'} onClick={() => setWindowPreset(p)}>
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </Button>
            ))}
          </div>

          {/* Layout & Cine Mode */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
              <Button size="sm" variant={layoutMode === '1-up' ? 'secondary' : 'ghost'} onClick={() => setLayoutMode('1-up')}>
                1-Up
              </Button>
              <Button size="sm" variant={layoutMode === '2x2' ? 'secondary' : 'ghost'} onClick={() => setLayoutMode('2x2')}>
                2x2 MPR
              </Button>
            </div>

            <Button
              size="sm"
              variant={isCinePlaying ? 'secondary' : 'ghost'}
              onClick={() => setIsCinePlaying(!isCinePlaying)}
            >
              {isCinePlaying ? <Pause size={12} style={{ marginRight: '4px' }} /> : <Play size={12} style={{ marginRight: '4px' }} />}
              Cine
            </Button>

            {/* Blinded First Toggle (Task 3) */}
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: 'var(--text-muted)', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={blindedMode}
                onChange={e => {
                  setBlindedMode(e.target.checked);
                  if (!e.target.checked) setModelAssessmentRevealed(true);
                }}
                style={{ accentColor: 'var(--accent)' }}
              />
              Blinded-First Mode
            </label>
          </div>
        </div>

        {/* Viewport Canvas Grid */}
        <div
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          style={{
            flex: 1,
            display: 'grid',
            gridTemplateColumns: layoutMode === '2x2' ? '1fr 1fr' : '1fr',
            gridTemplateRows: layoutMode === '2x2' ? '1fr 1fr' : '1fr',
            gap: '3px',
            background: 'var(--border)',
            border: '1px solid var(--border)',
            borderRadius: '2px',
            overflow: 'hidden',
            cursor: activeTool === 'pan' ? (isDragging ? 'grabbing' : 'grab') : 'crosshair',
          }}
        >
          <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0a0c0e' }}>
            <canvas ref={axialCanvasRef} width={520} height={360} style={{ width: '100%', height: '100%', display: 'block' }} />
          </div>

          {layoutMode === '2x2' && (
            <>
              <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0a0c0e' }}>
                <canvas ref={coronalCanvasRef} width={520} height={360} style={{ width: '100%', height: '100%', display: 'block' }} />
              </div>
              <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0a0c0e' }}>
                <canvas ref={sagittalCanvasRef} width={520} height={360} style={{ width: '100%', height: '100%', display: 'block' }} />
              </div>
              <div style={{ position: 'relative', width: '100%', height: '100%', background: '#0a0c0e' }}>
                <canvas ref={mipCanvasRef} width={520} height={360} style={{ width: '100%', height: '100%', display: 'block' }} />
              </div>
            </>
          )}
        </div>

        {/* Synced Per-Slice Probability & Risk Profile Dock */}
        <div
          style={{
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: '2px',
            padding: '6px 12px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '11px',
          }}
        >
          <span style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Slice Score Profile:</span>
          {/* Visual slice profile bar */}
          <div style={{ flex: 1, position: 'relative', height: '18px', background: 'var(--bg-surface)', borderRadius: '2px', overflow: 'hidden' }}>
            {/* Finding markers along Z */}
            {findings.map(f => {
              const posPercent = (f.center_zyx[0] / maxSlices) * 100;
              return (
                <div
                  key={f.id}
                  style={{
                    position: 'absolute',
                    left: `${posPercent}%`,
                    top: 0,
                    bottom: 0,
                    width: '3px',
                    background: f.needs_review ? '#f59e0b' : '#ef4444',
                    cursor: 'pointer',
                  }}
                  title={`${f.id} at Slice ${f.center_zyx[0]}`}
                  onClick={() => setSliceIndex(f.center_zyx[0])}
                />
              );
            })}
            {/* Current scroll position cursor */}
            <div
              style={{
                position: 'absolute',
                left: `${(sliceIndex / maxSlices) * 100}%`,
                top: 0,
                bottom: 0,
                width: '2px',
                background: '#38bdf8',
                boxShadow: '0 0 4px #38bdf8',
              }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Button size="sm" variant="ghost" onClick={() => setSliceIndex(prev => Math.max(1, prev - 1))} disabled={sliceIndex <= 1}>
              <ChevronLeft size={12} />
            </Button>
            <span className="tabular-nums" style={{ minWidth: '70px', textAlign: 'center', color: 'var(--text-main)' }}>
              {sliceIndex} / {maxSlices}
            </span>
            <Button size="sm" variant="ghost" onClick={() => setSliceIndex(prev => Math.min(maxSlices, prev + 1))} disabled={sliceIndex >= maxSlices}>
              <ChevronRight size={12} />
            </Button>
          </div>
        </div>
      </div>

      {/* 3. Right Findings & Verification Panel (Task 3 Evidence-First) */}
      <aside
        style={{
          width: '320px',
          background: 'var(--bg-panel)',
          border: '1px solid var(--border)',
          borderRadius: '2px',
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          overflowY: 'auto',
          padding: '10px',
          gap: '10px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Findings Table ({findings.length})
          </div>
          <Badge variant={blindedMode ? 'warning' : 'neutral'}>
            {blindedMode ? 'Blinded-First Active' : 'Assisted Mode'}
          </Badge>
        </div>

        {/* Compact Findings Table */}
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', textAlign: 'left' }}>
              <th style={{ padding: '4px' }}>Finding</th>
              <th style={{ padding: '4px' }}>Lobe</th>
              <th style={{ padding: '4px' }}>Ø (mm)</th>
              <th style={{ padding: '4px' }}>Vol (mm³)</th>
              <th style={{ padding: '4px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {findings.map(f => {
              const isSelected = f.id === selectedFindingId;
              return (
                <tr
                  key={f.id}
                  onClick={() => handleSelectFinding(f)}
                  style={{
                    background: isSelected ? 'var(--bg-active)' : 'transparent',
                    borderBottom: '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                >
                  <td style={{ padding: '6px 4px', fontWeight: 600, color: isSelected ? 'var(--accent)' : 'var(--text-main)' }}>
                    {f.id}
                  </td>
                  <td style={{ padding: '6px 4px', color: 'var(--text-muted)' }}>{f.lobe.replace('Right ', 'R ').replace('Left ', 'L ')}</td>
                  <td style={{ padding: '6px 4px' }} className="tabular-nums">{f.diameter_mm.toFixed(1)}</td>
                  <td style={{ padding: '6px 4px' }} className="tabular-nums">{f.volume_mm3.toFixed(0)}</td>
                  <td style={{ padding: '6px 4px' }}>
                    <span
                      style={{
                        padding: '1px 4px',
                        borderRadius: '2px',
                        fontSize: '9px',
                        fontWeight: 600,
                        background:
                          f.status === 'ACCEPTED' ? 'var(--status-pass-bg)' :
                          f.status === 'REJECTED' ? 'var(--status-fail-bg)' :
                          f.status === 'EDITED' ? 'var(--status-warning-bg)' : 'var(--bg-surface)',
                        color:
                          f.status === 'ACCEPTED' ? 'var(--status-pass)' :
                          f.status === 'REJECTED' ? 'var(--status-fail)' :
                          f.status === 'EDITED' ? 'var(--status-warning)' : 'var(--text-muted)',
                      }}
                    >
                      {f.status}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        {/* Selected Finding Detail & Blinded-First Decision Gate */}
        {selectedFinding && (
          <div
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border)',
              borderRadius: '2px',
              padding: '10px',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '12px' }}>{selectedFinding.id} Detail</span>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Centroid Z={selectedFinding.center_zyx[0]}</span>
            </div>

            {/* Geometric Measurements (Always Visible to Clinician) */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontSize: '11px' }}>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Max Caliper: </span>
                <strong className="tabular-nums">{selectedFinding.diameter_mm.toFixed(1)} mm</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>3D Volume: </span>
                <strong className="tabular-nums">{selectedFinding.volume_mm3.toFixed(1)} mm³</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Density: </span>
                <span>{selectedFinding.density}</span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Location: </span>
                <span>{selectedFinding.lobe}</span>
              </div>
            </div>

            {/* Task 3 Blinded First Gate */}
            {blindedMode && !modelAssessmentRevealed ? (
              <div
                style={{
                  background: 'var(--bg-panel)',
                  border: '1px dashed var(--border)',
                  padding: '8px',
                  borderRadius: '2px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                }}
              >
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Record your initial impression prior to viewing model malignancy estimate:
                </div>
                <div style={{ display: 'flex', gap: '4px' }}>
                  {['Benign', 'Indeterminate', 'Suspicious'].map(imp => (
                    <Button
                      key={imp}
                      size="sm"
                      variant={initialImpression === imp ? 'primary' : 'ghost'}
                      onClick={() => setInitialImpression(imp)}
                    >
                      {imp}
                    </Button>
                  ))}
                </div>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => setModelAssessmentRevealed(true)}
                  disabled={!initialImpression}
                >
                  Reveal Model Assessment
                </Button>
              </div>
            ) : (
              /* Model Output (Revealed) */
              <div
                style={{
                  background: 'var(--bg-panel)',
                  border: '1px solid var(--border)',
                  padding: '8px',
                  borderRadius: '2px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                }}
              >
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                  Model Decision Support (Research Only)
                </div>

                {selectedFinding.needs_review ? (
                  /* Abstain State (Task 3 Requirement 3) */
                  <div
                    style={{
                      background: 'var(--status-warning-bg)',
                      border: '1px solid var(--status-warning-border)',
                      color: 'var(--status-warning)',
                      padding: '6px',
                      borderRadius: '2px',
                      fontSize: '11px',
                    }}
                  >
                    <strong>Needs Human Review</strong>
                    <div style={{ fontSize: '10px', marginTop: '2px' }}>
                      Epistemic uncertainty exceeds validated threshold (±{selectedFinding.uncertainty_std?.toFixed(3)}). Confident score withheld.
                    </div>
                  </div>
                ) : (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Calibrated Probability:</span>
                      <span style={{ fontSize: '14px', fontWeight: 700, color: '#ef4444' }} className="tabular-nums">
                        {((selectedFinding.calibrated_prob || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)' }}>
                      <span>MC-Dropout Uncertainty:</span>
                      <span className="tabular-nums">±{(selectedFinding.uncertainty_std || 0).toFixed(3)}</span>
                    </div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
                      Confidence band: <em>{selectedFinding.confidence_band}</em>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Clinician Action Buttons: Accept / Edit / Reject (Task 3 Requirement 4) */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                Clinician Verdict &amp; Audit Action
              </div>
              <div style={{ display: 'flex', gap: '4px' }}>
                <Button
                  size="sm"
                  variant={selectedFinding.status === 'ACCEPTED' ? 'primary' : 'ghost'}
                  onClick={() => handleFindingAction(selectedFinding.id, 'ACCEPTED')}
                >
                  <CheckCircle size={12} style={{ marginRight: '4px' }} /> Accept
                </Button>
                <Button
                  size="sm"
                  variant={selectedFinding.status === 'EDITED' ? 'secondary' : 'ghost'}
                  onClick={() => handleFindingAction(selectedFinding.id, 'EDITED', 'Contour adjusted by clinician')}
                >
                  <Edit3 size={12} style={{ marginRight: '4px' }} /> Edit
                </Button>
                <Button
                  size="sm"
                  variant={selectedFinding.status === 'REJECTED' ? 'danger' : 'ghost'}
                  onClick={() => handleFindingAction(selectedFinding.id, 'REJECTED', 'Vascular branch mimic')}
                >
                  <XCircle size={12} style={{ marginRight: '4px' }} /> Reject
                </Button>
              </div>

              {selectedFinding.actionReason && (
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '2px' }}>
                  Action reason: {selectedFinding.actionReason}
                </div>
              )}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
};

export default CTViewerPanel;
