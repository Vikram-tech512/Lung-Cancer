import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Upload, Layers, Eye, Crosshair, AlertCircle, 
  FileText, Activity, CheckCircle2, Loader2, Sparkles, Target, Compass,
  Move, ZoomIn, Sun, Ruler, Square, RotateCcw, ChevronLeft, ChevronRight,
  ShieldAlert, Info
} from 'lucide-react';

import { AIAssistantExplanation, XAIMode } from '../components/AIAssistantExplanation';
import { HUHistogram } from '../components/EKGWaveform';
import { API_BASE } from '../services/api';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Panel } from '../components/ui/Panel';
import { Table, Thead, Tbody, Tr, Th, Td } from '../components/ui/Table';

export interface CaseItem {
  id: string;
  nodule_id?: string;
  name: string;
  size: string;
  timestamp: string;
  verdictClass: 'MALIGNANT' | 'BENIGN';
  verdictProb: number;
  uncertainty: 'LOW' | 'MODERATE' | 'HIGH';
  anatomicMark: string;
  crosshairLabel: string;
  area: number;
  densityPercent: number;
  centroid: string;
  compactness: number;
  lobularity: number;
  latencyMs: number;
  pid: string;
  severity: string;
  cause: string;
  imageUrl?: string;
  cxRel: number;
  cyRel: number;
  roc_auc: number;
  auc_ci_lower: number;
  auc_ci_upper: number;
  sensitivity: number;
  specificity: number;
  lobe: string;
  lobeSub: string;
  tumorImpact: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW';
  hasLesion?: boolean;
}

const PRIMARY_SAMPLE: CaseItem = {
  id: 'case-sample-primary',
  name: 'LIDC-0087-Axial-092.png',
  size: '310.1 KB',
  timestamp: '10:42:15',
  verdictClass: 'MALIGNANT',
  verdictProb: 84.9,
  uncertainty: 'MODERATE',
  severity: 'Lung-RADS 4B (High Suspicion)',
  cause: 'Solid Nodule with Spiculated Margins',
  anatomicMark: 'RUL-S1',
  crosshairLabel: '14.8 mm',
  area: 1696.5,
  densityPercent: 26.16,
  centroid: '195, 168',
  compactness: 0.44,
  lobularity: 0.42,
  latencyMs: 142,
  pid: 'PAT-0087-LIDC',
  imageUrl: '/samples/malignant_sample.png',
  cxRel: 0.62,
  cyRel: 0.38,
  roc_auc: 0.942,
  auc_ci_lower: 0.912,
  auc_ci_upper: 0.972,
  sensitivity: 0.925,
  specificity: 0.960,
  lobe: 'Right Upper Lobe',
  lobeSub: 'Apical segment (S1)',
  tumorImpact: 'HIGH',
  hasLesion: true
};

const SECONDARY_SAMPLE: CaseItem = {
  id: 'case-sample-benign',
  name: 'LIDC-0023-Axial-065.png',
  size: '248.4 KB',
  timestamp: '09:15:30',
  verdictClass: 'BENIGN',
  verdictProb: 14.2,
  uncertainty: 'LOW',
  severity: 'Lung-RADS 2 (Benign Appearance)',
  cause: 'Calcified Granuloma / Hamartoma',
  anatomicMark: 'LLL-S10',
  crosshairLabel: '8.4 mm',
  area: 412.3,
  densityPercent: 6.81,
  centroid: '407, 290',
  compactness: 0.89,
  lobularity: 0.12,
  latencyMs: 118,
  pid: 'PAT-0023-LIDC',
  imageUrl: '/samples/benign_sample.png',
  cxRel: 0.35,
  cyRel: 0.65,
  roc_auc: 0.956,
  auc_ci_lower: 0.929,
  auc_ci_upper: 0.978,
  sensitivity: 0.932,
  specificity: 0.961,
  lobe: 'Left Lower Lobe',
  lobeSub: 'Posterior basal segment (S10)',
  tumorImpact: 'LOW',
  hasLesion: true
};

type WindowPreset = 'lung' | 'mediastinum' | 'bone';
type ActiveTool = 'pan' | 'zoom' | 'wl' | 'measure' | 'roi';

const PRESET_VALUES: Record<WindowPreset, { w: number; l: number; label: string }> = {
  lung: { w: 1500, l: -600, label: 'Lung (W:1500 L:-600)' },
  mediastinum: { w: 350, l: 40, label: 'Mediastinum (W:350 L:40)' },
  bone: { w: 2000, l: 400, label: 'Bone (W:2000 L:400)' },
};

export const WorkstationPanel: React.FC = () => {
  const [activeCase, setActiveCase] = useState<CaseItem>(PRIMARY_SAMPLE);
  const [sliceIndex, setSliceIndex] = useState<number>(92);
  const maxSlices = 196;
  const [windowPreset, setWindowPreset] = useState<WindowPreset>('lung');
  const [activeTool, setActiveTool] = useState<ActiveTool>('pan');
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const [scanImage, setScanImage] = useState<HTMLImageElement | null>(null);
  const [showOverlays, setShowOverlays] = useState<boolean>(true);
  const [showCrosshairs, setShowCrosshairs] = useState<boolean>(true);

  // Canvases for 2x2 viewports
  const axialRef = useRef<HTMLCanvasElement | null>(null);
  const coronalRef = useRef<HTMLCanvasElement | null>(null);
  const sagittalRef = useRef<HTMLCanvasElement | null>(null);
  const mipRef = useRef<HTMLCanvasElement | null>(null);

  // Load active scan image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => setScanImage(img);
    img.onerror = () => setScanImage(null);
    img.src = activeCase.imageUrl || '/samples/malignant_sample.png';
  }, [activeCase]);

  // Keyboard navigation shortcuts: arrows = slice, W = window/level, R = reset
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSliceIndex(prev => Math.min(maxSlices, prev + 1));
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSliceIndex(prev => Math.max(1, prev - 1));
      } else if (e.key === 'w' || e.key === 'W') {
        e.preventDefault();
        setWindowPreset(prev => prev === 'lung' ? 'mediastinum' : prev === 'mediastinum' ? 'bone' : 'lung');
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleReset();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [maxSlices]);

  const handleReset = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    setWindowPreset('lung');
  };

  const drawViewport = useCallback((
    canvas: HTMLCanvasElement | null,
    plane: 'axial' | 'coronal' | 'sagittal' | 'mip'
  ) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Viewport background
    ctx.fillStyle = '#111315';
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.translate(W / 2 + pan.x, H / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-W / 2, -H / 2);

    if (scanImage && scanImage.complete && scanImage.naturalWidth > 0) {
      ctx.drawImage(scanImage, 0, 0, W, H);

      if (plane !== 'axial') {
        ctx.fillStyle = plane === 'mip' ? 'rgba(59, 130, 196, 0.12)' : 'rgba(15, 23, 42, 0.35)';
        ctx.fillRect(0, 0, W, H);
      }
    } else {
      ctx.beginPath();
      ctx.ellipse(W / 2, H / 2, W * 0.42, H * 0.38, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#181b1e';
      ctx.fill();
      ctx.strokeStyle = '#2a2e33';
      ctx.lineWidth = 1;
      ctx.stroke();

      [W * 0.32, W * 0.68].forEach(lx => {
        ctx.beginPath();
        ctx.ellipse(lx, H * 0.5, W * 0.14, H * 0.24, 0, 0, 2 * Math.PI);
        ctx.fillStyle = '#0f1113';
        ctx.fill();
        ctx.strokeStyle = '#22262a';
        ctx.stroke();
      });
    }

    // Nodule ROI Overlay
    if (showOverlays && activeCase.hasLesion) {
      const nx = activeCase.cxRel * W;
      const ny = activeCase.cyRel * H;
      const radius = 18;

      ctx.strokeStyle = activeCase.verdictClass === 'MALIGNANT' ? '#e05252' : '#3cb371';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(nx - radius, ny - radius, radius * 2, radius * 2);

      ctx.fillStyle = activeCase.verdictClass === 'MALIGNANT' ? 'rgba(224, 82, 82, 0.15)' : 'rgba(60, 179, 113, 0.15)';
      ctx.fillRect(nx - radius, ny - radius, radius * 2, radius * 2);

      ctx.fillStyle = activeCase.verdictClass === 'MALIGNANT' ? '#e05252' : '#3cb371';
      ctx.font = '10px monospace';
      ctx.fillText(`ROI: ${activeCase.crosshairLabel}`, nx - radius, ny - radius - 4);
    }

    // Crosshairs
    if (showCrosshairs) {
      const cx = W / 2;
      const cy = H / 2;
      ctx.strokeStyle = 'rgba(59, 130, 196, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);

      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
      ctx.moveTo(0, cy); ctx.lineTo(W, cy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();

    // Corner HUD Annotations
    ctx.font = '11px monospace';
    ctx.fillStyle = '#8b949e';

    ctx.textAlign = 'left';
    ctx.fillText(`PID: ${activeCase.pid}`, 10, 18);
    ctx.fillText(`CASE: ${activeCase.name.split('.')[0]}`, 10, 32);

    ctx.textAlign = 'right';
    ctx.fillText(plane.toUpperCase(), W - 10, 18);
    ctx.fillText(`Slice ${sliceIndex}/${maxSlices}`, W - 10, 32);

    ctx.textAlign = 'left';
    ctx.fillText(`W: ${PRESET_VALUES[windowPreset].w} L: ${PRESET_VALUES[windowPreset].l}`, 10, H - 24);
    ctx.fillText(`Preset: ${windowPreset.toUpperCase()}`, 10, H - 10);

    ctx.textAlign = 'right';
    ctx.fillText(`Zoom: ${(zoom * 100).toFixed(0)}%`, W - 10, H - 24);
    ctx.fillText(`1.25 mm`, W - 10, H - 10);
  }, [pan, zoom, scanImage, showOverlays, showCrosshairs, activeCase, sliceIndex, maxSlices, windowPreset]);

  useEffect(() => {
    drawViewport(axialRef.current, 'axial');
    drawViewport(coronalRef.current, 'coronal');
    drawViewport(sagittalRef.current, 'sagittal');
    drawViewport(mipRef.current, 'mip');
  }, [drawViewport]);

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
      setPan({
        x: e.clientX - dragStartRef.current.x,
        y: e.clientY - dragStartRef.current.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Top Reading Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--bg-panel)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '8px 12px',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Button
            size="sm"
            variant={activeTool === 'pan' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('pan')}
          >
            <Move size={13} style={{ marginRight: '4px' }} />
            Pan
          </Button>
          <Button
            size="sm"
            variant={activeTool === 'zoom' ? 'primary' : 'ghost'}
            onClick={() => {
              setActiveTool('zoom');
              setZoom(z => Math.min(4.0, z + 0.25));
            }}
          >
            <ZoomIn size={13} style={{ marginRight: '4px' }} />
            Zoom
          </Button>
          <Button
            size="sm"
            variant={activeTool === 'wl' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('wl')}
          >
            <Sun size={13} style={{ marginRight: '4px' }} />
            W/L
          </Button>
          <Button
            size="sm"
            variant={activeTool === 'measure' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('measure')}
          >
            <Ruler size={13} style={{ marginRight: '4px' }} />
            Measure
          </Button>
          <Button
            size="sm"
            variant={activeTool === 'roi' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('roi')}
          >
            <Square size={13} style={{ marginRight: '4px' }} />
            ROI
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={handleReset}
          >
            <RotateCcw size={13} style={{ marginRight: '4px' }} />
            Reset (R)
          </Button>
        </div>

        {/* W/L Presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Window:</span>
          {(['lung', 'mediastinum', 'bone'] as const).map(p => (
            <Button
              key={p}
              size="sm"
              variant={windowPreset === p ? 'secondary' : 'ghost'}
              onClick={() => setWindowPreset(p)}
            >
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </Button>
          ))}
        </div>

        {/* Case selector tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Button
            size="sm"
            variant={activeCase.id === PRIMARY_SAMPLE.id ? 'primary' : 'ghost'}
            onClick={() => setActiveCase(PRIMARY_SAMPLE)}
          >
            Case 1 (Malignant)
          </Button>
          <Button
            size="sm"
            variant={activeCase.id === SECONDARY_SAMPLE.id ? 'primary' : 'ghost'}
            onClick={() => setActiveCase(SECONDARY_SAMPLE)}
          >
            Case 2 (Benign)
          </Button>
        </div>
      </div>

      {/* Main 2-Column Workstation Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: '16px', alignItems: 'start' }}>
        {/* Left Column: 2x2 Viewports */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gridTemplateRows: '260px 260px',
              gap: '4px',
              background: 'var(--border)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              overflow: 'hidden',
              cursor: activeTool === 'pan' ? (isDragging ? 'grabbing' : 'grab') : 'crosshair',
            }}
          >
            <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
              <canvas ref={axialRef} width={400} height={260} style={{ width: '100%', height: '100%', display: 'block' }} />
            </div>
            <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
              <canvas ref={coronalRef} width={400} height={260} style={{ width: '100%', height: '100%', display: 'block' }} />
            </div>
            <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
              <canvas ref={sagittalRef} width={400} height={260} style={{ width: '100%', height: '100%', display: 'block' }} />
            </div>
            <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
              <canvas ref={mipRef} width={400} height={260} style={{ width: '100%', height: '100%', display: 'block' }} />
            </div>
          </div>

          {/* Slice Scroll Navigation */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Slice:</span>
            <input
              type="range"
              min={1}
              max={maxSlices}
              value={sliceIndex}
              onChange={e => setSliceIndex(Number(e.target.value))}
              style={{ flex: 1, accentColor: 'var(--accent)' }}
              aria-label="Slice"
            />
            <span className="tabular-nums" style={{ fontSize: '12px', color: 'var(--text-main)', width: '60px', textAlign: 'right' }}>
              {sliceIndex} / {maxSlices}
            </span>
          </div>
        </div>

        {/* Right Column: HU Histogram + Quantitative Morphometry + Findings */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* HU Histogram component */}
          <HUHistogram
            title={`Nodule Attenuation Histogram (${activeCase.pid})`}
            meanHU={activeCase.verdictClass === 'MALIGNANT' ? 38.4 : 12.2}
            stdHU={activeCase.verdictClass === 'MALIGNANT' ? 24.1 : 18.6}
            minHU={activeCase.verdictClass === 'MALIGNANT' ? -110 : -85}
            maxHU={activeCase.verdictClass === 'MALIGNANT' ? 92 : 68}
          />

          {/* Quantitative Morphometry Table */}
          <Panel title="Quantitative Lesion Descriptors" subtitle="Automated segmentation metrics">
            <Table>
              <Thead>
                <Tr>
                  <Th>Parameter</Th>
                  <Th>Value</Th>
                  <Th>Unit</Th>
                </Tr>
              </Thead>
              <Tbody>
                <Tr>
                  <Td style={{ fontWeight: 500 }}>Effective Diameter</Td>
                  <Td className="tabular-nums" style={{ fontWeight: 600 }}>{activeCase.crosshairLabel}</Td>
                  <Td style={{ color: 'var(--text-muted)' }}>mm</Td>
                </Tr>
                <Tr>
                  <Td style={{ fontWeight: 500 }}>Volume</Td>
                  <Td className="tabular-nums" style={{ fontWeight: 600 }}>{activeCase.area.toFixed(1)}</Td>
                  <Td style={{ color: 'var(--text-muted)' }}>mm³</Td>
                </Tr>
                <Tr>
                  <Td style={{ fontWeight: 500 }}>Density Index</Td>
                  <Td className="tabular-nums">{activeCase.densityPercent.toFixed(1)}%</Td>
                  <Td style={{ color: 'var(--text-muted)' }}>ratio</Td>
                </Tr>
                <Tr>
                  <Td style={{ fontWeight: 500 }}>Sphericity / Compactness</Td>
                  <Td className="tabular-nums">{activeCase.compactness.toFixed(2)}</Td>
                  <Td style={{ color: 'var(--text-muted)' }}>index [0–1]</Td>
                </Tr>
                <Tr>
                  <Td style={{ fontWeight: 500 }}>Margin Lobularity</Td>
                  <Td className="tabular-nums">{activeCase.lobularity.toFixed(2)}</Td>
                  <Td style={{ color: 'var(--text-muted)' }}>index [0–1]</Td>
                </Tr>
                <Tr>
                  <Td style={{ fontWeight: 500 }}>Anatomic Location</Td>
                  <Td colSpan={2} style={{ color: 'var(--text-main)' }}>{activeCase.lobe} ({activeCase.anatomicMark})</Td>
                </Tr>
              </Tbody>
            </Table>
          </Panel>

          {/* Diagnostic Assessment Panel */}
          <Panel title="Consensus Diagnostic Assessment" subtitle="3D Ensembles & Radiomics">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Consensus Verdict:</span>
                <Badge variant={activeCase.verdictClass === 'MALIGNANT' ? 'danger' : 'success'}>
                  {activeCase.verdictClass} ({(activeCase.verdictProb).toFixed(1)}%)
                </Badge>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Classification Category:</span>
                <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-main)' }}>{activeCase.severity}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Epistemic Uncertainty:</span>
                <Badge variant={activeCase.uncertainty === 'LOW' ? 'success' : activeCase.uncertainty === 'MODERATE' ? 'warning' : 'danger'}>
                  {activeCase.uncertainty} RISK
                </Badge>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', borderTop: '1px solid var(--border)', paddingTop: '8px', lineHeight: '1.4' }}>
                Notice: Research use only. Deep model consensus inference does not constitute standalone diagnostic proof.
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
};
