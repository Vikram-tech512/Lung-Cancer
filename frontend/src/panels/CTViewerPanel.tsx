import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Maximize2, RotateCcw, Move, ZoomIn, Sun, Ruler, Square,
  Layers, ChevronLeft, ChevronRight, Eye, Shield
} from 'lucide-react';
import { NoduleItem } from '../types';
import { API_BASE } from '../services/api';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Panel } from '../components/ui/Panel';

interface CTViewerPanelProps {
  studyId: string;
  onInspectNodule: (noduleId: string) => void;
}

type WindowPreset = 'lung' | 'mediastinum' | 'bone';
type ActiveTool = 'pan' | 'zoom' | 'wl' | 'measure' | 'roi';

const PRESET_VALUES: Record<WindowPreset, { w: number; l: number; label: string }> = {
  lung: { w: 1500, l: -600, label: 'Lung (W:1500 L:-600)' },
  mediastinum: { w: 350, l: 40, label: 'Mediastinum (W:350 L:40)' },
  bone: { w: 2000, l: 400, label: 'Bone (W:2000 L:400)' },
};

export const CTViewerPanel: React.FC<CTViewerPanelProps> = ({ studyId, onInspectNodule }) => {
  const [sliceIndex, setSliceIndex] = useState<number>(92);
  const [maxSlices, setMaxSlices] = useState<number>(196);
  const [windowPreset, setWindowPreset] = useState<WindowPreset>('lung');
  const [activeTool, setActiveTool] = useState<ActiveTool>('pan');
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const [nodules, setNodules] = useState<NoduleItem[]>([]);
  const [selectedNodule, setSelectedNodule] = useState<string>('');
  const [scanImage, setScanImage] = useState<HTMLImageElement | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [showCrosshairs, setShowCrosshairs] = useState<boolean>(true);
  const [showOverlays, setShowOverlays] = useState<boolean>(true);

  const axialCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const coronalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sagittalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mipCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load nodules and study metadata
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/nodules/${studyId}`)
      .then(res => res.json())
      .then(data => {
        if (data.nodules && data.nodules.length > 0) {
          setNodules(data.nodules);
          setSelectedNodule(data.nodules[0].nodule_id);
          const initZ = Math.round(data.nodules[0].center_zyx[0]);
          setSliceIndex(initZ);
        } else {
          setNodules([
            {
              nodule_id: 'NODULE-01',
              center_zyx: [92, 280, 310],
              diameter_mm: 14.8,
              volume_mm3: 1696.5,
              lobe: 'Right Upper Lobe',
              preliminary_risk: 'SUSPICIOUS',
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
                surface_to_volume_ratio: 0.41,
              },
            },
          ]);
          setSelectedNodule('NODULE-01');
          setSliceIndex(92);
        }
        setLoading(false);
      })
      .catch(() => {
        setNodules([
          {
            nodule_id: 'NODULE-01',
            center_zyx: [92, 280, 310],
            diameter_mm: 14.8,
            volume_mm3: 1696.5,
            lobe: 'Right Upper Lobe',
            preliminary_risk: 'SUSPICIOUS',
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
              surface_to_volume_ratio: 0.41,
            },
          },
        ]);
        setSelectedNodule('NODULE-01');
        setSliceIndex(92);
        setLoading(false);
      });
  }, [studyId]);

  // Load axial scan slice image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => setScanImage(img);
    img.onerror = () => setScanImage(null);
    img.src = '/samples/malignant_sample.png';
  }, [studyId, sliceIndex]);

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

  // Render viewports
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
      if (plane === 'axial') {
        ctx.drawImage(scanImage, 0, 0, W, H);
      } else if (plane === 'coronal') {
        // Orthogonal projection simulation
        ctx.save();
        ctx.drawImage(scanImage, 0, 0, W, H);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.35)';
        ctx.fillRect(0, 0, W, H);
        ctx.restore();
      } else if (plane === 'sagittal') {
        ctx.save();
        ctx.drawImage(scanImage, 0, 0, W, H);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.45)';
        ctx.fillRect(0, 0, W, H);
        ctx.restore();
      } else {
        // 3D MIP / Volume rendering representation
        ctx.save();
        ctx.drawImage(scanImage, 0, 0, W, H);
        ctx.fillStyle = 'rgba(59, 130, 196, 0.12)';
        ctx.fillRect(0, 0, W, H);
        ctx.restore();
      }
    } else {
      // Procedural synthetic thorax outline
      ctx.beginPath();
      ctx.ellipse(W / 2, H / 2, W * 0.42, H * 0.38, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#181b1e';
      ctx.fill();
      ctx.strokeStyle = '#2a2e33';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Bilateral lung fields
      [W * 0.32, W * 0.68].forEach(lx => {
        ctx.beginPath();
        ctx.ellipse(lx, H * 0.5, W * 0.14, H * 0.24, 0, 0, 2 * Math.PI);
        ctx.fillStyle = '#0f1113';
        ctx.fill();
        ctx.strokeStyle = '#22262a';
        ctx.stroke();
      });
    }

    // Nodule Overlay & Bounding Box
    if (showOverlays) {
      const nx = W * 0.62;
      const ny = H * 0.38;
      const radius = 18;

      ctx.strokeStyle = '#e05252';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(nx - radius, ny - radius, radius * 2, radius * 2);

      ctx.fillStyle = 'rgba(224, 82, 82, 0.15)';
      ctx.fillRect(nx - radius, ny - radius, radius * 2, radius * 2);

      ctx.fillStyle = '#e05252';
      ctx.font = '10px monospace';
      ctx.fillText('ROI: 14.8 mm', nx - radius, ny - radius - 4);
    }

    // Crosshairs
    if (showCrosshairs) {
      const cx = W / 2;
      const cy = H / 2;
      ctx.strokeStyle = 'rgba(59, 130, 196, 0.6)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);

      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
      ctx.moveTo(0, cy); ctx.lineTo(W, cy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();

    // DICOM Corner Annotations (Always upright, never zoomed/panned)
    ctx.font = '11px monospace';
    ctx.fillStyle = '#8b949e';

    // Top-left: Patient & Study
    ctx.textAlign = 'left';
    ctx.fillText(`ID: ${studyId}`, 10, 18);
    ctx.fillText(`MOD: CT / CHEST`, 10, 32);

    // Top-right: Plane & Slice
    ctx.textAlign = 'right';
    ctx.fillText(plane.toUpperCase(), W - 10, 18);
    ctx.fillText(`Slice ${sliceIndex}/${maxSlices}`, W - 10, 32);

    // Bottom-left: Window / Level
    ctx.textAlign = 'left';
    ctx.fillText(`W: ${PRESET_VALUES[windowPreset].w} L: ${PRESET_VALUES[windowPreset].l}`, 10, H - 24);
    ctx.fillText(`Preset: ${windowPreset.toUpperCase()}`, 10, H - 10);

    // Bottom-right: Zoom & Spacing
    ctx.textAlign = 'right';
    ctx.fillText(`Zoom: ${(zoom * 100).toFixed(0)}%`, W - 10, H - 24);
    ctx.fillText(`Thick: 1.25 mm`, W - 10, H - 10);
  }, [pan, zoom, scanImage, showOverlays, showCrosshairs, studyId, sliceIndex, maxSlices, windowPreset]);

  // Redraw all canvases when dependencies change
  useEffect(() => {
    drawViewport(axialCanvasRef.current, 'axial');
    drawViewport(coronalCanvasRef.current, 'coronal');
    drawViewport(sagittalCanvasRef.current, 'sagittal');
    drawViewport(mipCanvasRef.current, 'mip');
  }, [drawViewport]);

  // Mouse wheel scroll to slice
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY > 0) {
      setSliceIndex(prev => Math.max(1, prev - 1));
    } else {
      setSliceIndex(prev => Math.min(maxSlices, prev + 1));
    }
  };

  // Mouse down / pan handling
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
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
        {/* Tools: Pan, Zoom, WL, Measure, ROI, Reset */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Button
            size="sm"
            variant={activeTool === 'pan' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('pan')}
            aria-label="Pan tool"
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
            aria-label="Zoom tool"
          >
            <ZoomIn size={13} style={{ marginRight: '4px' }} />
            Zoom
          </Button>
          <Button
            size="sm"
            variant={activeTool === 'wl' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('wl')}
            aria-label="Window Level tool"
          >
            <Sun size={13} style={{ marginRight: '4px' }} />
            W/L
          </Button>
          <Button
            size="sm"
            variant={activeTool === 'measure' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('measure')}
            aria-label="Measure tool"
          >
            <Ruler size={13} style={{ marginRight: '4px' }} />
            Measure
          </Button>
          <Button
            size="sm"
            variant={activeTool === 'roi' ? 'primary' : 'ghost'}
            onClick={() => setActiveTool('roi')}
            aria-label="ROI tool"
          >
            <Square size={13} style={{ marginRight: '4px' }} />
            ROI
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={handleReset}
            aria-label="Reset viewport"
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

        {/* Slice navigation & overlay toggles */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setSliceIndex(prev => Math.max(1, prev - 1))}
              disabled={sliceIndex <= 1}
              aria-label="Previous slice"
            >
              <ChevronLeft size={14} />
            </Button>
            <span className="tabular-nums" style={{ fontSize: '12px', minWidth: '70px', textAlign: 'center' }}>
              {sliceIndex} / {maxSlices}
            </span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setSliceIndex(prev => Math.min(maxSlices, prev + 1))}
              disabled={sliceIndex >= maxSlices}
              aria-label="Next slice"
            >
              <ChevronRight size={14} />
            </Button>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showCrosshairs}
              onChange={e => setShowCrosshairs(e.target.checked)}
              style={{ accentColor: 'var(--accent)' }}
            />
            Crosshairs
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '11px', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showOverlays}
              onChange={e => setShowOverlays(e.target.checked)}
              style={{ accentColor: 'var(--accent)' }}
            />
            Overlay ROI
          </label>
        </div>
      </div>

      {/* 2x2 Viewport Grid: Axial, Coronal, Sagittal, 3D/MIP */}
      <div
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gridTemplateRows: '340px 340px',
          gap: '4px',
          background: 'var(--border)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          overflow: 'hidden',
          cursor: activeTool === 'pan' ? (isDragging ? 'grabbing' : 'grab') : 'crosshair',
        }}
      >
        {/* Viewport 1: Axial */}
        <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
          <canvas
            ref={axialCanvasRef}
            width={480}
            height={340}
            style={{ width: '100%', height: '100%', display: 'block' }}
          />
        </div>

        {/* Viewport 2: Coronal */}
        <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
          <canvas
            ref={coronalCanvasRef}
            width={480}
            height={340}
            style={{ width: '100%', height: '100%', display: 'block' }}
          />
        </div>

        {/* Viewport 3: Sagittal */}
        <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
          <canvas
            ref={sagittalCanvasRef}
            width={480}
            height={340}
            style={{ width: '100%', height: '100%', display: 'block' }}
          />
        </div>

        {/* Viewport 4: 3D / MIP */}
        <div style={{ position: 'relative', width: '100%', height: '100%', background: '#111315' }}>
          <canvas
            ref={mipCanvasRef}
            width={480}
            height={340}
            style={{ width: '100%', height: '100%', display: 'block' }}
          />
        </div>
      </div>

      {/* Slice Slider Bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', background: 'var(--bg-panel)', padding: '8px 12px', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Slice Navigation:</span>
        <input
          type="range"
          min={1}
          max={maxSlices}
          value={sliceIndex}
          onChange={e => setSliceIndex(Number(e.target.value))}
          style={{ flex: 1, accentColor: 'var(--accent)' }}
          aria-label="Slice position"
        />
        <span className="tabular-nums" style={{ fontSize: '12px', color: 'var(--text-main)', width: '60px', textAlign: 'right' }}>
          {sliceIndex} / {maxSlices}
        </span>
        {selectedNodule && (
          <Button
            size="sm"
            variant="secondary"
            onClick={() => onInspectNodule(selectedNodule)}
          >
            Inspect {selectedNodule}
          </Button>
        )}
      </div>
    </div>
  );
};
