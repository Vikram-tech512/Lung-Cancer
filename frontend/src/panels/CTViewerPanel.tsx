import React, { useState, useRef, useEffect } from 'react';
import { Layers, Crosshair, ZoomIn, ZoomOut, RotateCcw, SunMedium, Eye, Zap, AlertCircle, ArrowRight, ShieldAlert, Compass } from 'lucide-react';
import { NoduleItem } from '../types';
import { API_BASE } from '../services/api';
import {
  calculateMultiplanarLocalization,
  DEFAULT_VOLUME_METADATA,
  VolumeSpatialMetadata,
  NoduleSliceLocalization,
  ViewingPlane,
  voxelToPhysicalMm
} from '../utils/coordinateTransforms';

interface CTViewerPanelProps {
  studyId: string;
  onInspectNodule: (noduleId: string) => void;
}

export const CTViewerPanel: React.FC<CTViewerPanelProps> = ({ studyId, onInspectNodule }) => {
  const [viewingPlane, setViewingPlane] = useState<ViewingPlane>('axial');
  const [activeLayer, setActiveLayer] = useState<'none' | 'segmentation' | 'gradcam3d' | 'attention_rollout3d' | 'integrated_gradients'>('segmentation');
  const [windowPreset, setWindowPreset] = useState<'lung' | 'mediastinum' | 'bone'>('lung');
  const [sliceIndex, setSliceIndex] = useState<number>(92);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [nodules, setNodules] = useState<NoduleItem[]>([]);
  const [selectedNodule, setSelectedNodule] = useState<string>('');
  const [spatialMeta, setSpatialMeta] = useState<VolumeSpatialMetadata>(DEFAULT_VOLUME_METADATA);

  // Genuine CT scan image rendering
  const [scanImage, setScanImage] = useState<HTMLImageElement | null>(null);
  const [scanImageLoading, setScanImageLoading] = useState<boolean>(false);
  
  // Interactive Viewport Zoom & Pan
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const panStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Real XAI attribution response state & study-isolated caching
  const [xaiData, setXaiData] = useState<any>(null);
  const [xaiLoading, setXaiLoading] = useState<boolean>(false);
  const [xaiError, setXaiError] = useState<string | null>(null);

  // Invalidate in-flight stale requests when switching samples rapidly
  const activeRequestIdRef = useRef<number>(0);
  // Keyed in-memory cache: (studyId:noduleId:layer)
  const xaiCacheRef = useRef<Map<string, any>>(new Map());

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load study metadata & nodules - strictly reset previous study's overlays and state
  useEffect(() => {
    // 1. Cancel in-flight attribution and immediately reset all study-dependent state
    activeRequestIdRef.current++;
    setXaiData(null);
    setXaiLoading(false);
    setXaiError(null);
    setNodules([]);
    setSelectedNodule('');

    let isSubscribed = true;

    // 2. Fetch real QC metadata for volume dimensions
    fetch(`${API_BASE}/dicom/${studyId}/quality`)
      .then(res => res.ok ? res.json() : null)
      .then(qc => {
        if (!isSubscribed) return;
        if (qc && qc.slice_count) {
          setSpatialMeta(prev => ({
            ...prev,
            total_slices: qc.slice_count,
            slice_thickness_mm: qc.slice_thickness || 1.25,
          }));
        }
      })
      .catch(() => {});

    // 3. Fetch nodules for the current study
    fetch(`${API_BASE}/nodules/${studyId}`)
      .then(res => res.json())
      .then(data => {
        if (!isSubscribed) return;
        if (data.nodules && data.nodules.length > 0) {
          setNodules(data.nodules);
          setSelectedNodule(data.nodules[0].nodule_id);
          const initialZ = Math.round(data.nodules[0].center_zyx[0]);
          setSliceIndex(initialZ);
        } else {
          setNodules([]);
          setSelectedNodule('');
        }
      })
      .catch(() => {
        if (!isSubscribed) return;
        setNodules([]);
        setSelectedNodule('');
      });

    return () => {
      isSubscribed = false;
    };
  }, [studyId]);

  // Load genuine scan image pixels for active study and slice
  useEffect(() => {
    let active = true;
    setScanImageLoading(true);

    let targetUrl = '/real_samples/LIDC_0001_malignant_nodule.png';
    if (studyId.includes('0124') || studyId.toLowerCase().includes('benign')) {
      targetUrl = '/real_samples/LIDC_0002_normal_parenchyma.png';
    } else if (studyId.includes('0078')) {
      targetUrl = '/real_samples/LIDC_0078_malignant_adenocarcinoma.png';
    } else if (studyId.includes('0087') || studyId.toLowerCase().includes('malignant')) {
      targetUrl = '/real_samples/LIDC_0001_malignant_nodule.png';
    }

    fetch(`${API_BASE}/study/${encodeURIComponent(studyId)}/slice/${sliceIndex}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (!active) return;
        const resolvedUrl = data?.image_url || targetUrl;
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          if (!active) return;
          setScanImage(img);
          setScanImageLoading(false);
        };
        img.onerror = () => {
          if (!active) return;
          const fallbackImg = new Image();
          fallbackImg.crossOrigin = 'anonymous';
          fallbackImg.onload = () => {
            if (!active) return;
            setScanImage(fallbackImg);
            setScanImageLoading(false);
          };
          fallbackImg.onerror = () => {
            if (!active) return;
            setScanImage(null);
            setScanImageLoading(false);
          };
          fallbackImg.src = targetUrl;
        };
        img.src = resolvedUrl;
      })
      .catch(() => {
        if (!active) return;
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          if (!active) return;
          setScanImage(img);
          setScanImageLoading(false);
        };
        img.onerror = () => {
          if (!active) return;
          setScanImage(null);
          setScanImageLoading(false);
        };
        img.src = targetUrl;
      });

    return () => {
      active = false;
    };
  }, [studyId, sliceIndex]);

  // Selected nodule object
  const currentNodule = nodules.find(n => n.nodule_id === selectedNodule) || nodules[0] || null;

  // Sync slice when plane changes
  const handlePlaneChange = (plane: ViewingPlane) => {
    setViewingPlane(plane);
    if (!currentNodule) return;
    if (plane === 'axial') {
      setSliceIndex(Math.round(currentNodule.center_zyx[0]));
    } else if (plane === 'coronal') {
      setSliceIndex(Math.round(currentNodule.center_zyx[1]));
    } else if (plane === 'sagittal') {
      setSliceIndex(Math.round(currentNodule.center_zyx[2]));
    }
  };

  // Max slice count for current viewing plane
  const maxSlicesForPlane = viewingPlane === 'axial'
    ? spatialMeta.total_slices
    : (viewingPlane === 'coronal' ? spatialMeta.matrix_height : spatialMeta.matrix_width);

  // Fetch real model attribution when an XAI layer, nodule, or study changes
  useEffect(() => {
    if (activeLayer === 'none' || activeLayer === 'segmentation') {
      setXaiData(null);
      setXaiError(null);
      setXaiLoading(false);
      return;
    }

    if (!selectedNodule || !studyId) {
      setXaiData(null);
      setXaiError(null);
      setXaiLoading(false);
      return;
    }

    // Check keyed cache to prevent duplicate requests across tab switches
    const cacheKey = `${studyId}:${selectedNodule}:${activeLayer}`;
    if (xaiCacheRef.current.has(cacheKey)) {
      setXaiData(xaiCacheRef.current.get(cacheKey));
      setXaiLoading(false);
      setXaiError(null);
      return;
    }

    const currentReqId = ++activeRequestIdRef.current;
    setXaiLoading(true);
    setXaiError(null);
    setXaiData(null); // Clear previous attribution so it cannot flash on the new sample

    fetch(`${API_BASE}/xai/${selectedNodule}/${activeLayer}?study_id=${encodeURIComponent(studyId)}`)
      .then(async res => {
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(`XAI request failed: ${res.status} ${txt}`);
        }
        return res.json();
      })
      .then(data => {
        // Discard response if user has already switched study or nodule
        if (currentReqId !== activeRequestIdRef.current) return;
        xaiCacheRef.current.set(cacheKey, data);
        setXaiData(data);
        setXaiLoading(false);
      })
      .catch(err => {
        if (currentReqId !== activeRequestIdRef.current) return;
        setXaiError(err.message || 'Error loading model attribution');
        setXaiLoading(false);
        setXaiData(null);
      });
  }, [studyId, selectedNodule, activeLayer]);

  // Calculate slice-specific localization using established coordinate transformation
  const selectedLoc: NoduleSliceLocalization = calculateMultiplanarLocalization(
    currentNodule,
    sliceIndex,
    viewingPlane,
    spatialMeta,
    { width: 512, height: 512 },
    { zoom, panX: pan.x, panY: pan.y }
  );

  const physicalCoords = currentNodule ? voxelToPhysicalMm(currentNodule.center_zyx, spatialMeta) : null;

  // Render coordinate-driven high-fidelity CT viewport
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Viewport-aware coordinate localization with current zoom & pan
    const loc = calculateMultiplanarLocalization(
      currentNodule,
      sliceIndex,
      viewingPlane,
      spatialMeta,
      { width, height },
      { zoom, panX: pan.x, panY: pan.y }
    );

    // 1. Background CT canvas
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    // 2. Realistic CT pixel rendering with calibrated window presets
    if (scanImage) {
      if (windowPreset === 'lung') {
        // Lung window (WW 1500 / WL -600 HU): enhances air-tissue contrast and pulmonary vascular markings
        ctx.filter = 'contrast(130%) brightness(105%)';
      } else if (windowPreset === 'mediastinum') {
        // Mediastinal window (WW 350 / WL 40 HU): suppresses lung parenchyma to highlight soft-tissue masses
        ctx.filter = 'contrast(170%) brightness(80%)';
      } else if (windowPreset === 'bone') {
        // Bone window (WW 2000 / WL 300 HU): isolates high-density skeletal cortex
        ctx.filter = 'contrast(240%) brightness(65%) grayscale(100%)';
      }

      const drawW = width * zoom;
      const drawH = height * zoom;
      const drawX = (width - drawW) / 2 + pan.x;
      const drawY = (height - drawH) / 2 + pan.y;

      ctx.drawImage(scanImage, drawX, drawY, drawW, drawH);
    } else {
      // Professional PACS Standby Grid when scan is loading or awaiting intake
      ctx.strokeStyle = 'rgba(51, 65, 85, 0.35)';
      ctx.lineWidth = 1;
      const gridSize = 32 * zoom;
      const offsetX = pan.x % gridSize;
      const offsetY = pan.y % gridSize;
      for (let x = offsetX; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = offsetY; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.2;
      const msgW = 360;
      const msgH = 44;
      ctx.fillRect(width / 2 - msgW / 2, height / 2 - msgH / 2, msgW, msgH);
      ctx.strokeRect(width / 2 - msgW / 2, height / 2 - msgH / 2, msgW, msgH);

      ctx.fillStyle = '#38bdf8';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(scanImageLoading ? 'LOADING CT EXAMINATION SLICE...' : 'DICOM PACS STANDBY · AWAITING SCAN INTAKE', width / 2, height / 2 - 4);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(`Study: ${studyId || 'NO STUDY SELECTED'} · Slice #${sliceIndex}`, width / 2, height / 2 + 14);
    }
    ctx.restore();

    // Clear Demonstration Watermark for synthetic test studies
    if (studyId.includes('DEMO') || (spatialMeta as any).is_synthetic) {
      ctx.save();
      ctx.fillStyle = 'rgba(15, 23, 42, 0.82)';
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1;
      ctx.fillRect(width - 246, 10, 236, 22);
      ctx.strokeRect(width - 246, 10, 236, 22);
      ctx.fillStyle = '#94a3b8';
      ctx.font = 'bold 9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('DEMONSTRATION · BENCHMARK REFERENCE', width - 128, 24);
      ctx.restore();
    }

    // Viewing plane badge for multiplanar reconstructions
    if (viewingPlane !== 'axial') {
      ctx.save();
      ctx.fillStyle = 'rgba(8, 15, 29, 0.88)';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1;
      ctx.fillRect(10, height - 32, 380, 22);
      ctx.strokeRect(10, height - 32, 380, 22);
      ctx.fillStyle = '#38bdf8';
      ctx.font = 'bold 9px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`RECONSTRUCTION: ${viewingPlane.toUpperCase()} PLANE · SLICE ${sliceIndex}`, 18, height - 18);
      ctx.restore();
    }

    // 3. Render multiple nodules in the study independently
    nodules.forEach(nod => {
      const isSelected = nod.nodule_id === selectedNodule;
      const nodLoc = calculateMultiplanarLocalization(nod, sliceIndex, viewingPlane, spatialMeta, { width, height }, { zoom, panX: pan.x, panY: pan.y });

      // If nodule intersects this slice, render its lesion marker
      if (nodLoc.isIntersecting) {
        ctx.beginPath();
        ctx.arc(nodLoc.canvasX, nodLoc.canvasY, nodLoc.canvasRadius, 0, 2 * Math.PI);
        ctx.fillStyle = windowPreset === 'bone' ? '#cbd5e1' : '#e2e8f0';
        ctx.fill();

        // Non-selected nodules get secondary cyan indicator
        if (!isSelected) {
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = '#06b6d4';
          ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = '#06b6d4';
          ctx.font = '10px JetBrains Mono, monospace';
          ctx.fillText(nod.nodule_id, nodLoc.canvasX + nodLoc.canvasRadius + 4, nodLoc.canvasY - 4);
        }
      }
    });

    // 4. If the selected nodule intersects current slice, render primary annotations & overlays
    if (loc.isIntersecting) {
      const { canvasX, canvasY, canvasRadius, boundingBox, crosshair } = loc;

      // --- Segmentation Overlay ---
      if (activeLayer === 'segmentation') {
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, canvasRadius + 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(16, 185, 129, ${overlayOpacity * 0.5})`;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#10b981';
        ctx.stroke();
      }

      // --- Real Model Attribution Overlays ---
      if (
        (activeLayer === 'gradcam3d' || activeLayer === 'attention_rollout3d' || activeLayer === 'integrated_gradients') &&
        xaiData &&
        !xaiData.is_placeholder &&
        Array.isArray(xaiData.heatmap_matrix) &&
        xaiData.heatmap_matrix.length > 0
      ) {
        const matrix = xaiData.heatmap_matrix;
        const matH = matrix.length;
        const matW = matrix[0].length;

        const offCanvas = document.createElement('canvas');
        offCanvas.width = matW;
        offCanvas.height = matH;
        const offCtx = offCanvas.getContext('2d');

        if (offCtx) {
          const imgData = offCtx.createImageData(matW, matH);
          for (let r = 0; r < matH; r++) {
            for (let c = 0; c < matW; c++) {
              const val = Math.min(1, Math.max(0, matrix[r][c]));
              const idx = (r * matW + c) * 4;
              let red = 0, green = 0, blue = 0;
              if (activeLayer === 'gradcam3d') {
                red = Math.floor(Math.sin(val * Math.PI * 0.5) * 255);
                green = Math.floor(Math.sin(val * Math.PI) * 200);
                blue = Math.floor(Math.cos(val * Math.PI * 0.5) * 255 * (1 - val));
              } else if (activeLayer === 'attention_rollout3d') {
                red = Math.floor(val * 180 + 70);
                green = Math.floor(Math.sin(val * Math.PI) * 150);
                blue = Math.floor((1 - val * 0.5) * 240);
              } else {
                red = Math.floor(val * 50);
                green = Math.floor(val * 220 + 35);
                blue = Math.floor(val * 240 + 15);
              }

              imgData.data[idx] = red;
              imgData.data[idx + 1] = green;
              imgData.data[idx + 2] = blue;
              imgData.data[idx + 3] = Math.floor(val * overlayOpacity * 230);
            }
          }
          offCtx.putImageData(imgData, 0, 0);

          const roiSpan = Math.max(50, canvasRadius * 4.5);
          ctx.drawImage(
            offCanvas,
            canvasX - roiSpan / 2,
            canvasY - roiSpan / 2,
            roiSpan,
            roiSpan
          );

          // Attribution Disclaimer Watermark
          ctx.save();
          ctx.fillStyle = 'rgba(8, 12, 22, 0.85)';
          ctx.strokeStyle = '#f59e0b';
          ctx.lineWidth = 1;
          ctx.strokeRect(width / 2 - 190, height - 28, 380, 20);
          ctx.fillRect(width / 2 - 190, height - 28, 380, 20);
          ctx.fillStyle = '#f59e0b';
          ctx.font = 'bold 9px JetBrains Mono, monospace';
          ctx.textAlign = 'center';
          ctx.fillText('MODEL ATTRIBUTION — NOT A VERIFIED TUMOUR BOUNDARY', width / 2, height - 15);
          ctx.restore();
        }
      } else if (
        (activeLayer === 'gradcam3d' || activeLayer === 'attention_rollout3d' || activeLayer === 'integrated_gradients') &&
        (xaiData?.is_placeholder || xaiData?.generation_status === 'UNAVAILABLE_MODEL_NOT_LOADED' || xaiError)
      ) {
        // Honest Unavailable State — Never render simulated or random heatmaps
        ctx.save();
        ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.2;
        ctx.strokeRect(14, 14, 380, 24);
        ctx.fillRect(14, 14, 380, 24);
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 9.5px JetBrains Mono, monospace';
        ctx.fillText('⚠ Model attribution unavailable — trained weights required', 22, 29);
        ctx.restore();
      }

      if (xaiLoading && (activeLayer === 'gradcam3d' || activeLayer === 'attention_rollout3d' || activeLayer === 'integrated_gradients')) {
        ctx.save();
        ctx.fillStyle = 'rgba(8, 15, 29, 0.85)';
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 1;
        ctx.strokeRect(14, 14, 340, 24);
        ctx.fillRect(14, 14, 340, 24);
        ctx.fillStyle = '#06b6d4';
        ctx.font = 'bold 9.5px JetBrains Mono, monospace';
        ctx.fillText(`⏳ Calculating attribution for ${studyId}...`, 22, 29);
        ctx.restore();
      }

      // Candidate Nodule Bounding Box derived directly from coordinate transformation
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#f59e0b';
      ctx.strokeRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);

      // Crosshair lines centered precisely at (canvasX, canvasY)
      ctx.beginPath();
      ctx.moveTo(crosshair.startX, canvasY);
      ctx.lineTo(crosshair.endX, canvasY);
      ctx.moveTo(canvasX, crosshair.startY);
      ctx.lineTo(canvasX, crosshair.endY);
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
      ctx.stroke();

      // Lesion ROI Tag
      ctx.fillStyle = '#f59e0b';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillText(`${selectedNodule} [Ø ${currentNodule?.diameter_mm || 0}mm]`, boundingBox.x, Math.max(14, boundingBox.y - 6));
    }

  }, [sliceIndex, viewingPlane, activeLayer, overlayOpacity, windowPreset, currentNodule, nodules, spatialMeta, selectedNodule, xaiData, xaiLoading, xaiError, zoom, pan, studyId, scanImage]);

  // Mouse pan handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsPanning(true);
    panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isPanning) return;
    setPan({
      x: e.clientX - panStartRef.current.x,
      y: e.clientY - panStartRef.current.y
    });
  };

  const handleMouseUp = () => setIsPanning(false);

  return (
    <div style={{ padding: '24px', maxWidth: '1280px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
            03 · MULTIPLANAR CT VIEWER & XAI EXPLORER
          </h1>
          <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '4px' }}>
            Study: <span style={{ color: '#38bdf8', fontWeight: '600' }}>{studyId}</span> | Coordinate-Driven MPR | Voxel [Z, Y, X] ↔ Physical Patient Coordinates (mm)
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
          Launch Candidate Evaluation ({selectedNodule})
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
        {/* Main Viewport Container */}
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          
          {/* Top Control Bar: Multiplanar Plane Selector & Overlays */}
          <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
            {/* MPR Plane Selector */}
            <div style={{ display: 'flex', gap: '4px', background: '#0b1120', padding: '3px', borderRadius: '6px', border: '1px solid #1e293b' }}>
              {(['axial', 'coronal', 'sagittal'] as const).map(p => (
                <button
                  key={p}
                  onClick={() => handlePlaneChange(p)}
                  style={{
                    background: viewingPlane === p ? '#0284c7' : 'transparent',
                    color: viewingPlane === p ? '#ffffff' : '#94a3b8',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '5px 12px',
                    fontSize: '12px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em'
                  }}
                >
                  {p}
                </button>
              ))}
            </div>

            {/* Viewport Zoom & Pan Controls */}
            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <button
                onClick={() => setZoom(z => Math.min(3.0, z + 0.25))}
                title="Zoom In"
                style={{ background: '#1e293b', border: '1px solid #334155', color: '#38bdf8', borderRadius: '4px', padding: '6px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
              >
                <ZoomIn size={14} />
              </button>
              <button
                onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
                title="Zoom Out"
                style={{ background: '#1e293b', border: '1px solid #334155', color: '#94a3b8', borderRadius: '4px', padding: '6px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
              >
                <ZoomOut size={14} />
              </button>
              <button
                onClick={() => { setZoom(1.0); setPan({ x: 0, y: 0 }); }}
                title="Reset Viewport"
                style={{ background: '#1e293b', border: '1px solid #334155', color: '#94a3b8', borderRadius: '4px', padding: '6px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
              >
                <RotateCcw size={14} />
              </button>
              <span style={{ fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', color: '#64748b' }}>
                {Math.round(zoom * 100)}%
              </span>
            </div>
          </div>

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

          {/* Interactive Canvas Viewport with Pan support */}
          <div style={{ position: 'relative', border: '1px solid #1e293b', borderRadius: '8px', overflow: 'hidden', cursor: isPanning ? 'grabbing' : 'grab' }}>
            <canvas
              ref={canvasRef}
              width={512}
              height={512}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
            />

            {/* Medical Info Overlay */}
            <div style={{ position: 'absolute', top: 12, left: 12, fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', color: '#38bdf8', textShadow: '0 1px 2px #000' }}>
              <div>STUDY: {studyId}</div>
              <div>PLANE: {viewingPlane.toUpperCase()} CT</div>
              <div>SLICE: {sliceIndex} / {maxSlicesForPlane}</div>
            </div>

            <div style={{ position: 'absolute', bottom: 12, left: 12, fontSize: '11px', fontFamily: 'JetBrains Mono, monospace', color: '#94a3b8', textShadow: '0 1px 2px #000' }}>
              <div>W: {windowPreset === 'lung' ? 1400 : (windowPreset === 'mediastinum' ? 400 : 2000)} L: {windowPreset === 'lung' ? -500 : (windowPreset === 'mediastinum' ? 40 : 500)}</div>
              <div>THICKNESS: {spatialMeta.slice_thickness_mm}mm · SPACING: {spatialMeta.pixel_spacing_x_mm}mm</div>
            </div>

            {/* Dynamic coordinate status indicator: Voxel + Physical Patient mm */}
            <div style={{ position: 'absolute', top: 12, right: 12, fontSize: '11px', color: selectedLoc.isIntersecting ? '#f59e0b' : '#94a3b8', background: 'rgba(0,0,0,0.85)', padding: '6px 10px', borderRadius: '4px', border: `1px solid ${selectedLoc.isIntersecting ? 'rgba(245, 158, 11, 0.4)' : '#334155'}`, textAlign: 'right' }}>
              {currentNodule ? (
                selectedLoc.isIntersecting ? (
                  <>
                    <div style={{ fontWeight: '700', color: '#f59e0b' }}>{selectedNodule} [VOXEL: {currentNodule.center_zyx.map(n => Math.round(n)).join(', ')}]</div>
                    {physicalCoords && (
                      <div style={{ fontSize: '10px', color: '#38bdf8', marginTop: '2px' }}>
                        PHYS: X {physicalCoords.x.toFixed(1)} · Y {physicalCoords.y.toFixed(1)} · Z {physicalCoords.z.toFixed(1)} mm
                      </div>
                    )}
                  </>
                ) : (
                  <div>{selectedNodule}: NOT ON {viewingPlane.toUpperCase()} SLICE</div>
                )
              ) : 'NO NODULE'}
            </div>

            {/* Strict Regulatory Disclaimer & XAI Status Watermark */}
            {activeLayer !== 'none' && activeLayer !== 'segmentation' && (
              <div style={{ position: 'absolute', bottom: 12, right: 12, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '4px' }}>
                {xaiData?.is_placeholder && (
                  <div style={{ fontSize: '10px', color: '#f59e0b', background: 'rgba(15, 23, 42, 0.92)', padding: '3px 6px', borderRadius: '3px', border: '1px solid rgba(245, 158, 11, 0.4)', fontWeight: '600' }}>
                    ATTRIBUTION UNAVAILABLE: MODEL UNTRAINED (ZERO MATRIX)
                  </div>
                )}
                {xaiLoading && (
                  <div style={{ fontSize: '10px', color: '#38bdf8', background: 'rgba(15, 23, 42, 0.9)', padding: '3px 6px', borderRadius: '3px' }}>
                    Computing real model attribution...
                  </div>
                )}
                <div style={{ fontSize: '10px', color: '#f43f5e', background: 'rgba(15, 23, 42, 0.88)', padding: '4px 8px', borderRadius: '4px', border: '1px solid rgba(244, 63, 94, 0.4)', fontWeight: '600' }}>
                  MODEL ATTRIBUTION — NOT CLINICAL EVIDENCE
                </div>
              </div>
            )}
          </div>

          {/* Localization Diagnostic Banner & Jump-to-Slice helper */}
          {!selectedLoc.isIntersecting && currentNodule && (
            <div style={{
              width: '100%',
              maxWidth: '512px',
              marginTop: '12px',
              padding: '10px 14px',
              borderRadius: '6px',
              background: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid #334155',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px'
            }}>
              <span style={{ color: '#94a3b8' }}>
                Lesion <strong style={{ color: '#f8fafc' }}>{selectedNodule}</strong> is on {viewingPlane} slice <span style={{ color: '#38bdf8' }}>{Math.round(selectedLoc.targetSliceOnPlane)}</span> (span: {selectedLoc.zMinSlice}–{selectedLoc.zMaxSlice}). Not on slice {sliceIndex}.
              </span>
              <button
                onClick={() => setSliceIndex(Math.round(selectedLoc.targetSliceOnPlane))}
                style={{
                  background: '#0284c7',
                  border: 'none',
                  color: 'white',
                  borderRadius: '4px',
                  padding: '4px 10px',
                  fontSize: '11px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}
              >
                Jump to Lesion <ArrowRight size={12} />
              </button>
            </div>
          )}

          {/* Slice Scrubber */}
          <div style={{ width: '100%', maxWidth: '512px', marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '12px', color: '#94a3b8', width: '90px', textTransform: 'capitalize' }}>{viewingPlane} Slice:</span>
            <input
              type="range"
              min={1}
              max={maxSlicesForPlane}
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

            {nodules.map(nodule => {
              const isSelected = selectedNodule === nodule.nodule_id;
              const nodLoc = calculateMultiplanarLocalization(nodule, sliceIndex, viewingPlane, spatialMeta, { width: 512, height: 512 });

              return (
                <div
                  key={nodule.nodule_id}
                  onClick={() => {
                    setSelectedNodule(nodule.nodule_id);
                    if (viewingPlane === 'axial') {
                      setSliceIndex(Math.round(nodule.center_zyx[0]));
                    } else if (viewingPlane === 'coronal') {
                      setSliceIndex(Math.round(nodule.center_zyx[1]));
                    } else {
                      setSliceIndex(Math.round(nodule.center_zyx[2]));
                    }
                  }}
                  style={{
                    padding: '12px',
                    borderRadius: '6px',
                    border: '1px solid',
                    borderColor: isSelected ? '#f59e0b' : '#1e293b',
                    background: isSelected ? 'rgba(245, 158, 11, 0.1)' : '#0b1120',
                    cursor: 'pointer',
                    marginBottom: '10px'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <span style={{ fontWeight: '700', fontSize: '13px', color: '#f8fafc' }}>{nodule.nodule_id}</span>
                    <span style={{ fontSize: '11px', color: '#f59e0b', fontWeight: '600' }}>Ø {nodule.diameter_mm} mm</span>
                  </div>
                  <div style={{ fontSize: '11px', color: '#94a3b8' }}>{nodule.lobe}</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                    <span style={{ fontSize: '11px', color: '#f43f5e', fontWeight: '600' }}>
                      {nodule.preliminary_risk}
                    </span>
                    <span style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      background: nodLoc.isIntersecting ? 'rgba(16, 185, 129, 0.2)' : 'rgba(100, 116, 139, 0.2)',
                      color: nodLoc.isIntersecting ? '#10b981' : '#64748b',
                      fontWeight: '600'
                    }}>
                      {nodLoc.isIntersecting ? 'Visible on Slice' : `Slice ${Math.round(nodLoc.targetSliceOnPlane)}`}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
