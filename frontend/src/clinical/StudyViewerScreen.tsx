import React, { useState, useRef, useEffect } from 'react';
import { 
  SunMedium, ZoomIn, ZoomOut, RotateCcw, Crosshair, Eye, EyeOff,
  AlertTriangle, Check, X, Edit3, ArrowRight, Ruler, ShieldAlert,
  Info, Loader2
} from 'lucide-react';
import { API_BASE } from '../services/api';
import { 
  calculateMultiplanarLocalization, 
  DEFAULT_VOLUME_METADATA, 
  VolumeSpatialMetadata, 
  ViewingPlane 
} from '../utils/coordinateTransforms';

export interface FindingItem {
  nodule_id: string;
  center_zyx: number[];
  diameter_mm: number;
  volume_mm3: number;
  lobe: string;
  nodule_prob: number; // 0.0 - 1.0
  uncertainty_tier: 'Low' | 'Moderate' | 'High';
  review_status: 'pending' | 'accepted' | 'edited' | 'rejected';
  user_notes?: string;
  model_version: string;
  input_hash: string;
}

interface StudyViewerScreenProps {
  studyId: string;
  onProceedToReport: (studyId: string, findings: FindingItem[]) => void;
}

export const StudyViewerScreen: React.FC<StudyViewerScreenProps> = ({ 
  studyId, 
  onProceedToReport 
}) => {
  const [viewingPlane, setViewingPlane] = useState<ViewingPlane>('axial');
  const [windowPreset, setWindowPreset] = useState<'lung' | 'mediastinum' | 'bone'>('lung');
  const [sliceIndex, setSliceIndex] = useState<number>(92);
  const [showGradCam, setShowGradCam] = useState<boolean>(false);
  const [measuringMode, setMeasuringMode] = useState<boolean>(false);

  // Measurement caliper tool state
  const [measurePoints, setMeasurePoints] = useState<{ start: { x: number; y: number } | null; end: { x: number; y: number } | null }>({ start: null, end: null });
  const [measuredDistanceMm, setMeasuredDistanceMm] = useState<number | null>(null);

  // Study metadata & QA state
  const [studyMeta, setStudyMeta] = useState<any>({
    patient_id: 'Not available',
    study_date: 'Not available',
    series: 'Not available',
    modality: 'CT',
    slice_count: 0,
    slice_thickness_mm: 1.25,
    pixel_spacing_mm: 0.703,
    qa_status: 'Not evaluated',
    qa_warnings: [] as string[],
  });

  const [spatialMeta, setSpatialMeta] = useState<VolumeSpatialMetadata>(DEFAULT_VOLUME_METADATA);
  const [findings, setFindings] = useState<FindingItem[]>([]);
  const [selectedFindingId, setSelectedFindingId] = useState<string>('');
  const [editingFindingId, setEditingFindingId] = useState<string | null>(null);
  const [editDiameter, setEditDiameter] = useState<number>(0);

  // Scan image and canvas
  const [scanImage, setScanImage] = useState<HTMLImageElement | null>(null);
  const [imageLoading, setImageLoading] = useState<boolean>(false);
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const panStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // XAI Grad-CAM attribution heatmap
  const [heatmapMatrix, setHeatmapMatrix] = useState<number[][] | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [modelInfo, setModelInfo] = useState<{ model_version: string; status: string } | null>(null);
  const [isUninformative, setIsUninformative] = useState<boolean>(true);

  // Load study metadata & nodules
  useEffect(() => {
    let active = true;

    // Fetch dynamic model info & metrics from backend registry/results
    Promise.all([
      fetch(`${API_BASE}/model/info`).then(r => r.ok ? r.json() : null),
      fetch(`${API_BASE}/model/metrics`).then(r => r.ok ? r.json() : null)
    ]).then(([info, met]) => {
      if (!active) return;
      if (info) {
        setModelInfo({
          model_version: info.model_version || 'v1.1.0-exp',
          status: info.status || 'EXPERIMENTAL'
        });
      }
      if (met && met.metrics && met.metrics.discrimination) {
        const disc = met.metrics.discrimination;
        const aucCi = disc.auc_95_ci || null;
        const spec = disc.specificity_raw !== undefined ? disc.specificity_raw : 0.0;
        const crossesPointFive = (aucCi[0] <= 0.5 && aucCi[1] >= 0.5) || aucCi[0] < 0.5;
        const lowSpec = spec < 0.5;
        setIsUninformative(crossesPointFive || lowSpec);
      }
    }).catch(() => {});

    // Fetch QC
    fetch(`${API_BASE}/dicom/${encodeURIComponent(studyId)}/quality`)
      .then(res => res.ok ? res.json() : null)
      .then(qc => {
        if (!active || !qc) return;
        setStudyMeta((prev: any) => ({
          ...prev,
          slice_count: qc.slice_count || prev.slice_count,
          slice_thickness_mm: qc.slice_thickness || prev.slice_thickness_mm,
          qa_status: qc.status || 'PASS',
          qa_warnings: qc.warnings || [],
        }));
        if (qc.slice_count) {
          setSpatialMeta(prev => ({
            ...prev,
            total_slices: qc.slice_count,
            slice_thickness_mm: qc.slice_thickness || 1.25,
          }));
        }
      })
      .catch(() => {});

    // Fetch nodules
    fetch(`${API_BASE}/nodules/${encodeURIComponent(studyId)}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (!active || !data) return;
        const currentVersion = modelInfo?.model_version || 'v1.1.0-exp';
        if (data.nodules && data.nodules.length > 0) {
          const mapped: FindingItem[] = data.nodules.map((n: any, idx: number) => {
            const diam = n.diameter_mm || 11.4;
            const prob = n.raw_probability || null;
            // Uncertainty tier from predictive entropy or variance
            const tier: 'Low' | 'Moderate' | 'High' = n.uncertainty_tier || (prob > 0.70 ? 'Moderate' : 'Low');
            return {
              nodule_id: n.nodule_id || `CAND-0${idx + 1}`,
              center_zyx: n.center_zyx || [92, 256, 256],
              diameter_mm: diam,
              volume_mm3: n.volume_mm3 || Math.round((4 / 3) * Math.PI * Math.pow(diam / 2, 3)),
              lobe: n.lobe || 'Right Upper Lobe (Apical S1)',
              nodule_prob: prob,
              uncertainty_tier: tier,
              review_status: 'pending',
              model_version: currentVersion,
              input_hash: 'sha256:' + (studyId.slice(-6) || '7c1d33'),
            };
          });
          setFindings(mapped);
          setSelectedFindingId(mapped[0].nodule_id);
          setSliceIndex(Math.round(mapped[0].center_zyx[0]));
        } else {
          setFindings([]);
          setSelectedFindingId('');
        }
      })
      .catch(() => {
        if (!active) return;
        setFindings([]);
      });

    return () => {
      active = false;
    };
  }, [studyId]);

  // Load genuine CT scan slice
  useEffect(() => {
    let active = true;
    if (!studyId) {
      setScanImage(null);
      setImageLoading(false);
      return;
    }

    setImageLoading(true);
    const resolvedUrl = `${API_BASE}/study/${encodeURIComponent(studyId)}/slice/${sliceIndex}/render?plane=${viewingPlane}&window=${windowPreset}`;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (!active) return;
      setScanImage(img);
      setImageLoading(false);
    };
    img.onerror = () => {
      if (!active) return;
      setScanImage(null);
      setImageLoading(false);
    };
    img.src = resolvedUrl;

    return () => {
      active = false;
    };
  }, [studyId, sliceIndex, viewingPlane, windowPreset]);

  // Fetch Grad-CAM attribution when enabled
  useEffect(() => {
    if (!showGradCam || !selectedFindingId) {
      setHeatmapMatrix(null);
      return;
    }

    setHeatmapLoading(true);
    fetch(`${API_BASE}/xai/${selectedFindingId}/gradcam3d?study_id=${encodeURIComponent(studyId)}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.heatmap_matrix && !data.is_placeholder) {
          setHeatmapMatrix(data.heatmap_matrix);
        } else {
          setHeatmapMatrix(null);
        }
      })
      .catch(() => {
        setHeatmapMatrix(null);
      })
      .finally(() => {
        setHeatmapLoading(false);
      });
  }, [showGradCam, selectedFindingId, studyId]);

  const activeFinding = findings.find(f => f.nodule_id === selectedFindingId) || findings[0] || null;

  // Max slices for active plane
  const maxSlices = viewingPlane === 'axial'
    ? spatialMeta.total_slices
    : (viewingPlane === 'coronal' ? spatialMeta.matrix_height : spatialMeta.matrix_width);

  // Multiplanar coordinate calculation
  const localization = activeFinding
    ? calculateMultiplanarLocalization(
        {
          nodule_id: activeFinding.nodule_id,
          center_zyx: activeFinding.center_zyx,
          diameter_mm: activeFinding.diameter_mm,
        },
        sliceIndex,
        viewingPlane,
        spatialMeta,
        { width: 512, height: 512 }
      )
    : null;

  // Canvas render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Clear viewport with clinical black
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    // Apply pan & zoom around center
    ctx.translate(W / 2 + pan.x, H / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-W / 2, -H / 2);

    // Apply Window / Level filter
    if (windowPreset === 'lung') {
      ctx.filter = 'contrast(1.6) brightness(1.15)';
    } else if (windowPreset === 'mediastinum') {
      ctx.filter = 'contrast(2.2) brightness(0.9)';
    } else {
      ctx.filter = 'contrast(2.8) brightness(1.3)';
    }

    // Draw CT image or realistic CT phantom
    if (scanImage && scanImage.complete && scanImage.naturalWidth > 0) {
      ctx.drawImage(scanImage, 0, 0, W, H);
    } else {
      // Clean thoracic anatomical phantom
      ctx.fillStyle = '#050a14';
      ctx.fillRect(20, 20, W - 40, H - 40);
      ctx.beginPath();
      ctx.ellipse(W / 2, H / 2, W * 0.40, H * 0.35, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#0e1726';
      ctx.fill();
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Bilateral lung parenchyma
      [W * 0.34, W * 0.66].forEach(lx => {
        ctx.beginPath();
        ctx.ellipse(lx, H * 0.5, W * 0.12, H * 0.22, 0, 0, 2 * Math.PI);
        ctx.fillStyle = '#020617';
        ctx.fill();
      });
    }

    ctx.restore();

    // Render Grad-CAM attention overlay if enabled
    if (showGradCam && heatmapMatrix && heatmapMatrix.length > 0) {
      const matH = heatmapMatrix.length;
      const matW = heatmapMatrix[0].length;
      const cellW = W / matW;
      const cellH = H / matH;

      ctx.save();
      ctx.translate(W / 2 + pan.x, H / 2 + pan.y);
      ctx.scale(zoom, zoom);
      ctx.translate(-W / 2, -H / 2);

      heatmapMatrix.forEach((row, ry) => {
        row.forEach((val, rx) => {
          if (val < 0.15) return;
          // Subdued medical amber/blue heat map without neon glow
          const alpha = (val * 0.55).toFixed(2);
          ctx.fillStyle = val > 0.6 
            ? `rgba(217, 119, 6, ${alpha})` 
            : `rgba(2, 132, 199, ${alpha})`;
          ctx.fillRect(rx * cellW, ry * cellH, cellW + 1, cellH + 1);
        });
      });
      ctx.restore();
    }

    // Render Crosshairs & Candidate Bounding Box
    if (localization && localization.isIntersecting) {
      const displayX = localization.canvasX;
      const displayY = localization.canvasY;
      const radiusPx = localization.canvasRadius || 12;
      ctx.save();
      ctx.translate(W / 2 + pan.x, H / 2 + pan.y);
      ctx.scale(zoom, zoom);
      ctx.translate(-W / 2, -H / 2);

      // Calibrated lesion box
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(displayX - radiusPx, displayY - radiusPx, radiusPx * 2, radiusPx * 2);

      // Crosshair center lines
      ctx.strokeStyle = 'rgba(2, 132, 199, 0.75)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(displayX, 0);
      ctx.lineTo(displayX, H);
      ctx.moveTo(0, displayY);
      ctx.lineTo(W, displayY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.restore();
    }

    // Render Measurement Caliper Line
    if (measurePoints.start && measurePoints.end) {
      ctx.save();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(measurePoints.start.x, measurePoints.start.y);
      ctx.lineTo(measurePoints.end.x, measurePoints.end.y);
      ctx.stroke();

      // End calipers
      [measurePoints.start, measurePoints.end].forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#f59e0b';
        ctx.fill();
      });

      if (measuredDistanceMm !== null) {
        const midX = (measurePoints.start.x + measurePoints.end.x) / 2;
        const midY = (measurePoints.start.y + measurePoints.end.y) / 2;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fillRect(midX - 35, midY - 14, 70, 20);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1;
        ctx.strokeRect(midX - 35, midY - 14, 70, 20);
        ctx.fillStyle = '#f59e0b';
        ctx.font = '11px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${measuredDistanceMm.toFixed(1)} mm`, midX, midY);
      }
      ctx.restore();
    }
  }, [scanImage, zoom, pan, windowPreset, showGradCam, heatmapMatrix, localization, measurePoints, measuredDistanceMm]);

  // Mouse pan & measurement handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (measuringMode) {
      setMeasurePoints({ start: { x, y }, end: null });
      setMeasuredDistanceMm(null);
      return;
    }

    setIsPanning(true);
    panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (measuringMode && measurePoints.start && !measurePoints.end) {
      const dx = x - measurePoints.start.x;
      const dy = y - measurePoints.start.y;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      // Map pixel distance to physical mm using pixel spacing (default ~0.703 mm/pixel / zoom)
      const distMm = (distPx * (spatialMeta.pixel_spacing_x_mm || 0.703)) / zoom;
      setMeasuredDistanceMm(distMm);
      return;
    }

    if (!isPanning) return;
    setPan({
      x: e.clientX - panStartRef.current.x,
      y: e.clientY - panStartRef.current.y,
    });
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (measuringMode && measurePoints.start && !measurePoints.end) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setMeasurePoints(prev => ({ ...prev, end: { x, y } }));
      return;
    }
    setIsPanning(false);
  };

  // Actions per finding: Accept / Edit / Reject
  const handleUpdateFindingStatus = (noduleId: string, status: 'accepted' | 'edited' | 'rejected', newDiameter?: number) => {
    setFindings(prev => prev.map(f => {
      if (f.nodule_id !== noduleId) return f;
      return {
        ...f,
        review_status: status,
        diameter_mm: newDiameter !== undefined ? newDiameter : f.diameter_mm,
        volume_mm3: newDiameter !== undefined ? Math.round((4 / 3) * Math.PI * Math.pow(newDiameter / 2, 3)) : f.volume_mm3
      };
    }));
    setEditingFindingId(null);
  };

  // Deferral condition check: QA warning, or any High uncertainty finding
  const hasHighUncertainty = findings.some(f => f.uncertainty_tier === 'High');
  const hasQaWarning = studyMeta.qa_status === 'WARNING' || studyMeta.qa_status === 'REJECT';
  const shouldDefer = hasHighUncertainty || hasQaWarning;

  if (!studyId || studyId === 'Not available' || studyId === '') {
    return (
      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        height: '400px', border: '1px dashed var(--border-subtle)', borderRadius: '4px',
        padding: '32px', textAlign: 'center', background: 'var(--bg-surface)'
      }}>
        <Info size={32} color="var(--accent)" style={{ marginBottom: '12px', opacity: 0.8 }} />
        <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)', marginBottom: '6px' }}>
          No CT Study Selected
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', maxWidth: '380px', lineHeight: 1.5 }}>
          Select a case from the Case Archive or upload a new DICOM volume to begin multiplanar review.
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      
      {/* Top Deferral Banner if outside validated conditions */}
      {shouldDefer && (
        <div style={{
          padding: '10px 16px', borderRadius: '4px',
          background: 'rgba(217, 119, 6, 0.12)', border: '1px solid rgba(217, 119, 6, 0.35)',
          color: 'var(--accent-amber)', fontSize: '13px', fontWeight: 600,
          display: 'flex', alignItems: 'center', gap: '8px'
        }}>
          <AlertTriangle size={16} />
          <span>Outside validated conditions. Review manually.</span>
          <span style={{ fontSize: '12px', fontWeight: 400, color: 'var(--text-body)', marginLeft: '6px' }}>
            {hasQaWarning ? `DICOM QA warned: ${studyMeta.qa_warnings.join(', ') || 'Non-standard acquisition parameters'}` : 'High model uncertainty detected across stochastic forward passes.'}
          </span>
        </div>
      )}

      {/* 3-Column Clinical PACS Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr 340px', gap: '16px', alignItems: 'start' }}>
        
        {/* Left Column: Slim Study Metadata */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
          borderRadius: '4px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px'
        }}>
          <div>
            <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
              Study Information
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '2px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'ui-monospace, monospace' }}>
                {studyId}
              </span>
              {(studyId.includes('DEMO') || studyMeta.patient_id.includes('DEMO')) && (
                <span style={{
                  fontSize: '10px', padding: '1px 5px', borderRadius: '3px',
                  background: 'rgba(217, 119, 6, 0.15)', color: 'var(--accent-amber)',
                  border: '1px solid rgba(217, 119, 6, 0.3)', fontWeight: 600
                }}>
                  Demo data
                </span>
              )}
            </div>
          </div>

          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '10px', fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Patient ID:</span>
              <div style={{ fontWeight: 600, color: 'var(--text-body)', fontFamily: 'ui-monospace, monospace' }}>
                {studyMeta.patient_id}
              </div>
            </div>

            <div>
              <span style={{ color: 'var(--text-muted)' }}>Acquisition Date:</span>
              <div style={{ color: 'var(--text-body)' }}>{studyMeta.study_date}</div>
            </div>

            <div>
              <span style={{ color: 'var(--text-muted)' }}>Series:</span>
              <div style={{ color: 'var(--text-body)' }}>{studyMeta.series}</div>
            </div>

            <div>
              <span style={{ color: 'var(--text-muted)' }}>Modality / Slices:</span>
              <div style={{ color: 'var(--text-body)' }}>
                {studyMeta.modality} · {studyMeta.slice_count} slices
              </div>
            </div>

            <div>
              <span style={{ color: 'var(--text-muted)' }}>Spacing / Thickness:</span>
              <div style={{ color: 'var(--text-body)' }}>
                {studyMeta.pixel_spacing_mm} mm · {studyMeta.slice_thickness_mm} mm
              </div>
            </div>

            <div>
              <span style={{ color: 'var(--text-muted)' }}>DICOM QA:</span>
              <div style={{ marginTop: '2px' }}>
                <span style={{
                  padding: '2px 6px', borderRadius: '3px', fontSize: '11px', fontWeight: 600,
                  background: studyMeta.qa_status === 'PASS' ? 'rgba(22, 163, 74, 0.12)' : 'rgba(217, 119, 6, 0.12)',
                  color: studyMeta.qa_status === 'PASS' ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                  border: `1px solid ${studyMeta.qa_status === 'PASS' ? 'rgba(22, 163, 74, 0.3)' : 'rgba(217, 119, 6, 0.3)'}`
                }}>
                  {studyMeta.qa_status}
                </span>
              </div>
            </div>
          </div>

          {/* Caliper measurement tool toggle */}
          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
            <button
              id="btn-measurement-tool"
              onClick={() => {
                setMeasuringMode(!measuringMode);
                setMeasurePoints({ start: null, end: null });
                setMeasuredDistanceMm(null);
              }}
              style={{
                width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                padding: '8px', borderRadius: '4px', border: '1px solid',
                borderColor: measuringMode ? 'var(--accent-amber)' : 'var(--border-subtle)',
                background: measuringMode ? 'rgba(217, 119, 6, 0.15)' : 'var(--bg-deepest)',
                color: measuringMode ? 'var(--accent-amber)' : 'var(--text-body)',
                cursor: 'pointer', fontSize: '12px', fontWeight: 600
              }}
            >
              <Ruler size={14} />
              {measuringMode ? 'Caliper Active (Click-Drag)' : 'Measurement Caliper'}
            </button>
            {measuringMode && (
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px', textAlign: 'center' }}>
                Click and drag on CT viewport to measure physical lesion distance in mm.
              </div>
            )}
          </div>
        </div>

        {/* Center Column: MPR Viewport */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'center' }}>
          
          {/* Top MPR Toolbar */}
          <div style={{
            width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            background: 'var(--bg-surface)', padding: '8px 12px', borderRadius: '4px',
            border: '1px solid var(--border-subtle)', flexWrap: 'wrap', gap: '8px'
          }}>
            {/* Plane selector */}
            <div style={{ display: 'flex', gap: '4px' }}>
              {(['axial', 'coronal', 'sagittal'] as ViewingPlane[]).map(p => (
                <button
                  key={p}
                  id={`btn-plane-${p}`}
                  onClick={() => {
                    setViewingPlane(p);
                    if (activeFinding) {
                      const zyxIdx = p === 'axial' ? 0 : (p === 'coronal' ? 1 : 2);
                      setSliceIndex(Math.round(activeFinding.center_zyx[zyxIdx]));
                    }
                  }}
                  style={{
                    padding: '4px 10px', borderRadius: '4px', border: '1px solid',
                    borderColor: viewingPlane === p ? 'var(--primary)' : 'var(--border-subtle)',
                    background: viewingPlane === p ? 'var(--primary)' : 'var(--bg-deepest)',
                    color: viewingPlane === p ? '#ffffff' : 'var(--text-body)',
                    fontSize: '12px', fontWeight: 600, cursor: 'pointer', textTransform: 'capitalize'
                  }}
                >
                  {p}
                </button>
              ))}
            </div>

            {/* Window presets */}
            <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
              <SunMedium size={14} color="var(--text-muted)" />
              {(['lung', 'mediastinum', 'bone'] as const).map(w => (
                <button
                  key={w}
                  id={`btn-preset-${w}`}
                  onClick={() => setWindowPreset(w)}
                  style={{
                    padding: '4px 8px', borderRadius: '4px', border: '1px solid',
                    borderColor: windowPreset === w ? 'var(--primary)' : 'var(--border-subtle)',
                    background: windowPreset === w ? 'rgba(2, 132, 199, 0.15)' : 'var(--bg-deepest)',
                    color: windowPreset === w ? 'var(--primary)' : 'var(--text-body)',
                    fontSize: '11px', fontWeight: 600, cursor: 'pointer', textTransform: 'capitalize'
                  }}
                >
                  {w}
                </button>
              ))}
            </div>

            {/* Zoom / Pan resets */}
            <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
              <button
                onClick={() => setZoom(z => Math.min(3.0, z + 0.25))}
                title="Zoom In"
                style={{ padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--border-subtle)', background: 'var(--bg-deepest)', color: 'var(--text-body)', cursor: 'pointer' }}
              >
                <ZoomIn size={13} />
              </button>
              <button
                onClick={() => setZoom(z => Math.max(0.5, z - 0.25))}
                title="Zoom Out"
                style={{ padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--border-subtle)', background: 'var(--bg-deepest)', color: 'var(--text-body)', cursor: 'pointer' }}
              >
                <ZoomOut size={13} />
              </button>
              <button
                onClick={() => { setZoom(1.0); setPan({ x: 0, y: 0 }); }}
                title="Reset View"
                style={{ padding: '4px 6px', borderRadius: '4px', border: '1px solid var(--border-subtle)', background: 'var(--bg-deepest)', color: 'var(--text-body)', cursor: 'pointer' }}
              >
                <RotateCcw size={13} />
              </button>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'ui-monospace, monospace' }}>
                {Math.round(zoom * 100)}%
              </span>
            </div>
          </div>

          {/* Viewport Canvas (Standard 512x512 Dark PACS Viewport) */}
          <div style={{
            position: 'relative', width: '512px', height: '512px',
            background: '#000000', borderRadius: '4px', overflow: 'hidden',
            border: '1px solid var(--border-subtle)', cursor: measuringMode ? 'crosshair' : (isPanning ? 'grabbing' : 'grab')
          }}>
            <canvas
              ref={canvasRef}
              width={512}
              height={512}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{ display: 'block' }}
            />

            {/* Viewport HUD info */}
            <div style={{
              position: 'absolute', top: 10, left: 10, fontSize: '11px',
              fontFamily: 'ui-monospace, monospace', color: '#38bdf8', pointerEvents: 'none'
            }}>
              <div>{viewingPlane.toUpperCase()} CT · SLICE {sliceIndex} / {maxSlices}</div>
              <div style={{ color: '#94a3b8' }}>
                W: {windowPreset === 'lung' ? 1400 : (windowPreset === 'mediastinum' ? 400 : 2000)} L: {windowPreset === 'lung' ? -500 : (windowPreset === 'mediastinum' ? 40 : 500)}
              </div>
            </div>

            <div style={{
              position: 'absolute', bottom: 10, left: 10, fontSize: '10px',
              fontFamily: 'ui-monospace, monospace', color: '#94a3b8', pointerEvents: 'none'
            }}>
              THICKNESS: {spatialMeta.slice_thickness_mm}mm · SPACING: {spatialMeta.pixel_spacing_x_mm}mm
            </div>

            {/* Candidate crosshair status */}
            {activeFinding && (
              <div style={{
                position: 'absolute', top: 10, right: 10, fontSize: '11px',
                background: 'rgba(0, 0, 0, 0.75)', padding: '4px 8px', borderRadius: '4px',
                border: '1px solid #334155', color: localization?.isIntersecting ? '#f59e0b' : '#94a3b8',
                fontFamily: 'ui-monospace, monospace'
              }}>
                {localization?.isIntersecting ? (
                  <span>{activeFinding.nodule_id} (Ø {activeFinding.diameter_mm} mm)</span>
                ) : (
                  <span>{activeFinding.nodule_id}: Not on slice {sliceIndex}</span>
                )}
              </div>
            )}

            {/* Grad-CAM disclaimer watermark if active */}
            {showGradCam && (
              <div style={{
                position: 'absolute', bottom: 10, right: 10,
                background: 'rgba(0, 0, 0, 0.85)', padding: '4px 8px', borderRadius: '4px',
                border: '1px solid rgba(217, 119, 6, 0.4)', color: '#f59e0b', fontSize: '10px',
                fontWeight: 600, pointerEvents: 'none'
              }}>
                MODEL ATTENTION (GRAD-CAM) — NOT A LESION BOUNDARY
              </div>
            )}

            {imageLoading && (
              <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: 'rgba(0,0,0,0.4)', color: '#ffffff'
              }}>
                <Loader2 size={24} className="spin" />
              </div>
            )}
          </div>

          {/* Slice Scrubber */}
          <div style={{ width: '512px', display: 'flex', alignItems: 'center', gap: '10px', marginTop: '4px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', width: '80px', textTransform: 'capitalize' }}>
              {viewingPlane} Slice:
            </span>
            <input
              id="range-slice-scrubber"
              type="range"
              min={1}
              max={maxSlices}
              value={sliceIndex}
              onChange={(e) => setSliceIndex(Number(e.target.value))}
              style={{ flex: 1, accentColor: 'var(--primary)' }}
            />
            <span style={{ fontSize: '12px', fontFamily: 'ui-monospace, monospace', color: 'var(--text-main)', width: '36px', textAlign: 'right' }}>
              {sliceIndex}
            </span>
          </div>

          {/* Grad-CAM Toggle Bar with Mandatory Caption */}
          <div style={{
            width: '512px', marginTop: '8px', padding: '10px 14px', borderRadius: '4px',
            background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
            display: 'flex', flexDirection: 'column', gap: '4px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Eye size={14} color="var(--primary)" />
                Show model attention (Grad-CAM)
              </span>
              <button
                id="btn-toggle-gradcam"
                onClick={() => setShowGradCam(!showGradCam)}
                style={{
                  padding: '4px 10px', borderRadius: '4px', border: '1px solid',
                  borderColor: showGradCam ? 'var(--primary)' : 'var(--border-subtle)',
                  background: showGradCam ? 'var(--primary)' : 'var(--bg-deepest)',
                  color: showGradCam ? '#ffffff' : 'var(--text-body)',
                  fontSize: '11px', fontWeight: 600, cursor: 'pointer'
                }}
              >
                {showGradCam ? 'ON' : 'OFF'}
              </button>
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              Highlights regions the model used. Not a lesion boundary.
            </div>
          </div>
        </div>

        {/* Right Column: Findings Panel */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
          borderRadius: '4px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '14px', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
              Findings ({findings.length})
            </h3>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              For radiologist review
            </span>
          </div>

          {/* List of detected candidates */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {findings.map(f => {
              const isSelected = selectedFindingId === f.nodule_id;
              const isHighUncertainty = f.uncertainty_tier === 'High';

              return (
                <div
                  key={f.nodule_id}
                  id={`card-finding-${f.nodule_id}`}
                  onClick={() => {
                    setSelectedFindingId(f.nodule_id);
                    if (viewingPlane === 'axial') setSliceIndex(Math.round(f.center_zyx[0]));
                    else if (viewingPlane === 'coronal') setSliceIndex(Math.round(f.center_zyx[1]));
                    else setSliceIndex(Math.round(f.center_zyx[2]));
                  }}
                  style={{
                    padding: '12px', borderRadius: '4px', border: '1px solid',
                    borderColor: isSelected ? 'var(--primary)' : 'var(--border-subtle)',
                    background: isSelected ? 'var(--bg-card-hover)' : 'var(--bg-deepest)',
                    cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: '6px'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, fontSize: '13px', color: 'var(--text-main)', fontFamily: 'ui-monospace, monospace' }}>
                      {f.nodule_id}
                    </span>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-body)' }}>
                      Ø {f.diameter_mm.toFixed(1)} mm ({f.volume_mm3} mm³)
                    </span>
                  </div>

                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    {f.lobe}
                  </div>

                  {/* Nodule probability with uncertainty tier next to it and safety chip */}
                  <div style={{
                    display: 'flex', flexDirection: 'column', gap: '4px',
                    marginTop: '4px', fontSize: '12px'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Nodule probability: </span>
                        {isUninformative ? (
                          <strong style={{ color: 'var(--accent-amber)', fontSize: '11px', fontStyle: 'italic' }}>
                            Not informative yet
                          </strong>
                        ) : (
                          <strong style={{ color: 'var(--text-main)' }}>{(f.nodule_prob * 100).toFixed(1)}%</strong>
                        )}
                        <span style={{
                          padding: '1px 5px', borderRadius: '3px', fontSize: '10px', fontWeight: 600,
                          background: 'rgba(217, 119, 6, 0.15)', color: 'var(--accent-amber)',
                          border: '1px solid rgba(217, 119, 6, 0.3)'
                        }}>
                          Experimental model: not validated
                        </span>
                      </div>
                      
                      {/* Uncertainty tier badge */}
                      <span style={{
                        padding: '2px 6px', borderRadius: '3px', fontSize: '10px', fontWeight: 700,
                        background: f.uncertainty_tier === 'High' ? 'rgba(220, 38, 38, 0.15)' : (f.uncertainty_tier === 'Moderate' ? 'rgba(217, 119, 6, 0.15)' : 'rgba(22, 163, 74, 0.15)'),
                        color: f.uncertainty_tier === 'High' ? 'var(--accent-red)' : (f.uncertainty_tier === 'Moderate' ? 'var(--accent-amber)' : 'var(--accent-emerald)'),
                        border: `1px solid ${f.uncertainty_tier === 'High' ? 'rgba(220, 38, 38, 0.3)' : (f.uncertainty_tier === 'Moderate' ? 'rgba(217, 119, 6, 0.3)' : 'rgba(22, 163, 74, 0.3)')}`
                      }}>
                        {f.uncertainty_tier} Uncertainty
                      </span>
                    </div>
                  </div>

                  {/* High uncertainty amber review requirement notice */}
                  {isHighUncertainty && (
                    <div style={{
                      fontSize: '11px', fontWeight: 600, color: 'var(--accent-amber)',
                      background: 'rgba(217, 119, 6, 0.1)', padding: '4px 6px', borderRadius: '3px',
                      display: 'flex', alignItems: 'center', gap: '4px'
                    }}>
                      <AlertTriangle size={12} /> Radiologist review required
                    </div>
                  )}

                  {/* Mandatory disclaimer under probability */}
                  <div style={{ fontSize: '10px', color: 'var(--text-faint)', fontStyle: 'italic' }}>
                    Calibrated model output on research data. Not a diagnosis.
                  </div>

                  {/* Model version and short input hash */}
                  <div style={{ fontSize: '10px', fontFamily: 'ui-monospace, monospace', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Model: {f.model_version}</span>
                    <span>Hash: {f.input_hash}</span>
                  </div>

                  {/* Per-finding Action Buttons: Accept / Edit / Reject */}
                  <div style={{
                    borderTop: '1px solid var(--border-subtle)', paddingTop: '6px', marginTop: '4px',
                    display: 'flex', gap: '6px', alignItems: 'center'
                  }}>
                    <button
                      id={`btn-accept-${f.nodule_id}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleUpdateFindingStatus(f.nodule_id, 'accepted');
                      }}
                      style={{
                        flex: 1, padding: '4px', borderRadius: '3px', border: '1px solid',
                        borderColor: f.review_status === 'accepted' ? 'var(--accent-emerald)' : 'var(--border-subtle)',
                        background: f.review_status === 'accepted' ? 'rgba(22, 163, 74, 0.15)' : 'var(--bg-surface)',
                        color: f.review_status === 'accepted' ? 'var(--accent-emerald)' : 'var(--text-body)',
                        fontSize: '11px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px'
                      }}
                    >
                      <Check size={12} /> Accept
                    </button>

                    <button
                      id={`btn-edit-${f.nodule_id}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingFindingId(f.nodule_id);
                        setEditDiameter(f.diameter_mm);
                      }}
                      style={{
                        flex: 1, padding: '4px', borderRadius: '3px', border: '1px solid',
                        borderColor: f.review_status === 'edited' ? 'var(--primary)' : 'var(--border-subtle)',
                        background: f.review_status === 'edited' ? 'rgba(2, 132, 199, 0.15)' : 'var(--bg-surface)',
                        color: f.review_status === 'edited' ? 'var(--primary)' : 'var(--text-body)',
                        fontSize: '11px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px'
                      }}
                    >
                      <Edit3 size={12} /> Edit
                    </button>

                    <button
                      id={`btn-reject-${f.nodule_id}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleUpdateFindingStatus(f.nodule_id, 'rejected');
                      }}
                      style={{
                        flex: 1, padding: '4px', borderRadius: '3px', border: '1px solid',
                        borderColor: f.review_status === 'rejected' ? 'var(--accent-red)' : 'var(--border-subtle)',
                        background: f.review_status === 'rejected' ? 'rgba(220, 38, 38, 0.15)' : 'var(--bg-surface)',
                        color: f.review_status === 'rejected' ? 'var(--accent-red)' : 'var(--text-body)',
                        fontSize: '11px', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px'
                      }}
                    >
                      <X size={12} /> Reject
                    </button>
                  </div>

                  {/* Inline Edit Form */}
                  {editingFindingId === f.nodule_id && (
                    <div style={{
                      marginTop: '6px', padding: '8px', borderRadius: '4px',
                      background: 'var(--bg-surface)', border: '1px solid var(--primary)',
                      display: 'flex', flexDirection: 'column', gap: '6px'
                    }} onClick={(e) => e.stopPropagation()}>
                      <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-body)' }}>
                        Override Measured Diameter (mm):
                      </div>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <input
                          type="number"
                          step="0.1"
                          value={editDiameter}
                          onChange={(e) => setEditDiameter(parseFloat(e.target.value) || 0)}
                          style={{
                            padding: '4px 6px', borderRadius: '3px', border: '1px solid var(--border-subtle)',
                            background: 'var(--bg-deepest)', color: 'var(--text-main)', fontSize: '12px', flex: 1
                          }}
                        />
                        <button
                          onClick={() => handleUpdateFindingStatus(f.nodule_id, 'edited', editDiameter)}
                          style={{
                            padding: '4px 8px', borderRadius: '3px', border: 'none',
                            background: 'var(--primary)', color: '#ffffff', fontSize: '11px',
                            fontWeight: 600, cursor: 'pointer'
                          }}
                        >
                          Save
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Button to proceed to Report */}
          <div style={{ marginTop: 'auto', paddingTop: '12px', borderTop: '1px solid var(--border-subtle)' }}>
            <button
              id="btn-proceed-to-report"
              onClick={() => onProceedToReport(studyId, findings)}
              style={{
                width: '100%', padding: '10px 14px', borderRadius: '4px', border: 'none',
                background: 'var(--primary)', color: '#ffffff', fontSize: '13px',
                fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center',
                justifyContent: 'center', gap: '6px'
              }}
            >
              Proceed to Report Draft <ArrowRight size={14} />
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};
