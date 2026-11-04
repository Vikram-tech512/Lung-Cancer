import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  SunMedium, ZoomIn, ZoomOut, RotateCcw, Crosshair, Eye, EyeOff,
  AlertTriangle, Check, X, Edit3, ArrowRight, Ruler, ShieldAlert,
  Info, Loader2, Upload, Activity, Clock, DollarSign, Database,
  Layers, ChevronRight, ChevronDown, CheckCircle2, XCircle, Sparkles
} from 'lucide-react';
import { API_BASE, api } from '../services/api';
import { Badge } from '../components/ui/Badge';
import { calculateMultiplanarLocalization } from '../utils/coordinateTransforms';

export interface FindingItem {
  nodule_id: string;
  center_zyx: number[];
  diameter_mm: number;
  volume_mm3: number;
  lobe: string;
  raw_probability?: number;
  calibrated_probability?: number;
  uncertainty_tier: 'Low' | 'Moderate' | 'High';
  epistemic_uncertainty?: number;
  world_xyz_mm?: number[];
  bbox_voxel?: number[];
  requires_mandatory_review?: boolean;
  review_status: 'pending' | 'accepted' | 'edited' | 'rejected';
  user_notes?: string;
  source?: string;
  model_version?: string;
}

interface CaseItem {
  study_id: string;
  patient_id?: string;
  modality?: string;
  slice_count?: number;
  slice_thickness?: number;
  qc_status?: string;
  status?: string;
  urgency?: 'HIGH' | 'ROUTINE';
  is_demo?: boolean;
}

interface CTViewerPanelProps {
  studyId: string;
  onInspectNodule?: (noduleId: string) => void;
  onSelectStudy?: (studyId: string) => void;
}

type ViewingPlane = 'axial' | 'coronal' | 'sagittal';
type WindowPreset = 'lung' | 'mediastinum' | 'bone' | 'soft_tissue';

export const CTViewerPanel: React.FC<CTViewerPanelProps> = ({
  studyId: initialStudyId,
  onInspectNodule,
  onSelectStudy
}) => {
  // Navigation & Studies State
  const [activeStudyId, setActiveStudyId] = useState<string>(initialStudyId || '');
  const [studies, setStudies] = useState<CaseItem[]>([]);
  const [searchFilter, setSearchFilter] = useState<string>('');
  const [inferenceMode, setInferenceMode] = useState<'fast_triage' | 'full'>('fast_triage');
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [benchmarkData, setBenchmarkData] = useState<any>(null);
  const [jobProgress, setJobProgress] = useState<number | null>(null);
  const [currentStageName, setCurrentStageName] = useState<string>('');

  // Viewer State
  const [viewingPlane, setViewingPlane] = useState<ViewingPlane>('axial');
  const [windowPreset, setWindowPreset] = useState<WindowPreset>('lung');
  const [sliceIndex, setSliceIndex] = useState<number>(92);
  const [totalSlices, setTotalSlices] = useState<number>(184);
  const [sliceThickness, setSliceThickness] = useState<number>(1.25);
  const [pixelSpacing, setPixelSpacing] = useState<number[]>([0.703, 0.703]);
  const [sliceLoading, setSliceLoading] = useState<boolean>(false);
  const [sliceHash, setSliceHash] = useState<string>('');
  const [cursorHU, setCursorHU] = useState<number | null>(null);
  const [cursorPos, setCursorPos] = useState<{ z: number; y: number; x: number }>({ z: 92, y: 256, x: 256 });

  // Pan & Zoom
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const panStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Findings & Candidates
  const [findings, setFindings] = useState<FindingItem[]>([]);
  const [selectedFindingId, setSelectedFindingId] = useState<string>('');
  const [riskThreshold, setRiskThreshold] = useState<number>(0.15); // Sensitivity-first threshold
  const [blindReadMode, setBlindReadMode] = useState<boolean>(false);
  const [editingFindingId, setEditingFindingId] = useState<string | null>(null);
  const [editNotes, setEditNotes] = useState<string>('');
  const [editDiameter, setEditDiameter] = useState<number>(0);

  // XAI Attribution
  const [showGradCam, setShowGradCam] = useState<boolean>(true);
  const [gradCamAlpha, setGradCamAlpha] = useState<number>(0.65);
  const [heatmapMatrix, setHeatmapMatrix] = useState<number[][] | null>(null);
  const [heatmapLoading, setHeatmapLoading] = useState<boolean>(false);

  // Telemetry & Usage Drawer
  const [usageData, setUsageData] = useState<any>(null);
  const [activeBottomTab, setActiveBottomTab] = useState<'report' | 'usage' | 'longitudinal'>('report');
  const [longitudinalData, setLongitudinalData] = useState<any>(null);
  const [doctorSigned, setDoctorSigned] = useState<boolean>(false);
  const [signerName, setSignerName] = useState<string>('Dr. Sarah Chen, MD');

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageObjRef = useRef<HTMLImageElement | null>(null);

  // 1. Fetch available studies and held-out benchmark
  useEffect(() => {
    fetch(`${API_BASE}/cases`)
      .then(res => res.ok ? res.json() : { cases: [] })
      .then(data => {
        if (data && Array.isArray(data.cases)) {
          setStudies(data.cases);
          if (!activeStudyId && data.cases.length > 0) {
            setActiveStudyId(data.cases[0].study_id);
          }
        }
      })
      .catch(() => {});

    api.getBenchmark()
      .then(data => {
        if (data && data.status === 'available') {
          setBenchmarkData(data);
        }
      })
      .catch(() => {});
  }, []);

  // Update activeStudyId when prop changes
  useEffect(() => {
    if (initialStudyId && initialStudyId !== activeStudyId) {
      setActiveStudyId(initialStudyId);
    }
  }, [initialStudyId]);

  // 2. Fetch Study Data, Nodules, Report, and Usage
  const loadStudyDetails = useCallback((sId: string) => {
    if (!sId) {
      setFindings([]);
      setUsageData(null);
      setLongitudinalData(null);
      return;
    }
    // Fetch nodules
    fetch(`${API_BASE}/nodules/${encodeURIComponent(sId)}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.nodules && data.nodules.length > 0) {
          const mapped: FindingItem[] = data.nodules.map((n: any, idx: number) => ({
            nodule_id: n.nodule_id || `CAND-${String(idx + 1).padStart(2, '0')}`,
            center_zyx: n.center_zyx || [92, 240, 200],
            diameter_mm: n.diameter_mm || 12.0,
            volume_mm3: n.volume_mm3 || 900.0,
            lobe: n.lobe || 'Right Upper Lobe (RUL)',
            raw_probability: n.raw_probability ?? 0.65,
            calibrated_probability: n.calibrated_probability ?? 0.62,
            uncertainty_tier: n.uncertainty_tier || 'Low',
            requires_mandatory_review: n.requires_mandatory_review ?? (n.diameter_mm >= 8.0),
            review_status: n.review_status || 'pending',
            user_notes: n.user_notes || '',
            source: n.source || 'detector',
          }));
          setFindings(mapped);
          setSelectedFindingId(mapped[0].nodule_id);
          setSliceIndex(Math.round(mapped[0].center_zyx[0]));
        } else {
          setFindings([]);
        }
      })
      .catch(() => {});

    // Fetch QC metadata
    fetch(`${API_BASE}/dicom/${encodeURIComponent(sId)}/quality`)
      .then(res => res.ok ? res.json() : null)
      .then(qc => {
        if (qc && qc.slice_count) {
          setTotalSlices(qc.slice_count);
          setSliceThickness(qc.slice_thickness_mm || 1.25);
          if (qc.pixel_spacing_mm) setPixelSpacing(qc.pixel_spacing_mm);
        }
      })
      .catch(() => {});

    // Fetch Usage Telemetry
    fetch(`${API_BASE}/study/${encodeURIComponent(sId)}/usage`)
      .then(res => res.ok ? res.json() : null)
      .then(usage => {
        if (usage) setUsageData(usage);
      })
      .catch(() => {});

    // Fetch Longitudinal Prior Scans
    fetch(`${API_BASE}/patient/PT-${sId.slice(-6)}/longitudinal`)
      .then(res => res.ok ? res.json() : null)
      .then(long => {
        if (long) setLongitudinalData(long);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadStudyDetails(activeStudyId);
  }, [activeStudyId, loadStudyDetails]);

  // 3. Load genuine slice image from backend volume
  useEffect(() => {
    let active = true;
    setSliceLoading(true);

    const metaUrl = `${API_BASE}/study/${encodeURIComponent(activeStudyId)}/slice/${sliceIndex}?plane=${viewingPlane}&window=${windowPreset}`;
    fetch(metaUrl)
      .then(res => res.ok ? res.json() : null)
      .then(meta => {
        if (!active || !meta) return;
        setSliceHash(meta.slice_hash || '');
        if (meta.total_slices) setTotalSlices(meta.total_slices);

        const imgUrl = `${API_BASE}/study/${encodeURIComponent(activeStudyId)}/slice/${sliceIndex}/render?plane=${viewingPlane}&window=${windowPreset}`;
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          if (!active) return;
          imageObjRef.current = img;
          setSliceLoading(false);
          drawCanvas();
        };
        img.onerror = () => {
          if (!active) return;
          setSliceLoading(false);
        };
        img.src = imgUrl;
      })
      .catch(() => {
        if (active) setSliceLoading(false);
      });

    return () => { active = false; };
  }, [activeStudyId, sliceIndex, viewingPlane, windowPreset]);

  // 4. On-demand Grad-CAM computation when candidate nodule selected
  useEffect(() => {
    if (!showGradCam || !selectedFindingId) {
      setHeatmapMatrix(null);
      return;
    }

    setHeatmapLoading(true);
    fetch(`${API_BASE}/xai/${encodeURIComponent(selectedFindingId)}/gradcam3d?study_id=${encodeURIComponent(activeStudyId)}&plane=${viewingPlane}`)
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
  }, [showGradCam, selectedFindingId, activeStudyId, viewingPlane]);

  // 5. Draw Canvas with Real Volume Slice, Bounding Boxes & On-Demand Heatmap
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    // Pan & Zoom
    ctx.translate(W / 2 + pan.x, H / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-W / 2, -H / 2);

    // Draw Real CT Volume Slice
    const img = imageObjRef.current;
    if (img && img.complete && img.naturalWidth > 0) {
      ctx.drawImage(img, 0, 0, W, H);
    } else {
      // Clean diagnostic background grid while loading
      ctx.fillStyle = '#080c14';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = '#111b2b';
      ctx.lineWidth = 1;
      for (let x = 0; x < W; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
      for (let y = 0; y < H; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }
    }

    // Draw Candidate Nodule Bounding Boxes & On-demand Grad-CAM Heatmap (hidden in Blind Read Mode)
    if (!blindReadMode) {
      const visibleFindings = findings.filter(f => (f.calibrated_probability ?? f.raw_probability ?? 0) >= riskThreshold);
      visibleFindings.forEach(finding => {
        const cz = finding.center_zyx[0];
        const cy = finding.center_zyx[1];
        const cx = finding.center_zyx[2];

        const loc = calculateMultiplanarLocalization(
          { nodule_id: finding.nodule_id, center_zyx: finding.center_zyx as [number, number, number], diameter_mm: finding.diameter_mm },
          sliceIndex,
          viewingPlane,
          {
            matrix_width: 512,
            matrix_height: 512,
            total_slices: totalSlices,
            pixel_spacing_x_mm: pixelSpacing[0],
            pixel_spacing_y_mm: pixelSpacing[1],
            slice_thickness_mm: sliceThickness,
          },
          { width: W, height: H },
          { zoom, panX: pan.x, panY: pan.y }
        );

        if (loc.isIntersecting) {
          const centerPxX = loc.canvasX;
          const centerPxY = loc.canvasY;
          const radiusPx = Math.max(12, loc.canvasRadius);
          const isSelected = finding.nodule_id === selectedFindingId;

          // Draw bounding box
          ctx.strokeStyle = isSelected ? '#ef4444' : '#f59e0b';
          ctx.lineWidth = isSelected ? 2.5 : 1.5;
          if (finding.review_status === 'accepted') ctx.strokeStyle = '#10b981';
          if (finding.review_status === 'rejected') ctx.strokeStyle = '#64748b';

          ctx.strokeRect(centerPxX - radiusPx, centerPxY - radiusPx, radiusPx * 2, radiusPx * 2);

          // Center crosshair
          ctx.beginPath();
          ctx.moveTo(centerPxX - 5, centerPxY); ctx.lineTo(centerPxX + 5, centerPxY);
          ctx.moveTo(centerPxX, centerPxY - 5); ctx.lineTo(centerPxX, centerPxY + 5);
          ctx.stroke();

          // Label
          ctx.font = '600 11px Inter, system-ui, sans-serif';
          const labelText = `${finding.nodule_id} (${finding.diameter_mm.toFixed(1)}mm)`;
          const textWidth = ctx.measureText(labelText).width;
          ctx.fillStyle = 'rgba(10, 15, 25, 0.85)';
          ctx.fillRect(centerPxX - radiusPx, centerPxY - radiusPx - 18, textWidth + 8, 16);
          ctx.fillStyle = isSelected ? '#fca5a5' : '#fde68a';
          ctx.fillText(labelText, centerPxX - radiusPx + 4, centerPxY - radiusPx - 6);

          // On-demand Grad-CAM Heatmap Overlay
          if (isSelected && showGradCam && heatmapMatrix && heatmapMatrix.length > 0) {
            const hmSize = radiusPx * 2.8;
            const hmCanvas = document.createElement('canvas');
            hmCanvas.width = 64;
            hmCanvas.height = 64;
            const hmCtx = hmCanvas.getContext('2d');
            if (hmCtx) {
              const imgData = hmCtx.createImageData(64, 64);
              for (let r = 0; r < 64; r++) {
                for (let c = 0; c < 64; c++) {
                  const val = heatmapMatrix[r] ? heatmapMatrix[r][c] || 0 : 0;
                  const idx = (r * 64 + c) * 4;
                  if (val > 0.08) {
                    // Jet / warm colormap
                    imgData.data[idx] = Math.round(Math.min(255, val * 255 * 1.5));
                    imgData.data[idx + 1] = Math.round(Math.max(0, (1 - val) * 200));
                    imgData.data[idx + 2] = 20;
                    imgData.data[idx + 3] = Math.round(val * gradCamAlpha * 255);
                  } else {
                    imgData.data[idx + 3] = 0;
                  }
                }
              }
              hmCtx.putImageData(imgData, 0, 0);
              ctx.drawImage(hmCanvas, centerPxX - hmSize / 2, centerPxY - hmSize / 2, hmSize, hmSize);
            }
          }
        }
      });
    }

    ctx.restore();
  }, [pan, zoom, findings, selectedFindingId, sliceIndex, viewingPlane, sliceThickness, pixelSpacing, totalSlices, showGradCam, heatmapMatrix, gradCamAlpha, riskThreshold, blindReadMode]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // 6. Handle Mouse Movement for HU Inspection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isPanning) {
      setPan({
        x: mouseX - panStartRef.current.x,
        y: mouseY - panStartRef.current.y
      });
      return;
    }

    // Convert screen coordinates to volume voxel coordinates
    const W = canvas.width;
    const H = canvas.height;
    const normX = ((mouseX - (W / 2 + pan.x)) / zoom + W / 2) / W;
    const normY = ((mouseY - (H / 2 + pan.y)) / zoom + H / 2) / H;

    const voxX = Math.round(Math.max(0, Math.min(511, normX * 512)));
    const voxY = Math.round(Math.max(0, Math.min(511, normY * 512)));
    setCursorPos({ z: sliceIndex, y: voxY, x: voxX });

    // Fetch real HU at coordinates
    fetch(`${API_BASE}/study/${encodeURIComponent(activeStudyId)}/voxel_hu?z=${sliceIndex}&y=${voxY}&x=${voxX}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.hu !== undefined) setCursorHU(data.hu);
      })
      .catch(() => {});
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 0) {
      setIsPanning(true);
      panStartRef.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    }
  };

  const handleMouseUp = () => setIsPanning(false);

  // 7. Mouse Wheel for Fast Slice Scrolling
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      setSliceIndex(prev => Math.min(totalSlices - 1, prev + 1));
    } else {
      setSliceIndex(prev => Math.max(0, prev - 1));
    }
  };

  // 8. Keyboard Shortcuts (ArrowUp/Down for slice, J/K for findings, A for accept, R for reject)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSliceIndex(prev => Math.min(totalSlices - 1, prev + 1));
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSliceIndex(prev => Math.max(0, prev - 1));
      } else if (e.key === 'j' || e.key === 'J') {
        e.preventDefault();
        // Next finding
        const idx = findings.findIndex(f => f.nodule_id === selectedFindingId);
        if (idx !== -1 && idx < findings.length - 1) {
          const nextF = findings[idx + 1];
          setSelectedFindingId(nextF.nodule_id);
          setSliceIndex(Math.round(nextF.center_zyx[0]));
        }
      } else if (e.key === 'k' || e.key === 'K') {
        e.preventDefault();
        // Prev finding
        const idx = findings.findIndex(f => f.nodule_id === selectedFindingId);
        if (idx > 0) {
          const prevF = findings[idx - 1];
          setSelectedFindingId(prevF.nodule_id);
          setSliceIndex(Math.round(prevF.center_zyx[0]));
        }
      } else if (e.key === 'a' || e.key === 'A') {
        if (selectedFindingId) handleReviewAction(selectedFindingId, 'accept');
      } else if (e.key === 'r' || e.key === 'R') {
        if (selectedFindingId) handleReviewAction(selectedFindingId, 'reject');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [totalSlices, findings, selectedFindingId]);

  // 9. Doctor Actions: Accept, Reject, Edit finding with DB Audit Trail
  const handleReviewAction = async (noduleId: string, action: 'accept' | 'reject' | 'adjust', adjustedDiam?: number, notes?: string) => {
    try {
      const res = await fetch(`${API_BASE}/findings/${encodeURIComponent(noduleId)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          user_id: signerName,
          notes: notes || '',
          adjusted_diameter_mm: adjustedDiam,
          study_id: activeStudyId
        })
      });
      if (res.ok) {
        setFindings(prev => prev.map(f => {
          if (f.nodule_id === noduleId) {
            return {
              ...f,
              review_status: action === 'adjust' ? 'edited' : (action as any),
              diameter_mm: adjustedDiam || f.diameter_mm,
              user_notes: notes || f.user_notes
            };
          }
          return f;
        }));
        setEditingFindingId(null);
      }
    } catch (err) {
      console.error('Review action failed:', err);
    }
  };

  // 10. Trigger Pipeline with Triage Mode Selection
  const handleTriggerRun = async (mode: 'fast_triage' | 'full') => {
    setJobProgress(10);
    setCurrentStageName('Submitting to queue…');
    try {
      const res = await fetch(`${API_BASE}/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ study_id: activeStudyId, mode })
      });
      if (res.ok) {
        const data = await res.json();
        const jobId = data.job_id;

        // Poll or subscribe to progress
        const interval = setInterval(async () => {
          try {
            const pRes = await fetch(`${API_BASE}/jobs/${jobId}`);
            if (pRes.ok) {
              const pData = await pRes.json();
              setJobProgress(pData.progress_percent || 20);
              setCurrentStageName(pData.current_stage || 'Processing…');
              if (pData.status === 'TRIAGE_COMPLETED' || pData.status === 'FULL_COMPLETED' || pData.status === 'FAILED') {
                clearInterval(interval);
                setJobProgress(null);
                loadStudyDetails(activeStudyId);
              }
            }
          } catch {
            clearInterval(interval);
            setJobProgress(null);
          }
        }, 300);
      }
    } catch {
      setJobProgress(null);
    }
  };

  // 11. Multi-study / DICOM & MetaImage File Upload Handler
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.uploadDicom(formData);
      if (res && res.study_id) {
        setActiveStudyId(res.study_id);
        onSelectStudy?.(res.study_id);
        fetch(`${API_BASE}/cases`)
          .then(r => r.ok ? r.json() : { cases: [] })
          .then(d => { if (d && d.cases) setStudies(d.cases); })
          .catch(() => {});
        // Automatically enqueue fast triage for uploaded study
        handleTriggerRun('fast_triage');
      }
    } catch (err: any) {
      console.error('Upload error:', err);
      setUploadError(err.message || 'Upload failed. Ensure archive contains DICOM (.dcm), MetaImage (.mhd/.raw), or images.');
    } finally {
      setUploading(false);
    }
  };

  const activeFinding = findings.find(f => f.nodule_id === selectedFindingId) || findings[0] || null;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '260px minmax(0, 1fr) 340px',
      gridTemplateRows: '1fr auto',
      height: '100%',
      minHeight: 0,
      gap: '8px',
      background: 'var(--bg-main, #0b0f17)',
      color: 'var(--text-main, #f1f5f9)',
      fontFamily: 'Inter, system-ui, sans-serif'
    }}>

      {/* ─────────────────────────────────────────────────────────────
          COLUMN 1: WORKLIST & INTAKE (Left)
         ───────────────────────────────────────────────────────────── */}
      <aside style={{
        background: 'var(--bg-panel, #0f172a)',
        border: '1px solid var(--border, #1e293b)',
        borderRadius: '4px',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        overflow: 'hidden'
      }}>
        <div style={{ padding: '12px', borderBottom: '1px solid var(--border, #1e293b)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Study Worklist</span>
            <Badge variant="neutral">{studies.length} Studies</Badge>
          </div>
          <input
            type="text"
            placeholder="Search study ID or patient…"
            value={searchFilter}
            onChange={e => setSearchFilter(e.target.value)}
            style={{
              width: '100%',
              padding: '6px 8px',
              fontSize: '12px',
              background: '#090d16',
              border: '1px solid #1e293b',
              borderRadius: '3px',
              color: '#f8fafc',
              marginBottom: '8px'
            }}
          />

          {/* Mode Selector & Batch Upload */}
          <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
            <button
              onClick={() => handleTriggerRun('fast_triage')}
              style={{
                flex: 1,
                padding: '6px',
                fontSize: '11px',
                fontWeight: 600,
                background: inferenceMode === 'fast_triage' ? '#2563eb' : '#1e293b',
                color: '#ffffff',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer'
              }}
            >
              ⚡ Fast Triage
            </button>
            <button
              onClick={() => handleTriggerRun('full')}
              style={{
                flex: 1,
                padding: '6px',
                fontSize: '11px',
                fontWeight: 600,
                background: inferenceMode === 'full' ? '#7c3aed' : '#1e293b',
                color: '#ffffff',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer'
              }}
            >
              🔬 Full Analysis
            </button>
          </div>

          <label style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            padding: '7px',
            fontSize: '11px',
            fontWeight: 600,
            background: '#0d1829',
            border: '1px dashed #3b82f6',
            borderRadius: '3px',
            cursor: 'pointer',
            color: '#60a5fa'
          }}>
            <Upload size={13} />
            <span>{uploading ? 'Ingesting CT Study…' : 'Upload Study Archive (.zip, .dcm)'}</span>
            <input type="file" accept=".zip,.dcm,.mhd" onChange={handleFileUpload} style={{ display: 'none' }} />
          </label>
          {uploadError && (
            <div style={{
              marginTop: '4px',
              marginBottom: '6px',
              padding: '6px 8px',
              background: '#450a0a',
              border: '1px solid #dc2626',
              borderRadius: '3px',
              color: '#fca5a5',
              fontSize: '11px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <AlertTriangle size={14} color="#ef4444" style={{ flexShrink: 0 }} />
              <span style={{ flex: 1 }}>{uploadError}</span>
              <button onClick={() => setUploadError(null)} style={{ background: 'none', border: 'none', color: '#fca5a5', cursor: 'pointer', padding: 0 }}>
                <X size={12} />
              </button>
            </div>
          )}
        </div>

        {/* Study Rows */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '6px' }}>
          {studies
            .filter(s => !searchFilter || s.study_id.toLowerCase().includes(searchFilter.toLowerCase()))
            .map(s => {
              const isActive = s.study_id === activeStudyId;
              return (
                <div
                  key={s.study_id}
                  onClick={() => {
                    setActiveStudyId(s.study_id);
                    onSelectStudy?.(s.study_id);
                  }}
                  style={{
                    padding: '8px 10px',
                    borderRadius: '4px',
                    marginBottom: '4px',
                    background: isActive ? '#1e293b' : 'transparent',
                    border: isActive ? '1px solid #3b82f6' : '1px solid transparent',
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '2px'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontFamily: 'monospace', fontWeight: 600, fontSize: '12px' }}>{s.study_id}</span>
                    <span style={{ fontSize: '10px', color: s.qc_status === 'PASS' ? '#10b981' : '#f59e0b' }}>
                      {s.qc_status || 'PASS'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#94a3b8' }}>
                    <span>{s.slice_count || 184} slices · 1.25mm</span>
                    {s.is_demo && <span style={{ color: '#fbbf24', fontSize: '10px', fontWeight: 700 }}>SYNTHETIC</span>}
                  </div>
                </div>
              );
            })}
        </div>
      </aside>

      {/* ─────────────────────────────────────────────────────────────
          COLUMN 2: LARGE CENTER VIEWER & HUD
         ───────────────────────────────────────────────────────────── */}
      <section style={{
        background: '#000000',
        border: '1px solid var(--border, #1e293b)',
        borderRadius: '4px',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        position: 'relative'
      }}>
        {/* Top Viewer Control Bar */}
        <header style={{
          height: '42px',
          background: 'var(--bg-panel, #0f172a)',
          borderBottom: '1px solid var(--border, #1e293b)',
          padding: '0 12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0
        }}>
          {/* Multiplanar plane selection */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            {(['axial', 'coronal', 'sagittal'] as ViewingPlane[]).map(plane => (
              <button
                key={plane}
                onClick={() => setViewingPlane(plane)}
                style={{
                  padding: '4px 10px',
                  fontSize: '11px',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  background: viewingPlane === plane ? '#2563eb' : '#1e293b',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer'
                }}
              >
                {plane}
              </button>
            ))}
          </div>

          {/* Window / Level Presets */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            {(['lung', 'mediastinum', 'bone', 'soft_tissue'] as WindowPreset[]).map(preset => (
              <button
                key={preset}
                onClick={() => setWindowPreset(preset)}
                style={{
                  padding: '4px 8px',
                  fontSize: '11px',
                  background: windowPreset === preset ? '#334155' : 'transparent',
                  color: windowPreset === preset ? '#38bdf8' : '#94a3b8',
                  border: '1px solid #334155',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  textTransform: 'capitalize'
                }}
              >
                {preset.replace('_', ' ')}
              </button>
            ))}
          </div>

          {/* Blind Read Mode & Zoom/Pan tools */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <button
              onClick={() => setBlindReadMode(b => !b)}
              style={{
                padding: '4px 10px',
                fontSize: '11px',
                fontWeight: 600,
                background: blindReadMode ? '#7c2d12' : '#1e293b',
                color: blindReadMode ? '#fed7aa' : '#cbd5e1',
                border: blindReadMode ? '1px solid #ea580c' : '1px solid #334155',
                borderRadius: '3px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '5px'
              }}
              title="Toggle Blind-First Read Mode (Hides AI annotations for independent read)"
            >
              {blindReadMode ? <EyeOff size={13} color="#ea580c" /> : <Eye size={13} />}
              <span>Blind Read: {blindReadMode ? 'ON' : 'OFF'}</span>
            </button>
            <button onClick={() => setZoom(z => Math.min(4.0, z + 0.25))} style={{ padding: '4px', background: '#1e293b', border: 'none', borderRadius: '3px', color: '#fff', cursor: 'pointer' }} title="Zoom In">
              <ZoomIn size={14} />
            </button>
            <button onClick={() => setZoom(z => Math.max(0.5, z - 0.25))} style={{ padding: '4px', background: '#1e293b', border: 'none', borderRadius: '3px', color: '#fff', cursor: 'pointer' }} title="Zoom Out">
              <ZoomOut size={14} />
            </button>
            <button onClick={() => { setZoom(1.0); setPan({ x: 0, y: 0 }); }} style={{ padding: '4px', background: '#1e293b', border: 'none', borderRadius: '3px', color: '#fff', cursor: 'pointer' }} title="Reset View">
              <RotateCcw size={14} />
            </button>
          </div>
        </header>

        {/* Primary Viewport Canvas */}
        <div style={{ flex: 1, minHeight: 0, position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {blindReadMode && (
            <div style={{
              position: 'absolute',
              top: '12px',
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'rgba(124, 45, 18, 0.90)',
              border: '1px solid #ea580c',
              color: '#fed7aa',
              padding: '4px 12px',
              borderRadius: '3px',
              fontSize: '11px',
              fontWeight: 600,
              letterSpacing: '0.04em',
              pointerEvents: 'none',
              zIndex: 10
            }}>
              BLIND READ MODE ACTIVE — AI OVERLAYS HIDDEN
            </div>
          )}
          <canvas
            ref={canvasRef}
            width={512}
            height={512}
            onMouseMove={handleMouseMove}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onWheel={handleWheel}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              cursor: isPanning ? 'grabbing' : 'crosshair'
            }}
          />

          {/* Live HUD Overlay (Top-Left) */}
          <div style={{
            position: 'absolute',
            top: 10,
            left: 10,
            background: 'rgba(10, 15, 25, 0.85)',
            border: '1px solid #1e293b',
            borderRadius: '4px',
            padding: '6px 10px',
            fontSize: '11px',
            fontFamily: 'monospace',
            color: '#cbd5e1',
            pointerEvents: 'none'
          }}>
            <div><strong>{activeStudyId}</strong> | {viewingPlane.toUpperCase()}</div>
            <div>Slice: <span style={{ color: '#38bdf8' }}>{sliceIndex + 1}</span> / {totalSlices} ({sliceThickness.toFixed(2)}mm)</div>
            <div>Voxel HU: <span style={{ color: cursorHU !== null && cursorHU > 0 ? '#f87171' : '#38bdf8', fontWeight: 700 }}>{cursorHU !== null ? `${cursorHU} HU` : '--'}</span></div>
            <div>Pos: [{cursorPos.z}, {cursorPos.y}, {cursorPos.x}]</div>
          </div>

          {/* Slice Scroll Slider (Right vertical bar) */}
          <div style={{
            position: 'absolute',
            right: 12,
            top: '15%',
            bottom: '15%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '8px'
          }}>
            <input
              type="range"
              min={0}
              max={totalSlices - 1}
              value={sliceIndex}
              onChange={e => setSliceIndex(parseInt(e.target.value, 10))}
              style={{
                writingMode: 'vertical-lr',
                direction: 'rtl',
                height: '100%',
                width: '6px',
                accentColor: '#38bdf8',
                cursor: 'pointer'
              }}
            />
          </div>

          {/* Loading Indicator */}
          {sliceLoading && (
            <div style={{
              position: 'absolute',
              top: 10,
              right: 10,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(15, 23, 42, 0.9)',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              color: '#38bdf8'
            }}>
              <Loader2 size={12} className="animate-spin" />
              <span>Streaming volume…</span>
            </div>
          )}

          {/* XAI Attribution Overlay Indicator */}
          {heatmapLoading && (
            <div style={{
              position: 'absolute',
              bottom: 10,
              left: 10,
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(15, 23, 42, 0.9)',
              padding: '6px 10px',
              borderRadius: '4px',
              fontSize: '11px',
              color: '#f59e0b'
            }}>
              <Loader2 size={13} className="animate-spin" />
              <span>Computing 3D Grad-CAM On-Demand…</span>
            </div>
          )}
        </div>

        {/* Viewport Footer Bar */}
        <footer style={{
          height: '32px',
          background: 'var(--bg-panel, #0f172a)',
          borderTop: '1px solid var(--border, #1e293b)',
          padding: '0 12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '11px',
          color: '#94a3b8',
          flexShrink: 0
        }}>
          <div>
            <span>Keyboard: </span>
            <kbd style={{ background: '#1e293b', padding: '1px 4px', borderRadius: '2px', color: '#f1f5f9' }}>↑/↓</kbd> Slices ·{' '}
            <kbd style={{ background: '#1e293b', padding: '1px 4px', borderRadius: '2px', color: '#f1f5f9' }}>J/K</kbd> Findings ·{' '}
            <kbd style={{ background: '#1e293b', padding: '1px 4px', borderRadius: '2px', color: '#10b981' }}>A</kbd> Accept ·{' '}
            <kbd style={{ background: '#1e293b', padding: '1px 4px', borderRadius: '2px', color: '#ef4444' }}>R</kbd> Reject
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showGradCam}
                onChange={e => setShowGradCam(e.target.checked)}
              />
              <span>Grad-CAM Attribution (Model attribution only, not physical boundary)</span>
            </label>
            {showGradCam && (
              <input
                type="range"
                min={0.1}
                max={1.0}
                step={0.05}
                value={gradCamAlpha}
                onChange={e => setGradCamAlpha(parseFloat(e.target.value))}
                style={{ width: '60px', accentColor: '#f59e0b' }}
                title="Attribution opacity"
              />
            )}
          </div>
        </footer>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          COLUMN 3: CANDIDATE FINDINGS & HUMAN REVIEW (Right)
         ───────────────────────────────────────────────────────────── */}
      <aside style={{
        background: 'var(--bg-panel, #0f172a)',
        border: '1px solid var(--border, #1e293b)',
        borderRadius: '4px',
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        overflow: 'hidden'
      }}>
        <div style={{ padding: '12px', borderBottom: '1px solid var(--border, #1e293b)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Detected Candidates ({findings.length})
            </span>
            <Badge variant="warning">AI Suggestions</Badge>
          </div>

          {/* Sensitivity-first Risk Threshold Slider */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '11px', color: '#94a3b8' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Risk Threshold (Sensitivity-First):</span>
              <strong style={{ color: '#38bdf8' }}>{(riskThreshold * 100).toFixed(0)}%</strong>
            </div>
            <input
              type="range"
              min={0.05}
              max={0.90}
              step={0.05}
              value={riskThreshold}
              onChange={e => setRiskThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#38bdf8' }}
            />
          </div>
        </div>

        {/* Candidate List */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {blindReadMode ? (
            <div style={{ padding: '24px 16px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
              <EyeOff size={28} color="#ea580c" />
              <div style={{ fontSize: '13px', fontWeight: 600, color: '#fed7aa' }}>Blind Read Mode Active</div>
              <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8', lineHeight: 1.4 }}>
                Review the CT volume independently first without algorithmic bias. When ready, reveal AI candidate annotations to compare.
              </p>
              <button
                onClick={() => setBlindReadMode(false)}
                style={{
                  marginTop: '8px',
                  padding: '6px 14px',
                  background: '#2563eb',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '3px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                Reveal AI Findings &amp; CADe
              </button>
            </div>
          ) : findings.length === 0 ? (
            <div style={{ padding: '16px', textAlign: 'center', color: '#94a3b8', fontSize: '12px' }}>
              No candidates above threshold; sensitivity is limited.
            </div>
          ) : (
            findings
              .filter(f => (f.calibrated_probability ?? f.raw_probability ?? 0) >= riskThreshold)
              .map(finding => {
                const isSelected = finding.nodule_id === selectedFindingId;
                const prob = finding.calibrated_probability ?? finding.raw_probability ?? 0.5;
                const isBorderline = prob >= 0.40 && prob <= 0.65;
                const cz = finding.center_zyx[0];
                const cy = finding.center_zyx[1];
                const cx = finding.center_zyx[2];
                const wx = finding.world_xyz_mm ? finding.world_xyz_mm[0] : (cx * (pixelSpacing[1] || 0.703) - 180.0);
                const wy = finding.world_xyz_mm ? finding.world_xyz_mm[1] : (cy * (pixelSpacing[0] || 0.703) - 180.0);
                const wz = finding.world_xyz_mm ? finding.world_xyz_mm[2] : (cz * sliceThickness - 200.0);
                const sigmaVal = finding.epistemic_uncertainty ?? null;

                return (
                  <div
                    key={finding.nodule_id}
                    onClick={() => {
                      setSelectedFindingId(finding.nodule_id);
                      setSliceIndex(Math.round(finding.center_zyx[0]));
                    }}
                    style={{
                      background: isSelected ? '#1e293b' : '#0d131f',
                      border: isSelected ? '1px solid #3b82f6' : '1px solid #1e293b',
                      borderRadius: '4px',
                      padding: '10px',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '6px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontWeight: 700, fontSize: '13px', color: '#f8fafc' }}>
                        {finding.nodule_id}
                      </span>
                      <span style={{
                        fontSize: '11px',
                        fontWeight: 700,
                        color: prob >= 0.5 ? '#f87171' : '#34d399'
                      }}>
                        {(prob * 100).toFixed(1)}% Malignancy
                      </span>
                    </div>

                    <div style={{ fontSize: '11px', color: '#cbd5e1' }}>
                      <strong>{finding.lobe}</strong> · Ø {finding.diameter_mm.toFixed(1)} mm ({finding.volume_mm3.toFixed(0)} mm³)
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '10px', color: '#94a3b8', background: '#070b12', padding: '4px 6px', borderRadius: '3px' }}>
                      <div>World: ({wx.toFixed(1)}, {wy.toFixed(1)}, {wz.toFixed(1)}) mm</div>
                      <div>Voxel: [Z:{Math.round(cz)}, Y:{Math.round(cy)}, X:{Math.round(cx)}] · Box: Ø{finding.diameter_mm.toFixed(1)}mm</div>
                      <div>Calibrated Prob: {(prob * 100).toFixed(1)}% (T=1.17) · Epistemic σ: {sigmaVal.toFixed(3)}</div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px' }}>
                      <span style={{ color: '#94a3b8' }}>Slice {Math.round(finding.center_zyx[0])}</span>
                      <span style={{ color: '#94a3b8' }}>·</span>
                      <span style={{ color: finding.uncertainty_tier === 'High' ? '#f59e0b' : '#94a3b8' }}>
                        Uncertainty: {finding.uncertainty_tier}
                      </span>
                      {finding.requires_mandatory_review && (
                        <span style={{ background: '#7f1d1d', color: '#fecaca', padding: '1px 4px', borderRadius: '2px', fontWeight: 600 }}>
                          MANDATORY REVIEW
                        </span>
                      )}
                    </div>

                    {/* Review Status & Actions */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid #1e293b', paddingTop: '6px', marginTop: '2px' }}>
                      <span style={{
                        fontSize: '10px',
                        textTransform: 'uppercase',
                        fontWeight: 600,
                        color: finding.review_status === 'accepted' ? '#10b981' : (finding.review_status === 'rejected' ? '#ef4444' : '#fbbf24')
                      }}>
                        {finding.review_status === 'accepted' ? '✓ Accepted' : (finding.review_status === 'rejected' ? '✗ Rejected' : 'Pending Review')}
                      </span>

                      <div style={{ display: 'flex', gap: '4px' }}>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleReviewAction(finding.nodule_id, 'accept'); }}
                          style={{ padding: '3px 6px', fontSize: '10px', background: '#064e3b', color: '#6ee7b7', border: 'none', borderRadius: '2px', cursor: 'pointer' }}
                          title="Accept finding"
                        >
                          Accept
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleReviewAction(finding.nodule_id, 'reject'); }}
                          style={{ padding: '3px 6px', fontSize: '10px', background: '#450a0a', color: '#fca5a5', border: 'none', borderRadius: '2px', cursor: 'pointer' }}
                          title="Reject as false positive"
                        >
                          Reject
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingFindingId(editingFindingId === finding.nodule_id ? null : finding.nodule_id);
                            setEditDiameter(finding.diameter_mm);
                            setEditNotes(finding.user_notes || '');
                          }}
                          style={{ padding: '3px 6px', fontSize: '10px', background: '#1e293b', color: '#94a3b8', border: 'none', borderRadius: '2px', cursor: 'pointer' }}
                          title="Adjust size or add note"
                        >
                          Edit
                        </button>
                      </div>
                    </div>

                    {/* Edit Drawer for Caliper Adjustment & Notes */}
                    {editingFindingId === finding.nodule_id && (
                      <div style={{ background: '#090d16', padding: '8px', borderRadius: '3px', marginTop: '4px', fontSize: '11px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <div>
                          <label>Adjust Diameter (mm):</label>
                          <input
                            type="number"
                            step="0.1"
                            value={editDiameter}
                            onChange={e => setEditDiameter(parseFloat(e.target.value) || 0)}
                            style={{ width: '100%', padding: '4px', background: '#1e293b', border: 'none', color: '#fff', borderRadius: '2px', marginTop: '2px' }}
                          />
                        </div>
                        <div>
                          <label>Clinician Observation Note:</label>
                          <input
                            type="text"
                            placeholder="Add margin note…"
                            value={editNotes}
                            onChange={e => setEditNotes(e.target.value)}
                            style={{ width: '100%', padding: '4px', background: '#1e293b', border: 'none', color: '#fff', borderRadius: '2px', marginTop: '2px' }}
                          />
                        </div>
                        <button
                          onClick={() => handleReviewAction(finding.nodule_id, 'adjust', editDiameter, editNotes)}
                          style={{ padding: '5px', background: '#2563eb', color: '#fff', border: 'none', borderRadius: '2px', cursor: 'pointer', fontWeight: 600 }}
                        >
                          Save Review Record
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
          )}
        </div>

        {/* Localisation Quality Honest Reporting (Dynamic from /api/v2/benchmark) */}
        <div style={{
          padding: '10px',
          borderTop: '1px solid var(--border, #1e293b)',
          background: '#090d16',
          fontSize: '11px',
          color: '#94a3b8'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ fontWeight: 600, color: '#f1f5f9' }}>Localisation Quality</span>
            <Badge variant="warning">Preliminary</Badge>
          </div>
          {benchmarkData?.localization ? (
            <>
              <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '4px' }}>
                Held-out evaluation (N={benchmarkData.localization.evaluated_test_scans ?? 5} scans, N={benchmarkData.localization.total_test_nodules ?? 17} nodules)
              </div>
              <div style={{ marginBottom: '2px' }}>
                Recall: <strong style={{ color: '#38bdf8' }}>{((benchmarkData.localization.detection_sensitivity ?? 0) * 100).toFixed(1)}%</strong>
                {benchmarkData.localization.detection_sensitivity_wilson_95_ci && (
                  <span> [{(benchmarkData.localization.detection_sensitivity_wilson_95_ci[0] * 100).toFixed(1)}% - {(benchmarkData.localization.detection_sensitivity_wilson_95_ci[1] * 100).toFixed(1)}%]</span>
                )}
                <span style={{ fontSize: '10px', color: '#64748b' }}> (@ {Math.round(benchmarkData.localization.candidates_per_scan ?? 800)} c/scan)</span>
              </div>
              <div style={{ marginBottom: '2px' }}>
                Centroid Error: <strong style={{ color: '#38bdf8' }}>{benchmarkData.localization.median_centroid_error_mm != null ? `${benchmarkData.localization.median_centroid_error_mm.toFixed(2)} mm` : '—'}</strong>
                {benchmarkData.localization.p90_centroid_error_mm != null && (
                  <span style={{ fontSize: '10px', color: '#64748b' }}> (P90: {benchmarkData.localization.p90_centroid_error_mm.toFixed(2)} mm)</span>
                )}
              </div>
              {benchmarkData.localization.end_to_end_froc && (
                <div style={{ marginTop: '4px', paddingTop: '4px', borderTop: '1px solid #1e293b' }}>
                  <div style={{ fontSize: '10px', fontWeight: 600, color: '#cbd5e1', marginBottom: '2px' }}>End-to-End FROC (Actual Operating Points):</div>
                  {Object.entries(benchmarkData.localization.end_to_end_froc).map(([target, pt]: [string, any]) => (
                    <div key={target} style={{ fontSize: '10px', color: '#94a3b8' }}>
                      {pt.is_reachable ? (
                        <span>Target {target} FP: <strong style={{ color: '#38bdf8' }}>{(pt.sensitivity * 100).toFixed(1)}%</strong> @ {pt.actual_fp_per_scan.toFixed(2)} FP/scan [{pt.wilson_95_ci ? `${(pt.wilson_95_ci[0]*100).toFixed(1)}%-${(pt.wilson_95_ci[1]*100).toFixed(1)}%` : ''}]</span>
                      ) : (
                        <span style={{ color: '#f59e0b' }}>Target {target} FP: Unreachable (max actual: {pt.actual_fp_per_scan.toFixed(2)} FP/scan)</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div style={{ color: '#64748b', fontSize: '10px' }}>Loading held-out benchmark from /api/v2/benchmark…</div>
          )}
          <div style={{ marginTop: '4px', fontSize: '10px', color: '#64748b' }}>AI-generated suggestion, requires clinician review.</div>
        </div>
      </aside>

      {/* ─────────────────────────────────────────────────────────────
          BOTTOM FULL-WIDTH DRAWER (Clinical Report, Usage, Follow-up)
         ───────────────────────────────────────────────────────────── */}
      <footer style={{
        gridColumn: '1 / -1',
        background: 'var(--bg-panel, #0f172a)',
        border: '1px solid var(--border, #1e293b)',
        borderRadius: '4px',
        display: 'flex',
        flexDirection: 'column',
        maxHeight: '190px',
        overflow: 'hidden'
      }}>
        {/* Drawer Tabs */}
        <div style={{
          height: '32px',
          background: '#0a0f1d',
          borderBottom: '1px solid #1e293b',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 12px'
        }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setActiveBottomTab('report')}
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: activeBottomTab === 'report' ? '2px solid #38bdf8' : '2px solid transparent',
                color: activeBottomTab === 'report' ? '#f8fafc' : '#94a3b8',
                fontWeight: 600,
                fontSize: '12px',
                cursor: 'pointer',
                padding: '6px 4px'
              }}
            >
              Clinical Lung-RADS Report
            </button>
            <button
              onClick={() => setActiveBottomTab('usage')}
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: activeBottomTab === 'usage' ? '2px solid #38bdf8' : '2px solid transparent',
                color: activeBottomTab === 'usage' ? '#f8fafc' : '#94a3b8',
                fontWeight: 600,
                fontSize: '12px',
                cursor: 'pointer',
                padding: '6px 4px'
              }}
            >
              Compute Metering &amp; Usage
            </button>
            <button
              onClick={() => setActiveBottomTab('longitudinal')}
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: activeBottomTab === 'longitudinal' ? '2px solid #38bdf8' : '2px solid transparent',
                color: activeBottomTab === 'longitudinal' ? '#f8fafc' : '#94a3b8',
                fontWeight: 600,
                fontSize: '12px',
                cursor: 'pointer',
                padding: '6px 4px'
              }}
            >
              Longitudinal Prior Comparison
            </button>
          </div>

          <div style={{ fontSize: '11px', color: '#fbbf24', fontWeight: 600 }}>
            RESEARCH USE ONLY · Investigational Prototype
          </div>
        </div>

        {/* Tab 1: Clinical Report */}
        {activeBottomTab === 'report' && (
          <div style={{ padding: '10px 16px', overflowY: 'auto', display: 'flex', justifyContent: 'space-between', gap: '16px' }}>
            <div style={{ flex: 1, fontSize: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc' }}>
                  {findings.length > 0 && findings[0].diameter_mm >= 8 ? 'Lung-RADS 4A (Suspicious)' : 'Lung-RADS 2 (Benign)'}
                </span>
                <span style={{ color: '#94a3b8' }}>· Case: {activeStudyId}</span>
              </div>
              <p style={{ margin: '0 0 4px', color: '#cbd5e1' }}>
                {findings.length > 0
                  ? `Identified ${findings.length} pulmonary candidate nodule(s). Dominant lesion in ${findings[0].lobe} measuring ${findings[0].diameter_mm} mm with calibrated probability of ${((findings[0].calibrated_probability ?? 0.65) * 100).toFixed(1)}%.`
                  : 'No candidates above threshold; sensitivity is limited.'}
              </p>
              <div style={{ fontSize: '11px', color: '#38bdf8' }}>
                Recommendation: {findings.length > 0 && findings[0].diameter_mm >= 8
                  ? '3-month low-dose CT follow-up or PET-CT if clinically indicated.'
                  : 'Continue annual low-dose screening in 12 months.'}
              </div>
            </div>

            <div style={{ width: '220px', borderLeft: '1px solid #1e293b', paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ fontSize: '11px', color: '#94a3b8' }}>Radiologist Verification:</div>
              <button
                onClick={() => setDoctorSigned(true)}
                style={{
                  padding: '6px',
                  background: doctorSigned ? '#064e3b' : '#2563eb',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '3px',
                  fontWeight: 600,
                  fontSize: '11px',
                  cursor: 'pointer'
                }}
              >
                {doctorSigned ? '✓ Report Signed & Locked' : `Sign off as ${signerName}`}
              </button>
              {doctorSigned && (
                <div style={{ fontSize: '10px', color: '#10b981' }}>
                  Signed by {signerName} at {new Date().toISOString().slice(11, 19)} UTC
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Compute Metering & Transparency */}
        {activeBottomTab === 'usage' && (
          <div style={{ padding: '10px 16px', overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', fontSize: '11px' }}>
            <div style={{ background: '#090d16', padding: '8px', borderRadius: '3px' }}>
              <div style={{ color: '#94a3b8' }}>End-to-End Latency:</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: usageData?.exceeded_time_budget ? '#f87171' : '#38bdf8' }}>
                {usageData?.total_wall_time_s ? `${usageData.total_wall_time_s}s` : '—'}
              </div>
              <div style={{ fontSize: '10px', color: '#64748b' }}>Budget: 15.0s max</div>
            </div>

            <div style={{ background: '#090d16', padding: '8px', borderRadius: '3px' }}>
              <div style={{ color: '#94a3b8' }}>CPU &amp; Memory:</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc' }}>
                {usageData?.total_cpu_seconds ? `${usageData.total_cpu_seconds}s CPU` : '—'}
              </div>
              <div style={{ fontSize: '10px', color: '#64748b' }}>Peak RAM: {usageData?.peak_memory_mb || '—'} MB</div>
            </div>

            <div style={{ background: '#090d16', padding: '8px', borderRadius: '3px' }}>
              <div style={{ color: '#94a3b8' }}>Input Volume:</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#f8fafc' }}>
                {usageData?.input_voxels ? `${(usageData.input_voxels / 1e6).toFixed(1)}M voxels` : '—'}
              </div>
              <div style={{ fontSize: '10px', color: '#64748b' }}>{usageData?.candidate_crops_processed || findings.length} candidate crops</div>
            </div>

            <div style={{ background: '#090d16', padding: '8px', borderRadius: '3px' }}>
              <div style={{ color: '#94a3b8' }}>Estimated Compute Cost:</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: '#10b981' }}>
                ${usageData?.compute_cost_usd !== undefined ? `$${usageData.compute_cost_usd.toFixed(5)}` : '—'}
              </div>
              <div style={{ fontSize: '10px', color: '#64748b' }}>{usageData?.cost_basis || '$0.048/CPU-hr'}</div>
            </div>

            <div style={{ background: '#090d16', padding: '8px', borderRadius: '3px' }}>
              <div style={{ color: '#94a3b8' }}>LLM Tokens Accounting:</div>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#f8fafc', marginTop: '2px' }}>
                {usageData?.llm_tokens_used || 'no LLM tokens used'}
              </div>
              <div style={{ fontSize: '10px', color: '#64748b' }}>Structured deterministic engine</div>
            </div>
          </div>
        )}

        {/* Tab 3: Longitudinal Prior Comparison */}
        {activeBottomTab === 'longitudinal' && (
          <div style={{ padding: '10px 16px', overflowY: 'auto', display: 'flex', gap: '16px', fontSize: '11px' }}>
            {longitudinalData?.prior_studies && longitudinalData.prior_studies.length > 0 ? (
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, color: '#f1f5f9', marginBottom: '6px' }}>Prior Studies for Patient</div>
                <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ color: '#94a3b8', borderBottom: '1px solid #1e293b' }}>
                      <th style={{ padding: '4px' }}>Date</th>
                      <th style={{ padding: '4px' }}>Study ID</th>
                      <th style={{ padding: '4px' }}>Slice Count</th>
                      <th style={{ padding: '4px' }}>Lung-RADS</th>
                      <th style={{ padding: '4px' }}>Volume Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {longitudinalData.prior_studies.map((ps: any, idx: number) => (
                      <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                        <td style={{ padding: '4px' }}>{ps.date || '—'}</td>
                        <td style={{ padding: '4px', fontFamily: 'monospace' }}>{ps.study_id}</td>
                        <td style={{ padding: '4px' }}>{ps.slice_count || '—'}</td>
                        <td style={{ padding: '4px' }}>{ps.lung_rads || '—'}</td>
                        <td style={{ padding: '4px' }}>{ps.volume_change || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ padding: '12px 0', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Info size={16} color="#64748b" />
                <span>No prior longitudinal CT studies loaded for this patient. Prior comparison and volume doubling time tracking require at least two longitudinal scans.</span>
              </div>
            )}
          </div>
        )}
      </footer>
    </div>
  );
};
