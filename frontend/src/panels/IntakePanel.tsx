import React, { useState, useRef } from 'react';
import { Upload, CheckCircle2, AlertTriangle, XCircle, Shield, FileText, ArrowRight, Loader2, HardDrive, RefreshCw } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Badge } from '../components/ui/Badge';
import { Panel } from '../components/ui/Panel';

export interface IntakePanelProps {
  onStudyLoaded: (studyId: string) => void;
}

interface QCResult {
  status: string;
  study_id: string;
  slice_count: number;
  slice_thickness?: number;
  pixel_spacing?: number[];
  qc_status: 'PASS' | 'WARNING' | 'REJECT';
  qc_findings: string[];
  compatibility?: {
    vendor: string;
    model_name: string;
    kernel: string;
    raw_slice_count: number;
    scout_localizers_filtered: number;
    gantry_detector_tilt_detected: boolean;
  };
  deidentification?: {
    stage_status: string;
    direct_identifiers_removed: boolean;
    private_vendor_tags_removed: boolean;
    timestamp: string;
  };
}

export const IntakePanel: React.FC<IntakePanelProps> = ({ onStudyLoaded }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<QCResult | null>(null);
  const [dragOver, setDragOver] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setErrorMsg(null);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setErrorMsg(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setErrorMsg(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/dicom/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `Upload failed with HTTP ${response.status}`);
      }

      const data: QCResult = await response.json();
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'An error occurred during DICOM upload.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            DICOM Intake &amp; Quality Control Gate
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Automated 10-point DICOM QA, Safe Harbor de-identification, and multi-vendor scanner compatibility inspection.
          </p>
        </div>
        <Badge variant="neutral">Research Use Only</Badge>
      </header>

      {/* Upload Dropzone */}
      <Panel>
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleFileDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
              borderRadius: 'var(--radius)',
              background: dragOver ? 'var(--bg-hover)' : 'var(--bg-panel)',
              padding: '36px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '10px',
              transition: 'border-color 0.2s',
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".zip,.dcm,.tar.gz"
              style={{ display: 'none' }}
            />
            <Upload size={32} color="var(--accent)" />
            <div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)' }}>
                {file ? file.name : 'Select or drag & drop thoracic CT DICOM archive (.zip) or slice (.dcm)'}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                {file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · Click to choose different file` : 'Supports multi-slice DICOM series, volumetric archives, and LUNA16/LIDC archives'}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              DICOM header de-identification is applied before any volumetric cache is generated.
            </div>
            <button
              onClick={handleUpload}
              disabled={!file || isUploading}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 18px',
                borderRadius: 'var(--radius)',
                border: 'none',
                background: !file || isUploading ? 'var(--border)' : 'var(--accent)',
                color: '#ffffff',
                fontWeight: 600,
                fontSize: '13px',
                cursor: !file || isUploading ? 'not-allowed' : 'pointer',
              }}
            >
              {isUploading ? <Loader2 size={16} className="animate-spin" /> : <Shield size={16} />}
              {isUploading ? 'Validating DICOM Series…' : 'Run Quality Gate & Ingest'}
            </button>
          </div>

          {errorMsg && (
            <div style={{
              padding: '12px 14px',
              borderRadius: 'var(--radius)',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: 'var(--status-danger)',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <XCircle size={16} />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>
      </Panel>

      {/* QC Results & Scanner Compatibility Display */}
      {result && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '16px' }}>
          {/* Column 1: 10-Point QC Validation */}
          <Panel>
            <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: '14px' }}>10-Point Quality Gate</span>
                <Badge variant={result.qc_status === 'PASS' ? 'success' : result.qc_status === 'WARNING' ? 'warning' : 'danger'}>
                  {result.qc_status}
                </Badge>
              </div>

              <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div><strong>Assigned Study ID:</strong> <span style={{ fontFamily: 'var(--font-mono)' }}>{result.study_id}</span></div>
                <div><strong>Usable Slices:</strong> {result.slice_count} contiguous axial images</div>
                <div><strong>Slice Thickness:</strong> {result.slice_thickness ? `${result.slice_thickness} mm` : 'Not available'}</div>
                <div><strong>Pixel Spacing:</strong> {result.pixel_spacing ? `${result.pixel_spacing[0]} × ${result.pixel_spacing[1]} mm` : 'Not available'}</div>
              </div>

              <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px' }}>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 600 }}>
                  Quality Checklist Findings:
                </div>
                <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {result.qc_findings.map((f, idx) => (
                    <li key={idx} style={{ color: f.toLowerCase().includes('fail') ? 'var(--status-danger)' : 'var(--text-main)' }}>
                      {f}
                    </li>
                  ))}
                </ul>
              </div>

              <div style={{ marginTop: '8px' }}>
                <button
                  onClick={() => onStudyLoaded(result.study_id)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    padding: '8px 14px',
                    borderRadius: 'var(--radius)',
                    border: 'none',
                    background: 'var(--accent)',
                    color: '#ffffff',
                    fontWeight: 600,
                    fontSize: '13px',
                    cursor: 'pointer'
                  }}
                >
                  <span>Open in Diagnostic Workstation</span>
                  <ArrowRight size={14} />
                </button>
              </div>
            </div>
          </Panel>

          {/* Column 2: Multi-Vendor Scanner Adaptation */}
          {result.compatibility && (
            <Panel>
              <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600, fontSize: '14px' }}>Scanner Hardware &amp; Acquisition</span>
                  <Badge variant="neutral">{result.compatibility.vendor}</Badge>
                </div>

                <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div><strong>Scanner Model:</strong> {result.compatibility.model_name || 'Standard CT Scanner'}</div>
                  <div><strong>Reconstruction Kernel:</strong> {result.compatibility.kernel || 'Standard Thorax'}</div>
                  <div><strong>Raw Archive Slices:</strong> {result.compatibility.raw_slice_count}</div>
                  <div><strong>Scout Localizers Filtered:</strong> {result.compatibility.scout_localizers_filtered}</div>
                  <div><strong>Gantry Detector Tilt:</strong> {result.compatibility.gantry_detector_tilt_detected ? 'Tilted (Corrected)' : '0° (Axial Standard)'}</div>
                </div>

                {result.deidentification && (
                  <div style={{ borderTop: '1px solid var(--border)', paddingTop: '10px', marginTop: '6px' }}>
                    <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '4px', fontWeight: 600 }}>
                      Safe Harbor De-identification
                    </div>
                    <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <CheckCircle2 size={13} color="var(--status-success)" />
                        <span>Direct patient identifiers stripped (DICOM header de-identification)</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <CheckCircle2 size={13} color="var(--status-success)" />
                        <span>Private vendor tags removed</span>
                      </div>
                      <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '4px' }}>
                        Verified at: {result.deidentification.timestamp}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </Panel>
          )}
        </div>
      )}
    </section>
  );
};