import React, { useState, useEffect } from 'react';
import { BrainCircuit, Play, CheckCircle2, AlertTriangle, Layers, Clock, Cpu, HardDrive, ShieldCheck, ArrowRight, Loader2, Info } from 'lucide-react';
import { API_BASE } from '../services/api';
import { Panel } from '../components/ui/Panel';
import { Badge } from '../components/ui/Badge';

export interface AIAnalysisPanelProps {
  studyId?: string;
  noduleId?: string;
}

interface StageInfo {
  number: number;
  name: string;
  model: string;
  status: 'trained' | 'uninitialized' | 'deterministic';
  checkpoint: string | null;
  description: string;
}

const PIPELINE_STAGES: StageInfo[] = [
  { number: 0, name: 'DICOM QA Gate', model: 'DicomValidator', status: 'deterministic', checkpoint: null, description: '10-point DICOM QA and Safe Harbor de-identification.' },
  { number: 1, name: 'CT Preprocessing', model: 'Volumetric Resampler', status: 'deterministic', checkpoint: null, description: 'Isotropic 1.0 mm spatial normalization and HU windowing.' },
  { number: 2, name: 'Lung Field Segmentation', model: '3D Cavity Extractor', status: 'deterministic', checkpoint: null, description: 'Parenchymal thresholding & morphological lung bounding box.' },
  { number: 3, name: 'Candidate Detection', model: 'Multi-Scale LoG/DoG (Heuristic)', status: 'deterministic', checkpoint: null, description: 'Multi-scale blob detection with vesselness filtering (preliminary heuristic; held-out test metrics loaded from API).' },
  { number: 4, name: '3D Classification', model: 'ResNet3D-18 Half', status: 'trained', checkpoint: 'resnet3d_luna16_best.pt', description: 'Trained checkpoint loaded from LUNA16 candidates (64³ patch).' },
  { number: 5, name: 'IBSI Radiomics', model: 'PyRadiomics Engine', status: 'deterministic', checkpoint: null, description: 'Algorithmic 3D morphological and intensity feature extraction.' },
  { number: 6, name: 'Multimodal Ensemble', model: 'Logistic Fusion', status: 'deterministic', checkpoint: null, description: 'Calibrated weighting of DL representations + Radiomics.' },
  { number: 7, name: 'Explainability (XAI)', model: '3D Grad-CAM', status: 'trained', checkpoint: 'resnet3d_luna16_best.pt', description: 'Computed on-demand on selected 32³ nodule crop.' },
  { number: 8, name: 'Epistemic Uncertainty', model: 'Monte Carlo Dropout', status: 'trained', checkpoint: 'resnet3d_luna16_best.pt', description: '8 stochastic forward passes on borderline candidates.' },
  { number: 9, name: 'Calibration', model: 'Temperature Scaling', status: 'deterministic', checkpoint: null, description: 'Post-hoc empirical calibration (T = 1.18).' },
  { number: 10, name: 'Clinical Report', model: 'Lung-RADS v2022', status: 'deterministic', checkpoint: null, description: 'ACR-standard structured classification rules.' },
];

function computeWilsonCI(k: number, n: number): [number, number] {
  if (n <= 0) return [0, 0];
  const z = 1.95996;
  const z2 = z * z;
  const p = k / n;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const margin = (z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

export const AIAnalysisPanel: React.FC<AIAnalysisPanelProps> = ({ studyId = '' }) => {
  const [benchmarkData, setBenchmarkData] = useState<any>(null);

  useEffect(() => {
    fetch(`${API_BASE}/benchmark`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'available') {
          setBenchmarkData(data);
        }
      })
      .catch(() => {});
  }, []);
  const [inferenceMode, setInferenceMode] = useState<'fast_triage' | 'full'>('fast_triage');
  const [running, setRunning] = useState<boolean>(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState<string>('Ready for execution');
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const cr = benchmarkData?.continuous_reporting;
  const fastMedianS = cr?.fast_triage_mode?.median_wall_time_s;
  const fastBudgetS = cr?.fast_triage_mode?.time_budget_s;
  const fullWallS = cr?.full_mode?.wall_time_s;
  const frocSubset3 = benchmarkData?.metrics?.test_froc_subset3;
  const headline = frocSubset3?.headline_validation_frozen_operating_points?.target_1_0_fp_scan;
  const sweeps = frocSubset3?.exploratory_test_sweeps;
  const totalScans = benchmarkData?.metrics?.test_scans_total ?? benchmarkData?.localization?.evaluated_test_scans;
  const totalNodules = benchmarkData?.metrics?.localization?.total_test_nodules;

  const handleStartInference = async () => {
    if (!studyId || studyId === 'Not available') return;
    setRunning(true);
    setProgress(5);
    setCurrentStage('Submitting job to queue…');
    setErrorMsg(null);
    setExecutionResult(null);

    try {
      const res = await fetch(`${API_BASE}/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ study_id: studyId, mode: inferenceMode }),
      });

      if (!res.ok) throw new Error('Failed to enqueue job');
      const data = await res.json();
      const jId = data.job_id;
      setJobId(jId);

      // Connect to SSE stream
      const evtSource = new EventSource(`${API_BASE}/jobs/${encodeURIComponent(jId)}/stream`);
      evtSource.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          setProgress(update.progress || 0);
          setCurrentStage(update.current_stage || update.status || 'Running');

          if (update.status === 'TRIAGE_COMPLETED' || update.status === 'FULL_COMPLETED') {
            setExecutionResult(update.triage_report || update.report || update);
            setRunning(false);
            evtSource.close();
          } else if (update.status === 'FAILED') {
            setErrorMsg(update.error || 'Pipeline execution failed');
            setRunning(false);
            evtSource.close();
          }
        } catch {
          // Keep waiting
        }
      };

      evtSource.onerror = () => {
        evtSource.close();
        setRunning(false);
      };
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to trigger pipeline inference');
      setRunning(false);
    }
  };

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 700, color: 'var(--text-main)' }}>
            3D Deep Learning Model Inference Engine
          </h2>
          <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            11-stage volumetric pipeline status, checkpoint reality, and asynchronous execution control.
          </p>
        </div>
        <Badge variant="neutral">Research Use Only</Badge>
      </header>

      {/* Execution Controller */}
      <Panel>
        <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <div>
              <span style={{ fontSize: '13px', fontWeight: 600 }}>Target Study: </span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{studyId}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
                  <button
                    onClick={() => setInferenceMode('fast_triage')}
                    style={{
                      padding: '6px 12px',
                      border: 'none',
                      background: inferenceMode === 'fast_triage' ? 'var(--accent)' : 'var(--bg-panel)',
                      color: inferenceMode === 'fast_triage' ? '#ffffff' : 'var(--text-muted)',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Fast Triage Mode {fastMedianS != null ? `(${fastMedianS.toFixed(2)}s warm${fastBudgetS != null ? `, < ${fastBudgetS.toFixed(0)}s budget` : ''})` : ''}
                  </button>
                  <button
                    onClick={() => setInferenceMode('full')}
                    style={{
                      padding: '6px 12px',
                      border: 'none',
                      background: inferenceMode === 'full' ? 'var(--accent)' : 'var(--bg-panel)',
                      color: inferenceMode === 'full' ? '#ffffff' : 'var(--text-muted)',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Full Mode {fullWallS != null ? `(${fullWallS.toFixed(1)}s, Radiomics + MC Dropout)` : '(Radiomics + MC Dropout)'}
                  </button>
                </div>
                {/* Mode Performance FROC Indicators */}
                <div style={{ display: 'flex', gap: '8px', fontSize: '10px' }}>
                  {headline && (
                    <span style={{ color: 'var(--accent)', background: 'rgba(37,99,235,0.08)', padding: '3px 6px', borderRadius: '3px', border: '1px solid rgba(37,99,235,0.2)' }}>
                      <strong>Fast Triage (pool 100/top 20):</strong>{' '}
                      Headline FROC (Validation-Frozen τ={headline.threshold}): {headline.sensitivity_pct}% sens @ {headline.actual_fp_per_scan} FP/scan [{headline.wilson_ci_pct[0]}%–{headline.wilson_ci_pct[1]}%] (N={totalScans ?? '—'} scans, {totalNodules ?? '—'} nodules)
                    </span>
                  )}
                  {sweeps && (
                    <span style={{ color: '#a78bfa', background: 'rgba(124,58,237,0.08)', padding: '3px 6px', borderRadius: '3px', border: '1px solid rgba(124,58,237,0.2)' }}>
                      <strong>Full Mode (pool ≤ 400/all):</strong>{' '}
                      Exploratory sweep: {sweeps.target_2_0_fp_scan ? `${sweeps.target_2_0_fp_scan.sensitivity_pct}% sens @ ${sweeps.target_2_0_fp_scan.actual_fp_per_scan.toFixed(2)} FP/scan [${sweeps.target_2_0_fp_scan.wilson_ci_pct[0]}%–${sweeps.target_2_0_fp_scan.wilson_ci_pct[1]}%], ` : ''}{sweeps.target_4_0_fp_scan ? `${sweeps.target_4_0_fp_scan.sensitivity_pct}% sens @ ${sweeps.target_4_0_fp_scan.actual_fp_per_scan.toFixed(2)} FP/scan [${sweeps.target_4_0_fp_scan.wilson_ci_pct[0]}%–${sweeps.target_4_0_fp_scan.wilson_ci_pct[1]}%]` : ''} (N={totalScans ?? '—'} scans, {totalNodules ?? '—'} nodules; exploratory)
                    </span>
                  )}
                </div>
              </div>

              <button
                onClick={handleStartInference}
                disabled={running || !studyId || studyId === 'Not available'}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 16px',
                  borderRadius: 'var(--radius)',
                  border: 'none',
                  background: running ? 'var(--border)' : 'var(--accent)',
                  color: '#ffffff',
                  fontSize: '12px',
                  fontWeight: 600,
                  cursor: running ? 'not-allowed' : 'pointer',
                }}
              >
                {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                <span>{running ? 'Executing Pipeline…' : 'Run Pipeline Inference'}</span>
              </button>
            </div>
          </div>

          {/* Progress Bar & Status */}
          {running && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
                <span style={{ fontWeight: 600, color: 'var(--accent)' }}>{currentStage}</span>
                <span>{progress}%</span>
              </div>
              <div style={{ height: '6px', background: 'var(--border)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progress}%`, background: 'var(--accent)', transition: 'width 0.3s' }} />
              </div>
            </div>
          )}

          {errorMsg && (
            <div style={{ color: 'var(--status-danger)', fontSize: '12px', padding: '8px 12px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius)' }}>
              {errorMsg}
            </div>
          )}

          {executionResult && (
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--status-success)' }}>
                  ✓ Pipeline Execution Completed ({executionResult.status})
                </span>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Wall Time: {executionResult.usage?.total_wall_time_ms ? `${(executionResult.usage.total_wall_time_ms / 1000).toFixed(2)}s` : 'Not available'} · Cost: ${executionResult.usage?.estimated_cost_usd?.toFixed(6) ?? '0.000000'}
                </span>
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-main)', background: 'var(--bg-hover)', padding: '10px 12px', borderRadius: 'var(--radius)' }}>
                {executionResult.clinical_impression || 'Inference completed. Review findings in Diagnostic Workstation.'}
              </div>
            </div>
          )}
        </div>
      </Panel>

      {/* Model Reality & Stage Inventory */}
      <Panel>
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontWeight: 600, fontSize: '14px' }}>Stage-by-Stage Model Reality &amp; Weight Verification</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Audit basis: checkpoints/ directory inspection</span>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '12px', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>
                  <th style={{ padding: '6px 8px' }}>Stage</th>
                  <th style={{ padding: '6px 8px' }}>Name</th>
                  <th style={{ padding: '6px 8px' }}>Architecture</th>
                  <th style={{ padding: '6px 8px' }}>Weight Status</th>
                  <th style={{ padding: '6px 8px' }}>Active Checkpoint</th>
                  <th style={{ padding: '6px 8px' }}>Details</th>
                </tr>
              </thead>
              <tbody>
                {PIPELINE_STAGES.map((s) => (
                  <tr key={s.number} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '8px', fontFamily: 'var(--font-mono)' }}>Stage {s.number}</td>
                    <td style={{ padding: '8px', fontWeight: 600 }}>{s.name}</td>
                    <td style={{ padding: '8px' }}>{s.model}</td>
                    <td style={{ padding: '8px' }}>
                      {s.status === 'trained' ? (
                        <span style={{ color: 'var(--status-success)', fontWeight: 600 }}>● Trained weights</span>
                      ) : s.status === 'uninitialized' ? (
                        <span style={{ color: 'var(--status-warning)', fontWeight: 600 }}>▲ Model not trained: demo</span>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>○ Algorithmic / Rules</span>
                      )}
                    </td>
                    <td style={{ padding: '8px', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                      {s.checkpoint || 'None required'}
                    </td>
                    <td style={{ padding: '8px', color: 'var(--text-muted)', fontSize: '11px' }}>
                      {(() => {
                        if (s.number === 3 && (benchmarkData?.localization || benchmarkData?.metrics?.localization)) {
                          const locData = benchmarkData.localization || benchmarkData.metrics.localization;
                          const k = locData.recalled_nodules_count;
                          const n = locData.total_test_nodules;
                          const scans = locData.evaluated_test_scans;
                          const candsPerScan = locData.candidates_per_scan != null ? Math.round(locData.candidates_per_scan) : null;
                          const [ciLow, ciHigh] = (k != null && n != null) ? computeWilsonCI(k, n) : [0, 0];
                          const sensPct = locData.detection_sensitivity != null ? (locData.detection_sensitivity * 100).toFixed(1) : (k != null && n ? ((k / n) * 100).toFixed(1) : null);
                          return `Stage 3 candidate detector: ${sensPct ?? '—'}% held-out test recall (95% Wilson CI: ${(ciLow * 100).toFixed(1)}%–${(ciHigh * 100).toFixed(1)}%)${candsPerScan != null ? ` @ ${candsPerScan} cands/scan` : ''} (N=${scans ?? '—'} scans, ${n ?? '—'} nodules; pool 400).`;
                        }
                        return s.description;
                      })()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Panel>
    </section>
  );
};