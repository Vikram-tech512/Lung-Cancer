import React, { useState, useEffect } from 'react';
import { 
  Info, AlertTriangle, ShieldCheck, CheckCircle2, 
  Layers, Cpu, Activity, BarChart2, Mail, Calendar, HelpCircle
} from 'lucide-react';
import { API_BASE } from '../services/api';

export const AboutModelScreen: React.FC = () => {
  const [metricsData, setMetricsData] = useState<any>(null);
  const [uncertaintyConfig, setUncertaintyConfig] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let active = true;

    // Load real test metrics, uncertainty config and dynamic model info
    Promise.all([
      fetch(`${API_BASE}/model/metrics`).then(r => r.ok ? r.json() : null),
      fetch(`${API_BASE}/config/uncertainty`).then(r => r.ok ? r.json() : null),
      fetch(`${API_BASE}/model/info`).then(r => r.ok ? r.json() : null)
    ])
    .then(([mRes, uRes, iRes]) => {
      if (!active) return;
      if (mRes && mRes.metrics) setMetricsData(mRes.metrics);
      if (uRes) setUncertaintyConfig(uRes);
      if (iRes) setModelInfo(iRes);
      else if (mRes && mRes.model_info) setModelInfo(mRes.model_info);
    })
    .catch(() => {})
    .finally(() => {
      if (active) setLoading(false);
    });

    return () => {
      active = false;
    };
  }, []);

  const discrimination = metricsData?.discrimination || {
    auc: 0.6471,
    auc_95_ci: [0.2632, 0.9412],
    accuracy_raw: 0.15,
    sensitivity_raw: 1.0,
    specificity_raw: 0.0
  };

  const calibration = metricsData?.calibration || {
    optimal_temperature: 3.3978,
    ece_uncalibrated: 0.7771,
    ece_calibrated: 0.5320
  };

  const uncPasses = metricsData?.uncertainty_mc_dropout?.passes || 30;
  const testCandidateCount = metricsData?.test_candidates || 20;
  const testPatientCount = metricsData?.cohort_patients || 10;
  const modelVer = modelInfo?.model_version || metricsData?.model_info?.model_version || 'v1.1.0-exp';
  const modelStage = modelInfo?.stage || modelInfo?.status || 'EXPERIMENTAL';

  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* Header with Experimental Validation Status */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
        borderRadius: '4px', padding: '24px', display: 'flex', justifyContent: 'space-between',
        alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: '4px',
              padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 700,
              background: 'rgba(217, 119, 6, 0.15)', color: 'var(--accent-amber)',
              border: '1px solid rgba(217, 119, 6, 0.3)', letterSpacing: '0.04em'
            }}>
              VALIDATION STATUS: {modelStage}
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Awaiting multi-center clinical cross-validation
            </span>
          </div>

          <h1 style={{ fontSize: '20px', fontWeight: 700, margin: '4px 0 0 0', color: 'var(--text-main)' }}>
            About This Model — Specifications &amp; Transparency
          </h1>
          <p style={{ margin: '6px 0 0 0', fontSize: '13px', color: 'var(--text-body)', lineHeight: 1.5, maxWidth: '680px' }}>
            Full transparency disclosure on model architecture, training data, benchmark performance with 95% confidence intervals, and explicit tested clinical boundaries.
          </p>
        </div>

        <div style={{
          display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px',
          color: 'var(--text-muted)', textAlign: 'right'
        }}>
          <div>Version: <strong style={{ color: 'var(--text-main)', fontFamily: 'ui-monospace, monospace' }}>{modelVer}</strong></div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', justifyContent: 'flex-end' }}>
            <Calendar size={13} /> Last updated: 2026-10-24
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', justifyContent: 'flex-end' }}>
            <Mail size={13} /> Contact: Research Team
          </div>
        </div>
      </div>

      {/* Grid: Architecture & Training Dataset */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        
        {/* Architecture */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
          borderRadius: '4px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '10px'
        }}>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: 0, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={15} color="var(--primary)" /> Model Architecture
          </h2>
          <div style={{ fontSize: '13px', color: 'var(--text-body)', lineHeight: 1.5 }}>
            Multimodal 3D deep learning ensemble combining:
            <ul style={{ margin: '6px 0', paddingLeft: '20px' }}>
              <li><strong>3D ResNet-50</strong> (volumetric residual convolutional backbone)</li>
              <li><strong>3D Swin Transformer</strong> (hierarchical shifted-window self-attention)</li>
              <li><strong>3D DenseNet-121</strong> (dense volumetric feature re-use)</li>
              <li><strong>IBSI PyRadiomics</strong> (107 standard morphological &amp; texture descriptors)</li>
            </ul>
          </div>
        </div>

        {/* Training Data */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
          borderRadius: '4px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '10px'
        }}>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: 0, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Layers size={15} color="var(--primary)" /> Training Data &amp; Cohort
          </h2>
          <div style={{ fontSize: '13px', color: 'var(--text-body)', lineHeight: 1.5 }}>
            <div>Dataset: <strong>LUNA16 (Grand Challenge)</strong> / LIDC-IDRI</div>
            <div style={{ marginTop: '4px' }}>Total Scans: <strong>888 thoracic CT scans</strong> across 10 subsets</div>
            <div style={{ marginTop: '4px' }}>Candidate indexing: 1,186 radiologist consensus nodule annotations (diameter &gt;= 3mm)</div>
            <div style={{ marginTop: '6px', fontSize: '12px', color: 'var(--text-muted)' }}>
              Note: Current runtime checkpoint is trained on a verified smoke-test cohort (subset0) awaiting multi-center cross-validation.
            </div>
          </div>
        </div>

      </div>

      {/* Test Metrics Section with Real Confidence Intervals (Loaded from results/*.json) */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
        borderRadius: '4px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: 0, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <BarChart2 size={15} color="var(--primary)" /> Benchmark Test Performance (Traced from results/exp_01_metrics.json)
          </h2>
          <span style={{
            fontSize: '11px', color: 'var(--text-main)', background: 'var(--bg-deepest)',
            padding: '2px 8px', borderRadius: '3px', border: '1px solid var(--border-subtle)', fontWeight: 600
          }}>
            Test Cohort: N = {testCandidateCount} candidates ({testPatientCount} scans)
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
          
          <div style={{ background: 'var(--bg-deepest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>ROC AUC</span>
              <span style={{ fontSize: '10px', color: 'var(--accent-amber)', fontWeight: 600 }}>N = {testCandidateCount}</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', marginTop: '2px' }}>
              {discrimination.auc.toFixed(3)}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
              95% CI: [{discrimination.auc_95_ci[0].toFixed(3)} - {discrimination.auc_95_ci[1].toFixed(3)}]
            </div>
          </div>

          <div style={{ background: 'var(--bg-deepest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Sensitivity / Specificity</span>
              <span style={{ fontSize: '10px', color: 'var(--accent-amber)', fontWeight: 600 }}>N = {testCandidateCount}</span>
            </div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-main)', marginTop: '4px' }}>
              Sens: {(discrimination.sensitivity_raw * 100).toFixed(0)}% · Spec: {(discrimination.specificity_raw * 100).toFixed(0)}%
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Raw Accuracy: {(discrimination.accuracy_raw * 100).toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'var(--bg-deepest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Calibration Error (ECE)</span>
              <span style={{ fontSize: '10px', color: 'var(--accent-amber)', fontWeight: 600 }}>N = {testCandidateCount}</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', marginTop: '2px' }}>
              {calibration.ece_calibrated.toFixed(3)}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
              Uncalibrated: {calibration.ece_uncalibrated.toFixed(3)} (T = {calibration.optimal_temperature.toFixed(2)})
            </div>
          </div>

          <div style={{ background: 'var(--bg-deepest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Uncertainty Passes</span>
              <span style={{ fontSize: '10px', color: 'var(--accent-amber)', fontWeight: 600 }}>N = {testCandidateCount}</span>
            </div>
            <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', marginTop: '2px' }}>
              T = {uncPasses}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
              MC Dropout passes per candidate
            </div>
          </div>

        </div>
      </div>

      {/* Uncertainty Tier Definitions (Read from config) */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
        borderRadius: '4px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: '8px' }}>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: 0, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Uncertainty Tier Definitions (Configurable Defaults)
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--accent-amber)', fontWeight: 600 }}>
            Unvalidated defaults to tune on validation subset · Not clinically validated
          </span>
        </div>
        
        <div style={{
          fontSize: '12px', color: 'var(--text-muted)', background: 'var(--bg-deepest)',
          padding: '8px 12px', borderRadius: '4px', border: '1px solid var(--border-subtle)'
        }}>
          <strong>Note:</strong> Uncertainty tier cutoffs are defined on the predictive standard deviation (σ) of candidate probability across T=30 MC Dropout passes. They are unvalidated default heuristics to tune on the validation subset, not clinically validated diagnostic boundaries.
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', fontSize: '12px' }}>
          
          <div style={{
            background: 'var(--bg-deepest)', padding: '12px', borderRadius: '4px',
            border: '1px solid rgba(22, 163, 74, 0.3)'
          }}>
            <div style={{ fontWeight: 700, color: 'var(--accent-emerald)' }}>Low Uncertainty (σ &lt; 0.05)</div>
            <div style={{ color: 'var(--text-body)', marginTop: '4px', lineHeight: 1.4 }}>
              High predictive concordance across stochastic forward passes.
            </div>
          </div>

          <div style={{
            background: 'var(--bg-deepest)', padding: '12px', borderRadius: '4px',
            border: '1px solid rgba(217, 119, 6, 0.3)'
          }}>
            <div style={{ fontWeight: 700, color: 'var(--accent-amber)' }}>Moderate Uncertainty (0.05 &le; σ &le; 0.15)</div>
            <div style={{ color: 'var(--text-body)', marginTop: '4px', lineHeight: 1.4 }}>
              Borderline model agreement. Review recommended; manual caliper verification advised.
            </div>
          </div>

          <div style={{
            background: 'var(--bg-deepest)', padding: '12px', borderRadius: '4px',
            border: '1px solid rgba(220, 38, 38, 0.3)'
          }}>
            <div style={{ fontWeight: 700, color: 'var(--accent-red)' }}>High Uncertainty (σ &gt; 0.15)</div>
            <div style={{ color: 'var(--text-body)', marginTop: '4px', lineHeight: 1.4 }}>
              Divergent stochastic outputs across forward passes. Flagged for manual review.
            </div>
          </div>

        </div>
      </div>

      {/* Manual Review Rules & Clinical Limitations */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        
        {/* Flagged for Manual Review */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
          borderRadius: '4px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '10px'
        }}>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: 0, color: 'var(--accent-amber)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <AlertTriangle size={15} /> Flagged for Manual Review Rules
          </h2>
          <div style={{ fontSize: '13px', color: 'var(--text-body)', lineHeight: 1.5 }}>
            The workstation automatically displays: <br />
            <code style={{ color: 'var(--accent-amber)', background: 'var(--bg-deepest)', padding: '2px 6px', borderRadius: '3px' }}>
              Outside validated conditions. Review manually.
            </code>
            <ul style={{ margin: '6px 0', paddingLeft: '20px' }}>
              <li>DICOM QA check warns or fails (slice thickness &gt; 2.5mm, gantry tilt != 0)</li>
              <li>Highest uncertainty tier is High across candidates</li>
              <li>Severe motion ring artifacts or clipped lung anatomy detected</li>
            </ul>
          </div>
        </div>

        {/* Populations Not Tested */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
          borderRadius: '4px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '10px'
        }}>
          <h2 style={{ fontSize: '13px', fontWeight: 700, margin: 0, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ShieldCheck size={15} color="var(--primary)" /> Populations &amp; Conditions Not Tested
          </h2>
          <div style={{ fontSize: '13px', color: 'var(--text-body)', lineHeight: 1.5 }}>
            <ul style={{ margin: '0', paddingLeft: '20px' }}>
              <li>Pediatric patients (&lt; 18 years of age)</li>
              <li>Post-surgical lobectomy or segmentectomy patients</li>
              <li>Active severe diffuse interstitial lung disease or ARDS</li>
              <li>Scans with heavy intravenous contrast bolus timing</li>
            </ul>
          </div>
        </div>

      </div>

      {/* Mandatory Disclaimer Footer */}
      <div style={{
        padding: '12px 16px', borderRadius: '4px',
        background: 'rgba(220, 38, 38, 0.08)', border: '1px solid rgba(220, 38, 38, 0.25)',
        textAlign: 'center', fontSize: '12px', color: 'var(--accent-red)', fontWeight: 600
      }}>
        RESEARCH USE ONLY (RUO) — NOT A MEDICAL DEVICE · Not cleared by FDA or CE for clinical decision-making, patient triage, or direct medical management.
      </div>

    </div>
  );
};
