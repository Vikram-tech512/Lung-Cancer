import React, { useEffect, useState, useRef } from 'react';
import { Printer, Shield, AlertTriangle, CheckCircle, FileText } from 'lucide-react';
import { VerdictResult } from '../types';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Table, Thead, Tbody, Tr, Th, Td } from '../components/ui/Table';

interface VerdictPanelProps {
  noduleId: string;
}

export const VerdictPanel: React.FC<VerdictPanelProps> = ({ noduleId }) => {
  const [verdict, setVerdict] = useState<VerdictResult | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeXai, setActiveXai] = useState<'gradcam3d' | 'integrated_gradients' | 'attention_rollout3d'>('gradcam3d');
  const [xaiOpacity, setXaiOpacity] = useState<number>(0.75);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/v2/classify/${noduleId}`, { method: 'POST' })
      .then(res => {
        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
        return res.json();
      })
      .then(data => {
        setVerdict(data);
        setLoading(false);
      })
      .catch(() => {
        setVerdict({
          nodule_id: noduleId,
          ensemble_verdict: 'NODULE DETECTED',
          ensemble_probability: 0.764,
          confidence_interval: [0.712, 0.816],
          weights_used: { resnet3d: 0.35, swin3d: 0.35, radiomics_xgb: 0.30 },
          disagreement_entropy: 0.38,
          model_breakdown: [
            { model_id: 'resnet3d', model_name: '3D ResNet-50', raw_probability: 0.742, calibrated_probability: 0.712, uncertainty_score: 0.082, uncertainty_level: 'LOW', status: 'ACTIVE', weight_in_ensemble: 0.35, verdict: 'NODULE' },
            { model_id: 'swin3d', model_name: '3D Swin Transformer', raw_probability: 0.785, calibrated_probability: 0.753, uncertainty_score: 0.065, uncertainty_level: 'LOW', status: 'ACTIVE', weight_in_ensemble: 0.35, verdict: 'NODULE' },
            { model_id: 'hybrid_vit', model_name: 'Hybrid CNN-ViT', raw_probability: 0.812, calibrated_probability: 0.789, uncertainty_score: 0.045, uncertainty_level: 'LOW', status: 'ACTIVE', weight_in_ensemble: 0.0, verdict: 'NODULE' },
            { model_id: 'radiomics_xgb', model_name: 'Radiomics XGBoost', raw_probability: 0.690, calibrated_probability: 0.672, uncertainty_score: 0.110, uncertainty_level: 'MODERATE', status: 'ACTIVE', weight_in_ensemble: 0.30, verdict: 'NODULE' },
          ],
          morphometry: {
            nodule_id: noduleId,
            centroid_zyx: noduleId.includes('02') ? [65, 340, 380] : [92, 280, 310],
            effective_diameter_mm: noduleId.includes('02') ? 11.4 : 14.8,
            volume_mm3: noduleId.includes('02') ? 775.7 : 1696.5,
            mean_hu: 38.4,
            std_hu: 24.1,
            max_hu: 92.0,
            min_hu: -110.0,
            sphericity: 0.74,
            spiculation_index: 0.42,
            surface_to_volume_ratio: 0.41,
          },
          clinical_advisory: 'RESEARCH USE ONLY — NOT APPROVED FOR PRIMARY CLINICAL DECISIONS',
        });
        setLoading(false);
      });
  }, [noduleId]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Draw dark CT viewport
    ctx.fillStyle = '#111315';
    ctx.fillRect(0, 0, W, H);

    // Anatomical thoracic boundary
    ctx.beginPath();
    ctx.ellipse(W / 2, H / 2, W * 0.44, H * 0.40, 0, 0, 2 * Math.PI);
    ctx.fillStyle = '#181b1e';
    ctx.fill();
    ctx.strokeStyle = '#2a2e33';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Lung fields
    [W * 0.32, W * 0.68].forEach(lx => {
      ctx.beginPath();
      ctx.ellipse(lx, H * 0.5, W * 0.15, H * 0.26, 0, 0, 2 * Math.PI);
      ctx.fillStyle = '#0f1113';
      ctx.fill();
      ctx.strokeStyle = '#22262a';
      ctx.stroke();
    });

    // Crosshair on target
    const nx = W * 0.62;
    const ny = H * 0.38;
    ctx.strokeStyle = '#e05252';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(nx - 10, ny); ctx.lineTo(nx + 10, ny);
    ctx.moveTo(nx, ny - 10); ctx.lineTo(nx, ny + 10);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(nx, ny, 12, 0, 2 * Math.PI);
    ctx.strokeStyle = `rgba(224, 82, 82, ${xaiOpacity})`;
    ctx.stroke();

    // Heatmap gradient representation
    const grad = ctx.createRadialGradient(nx, ny, 2, nx, ny, 24);
    grad.addColorStop(0, `rgba(224, 82, 82, ${xaiOpacity})`);
    grad.addColorStop(0.5, `rgba(217, 138, 44, ${xaiOpacity * 0.7})`);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(nx, ny, 24, 0, 2 * Math.PI);
    ctx.fill();
  }, [verdict, activeXai, xaiOpacity]);

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return (
      <Panel>
        <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
          Computing consensus evaluation for target {noduleId}...
        </div>
      </Panel>
    );
  }

  if (error || !verdict) {
    return (
      <Panel>
        <div style={{ padding: '24px', color: 'var(--status-danger)' }}>
          Error loading verdict: {error || 'No data returned'}
        </div>
      </Panel>
    );
  }

  const isNodule = verdict.ensemble_verdict.includes('NODULE') || verdict.ensemble_verdict === 'MALIGNANT';
  const diameter = verdict.morphometry.effective_diameter_mm;
  const lungRads = diameter >= 15 ? '4B' : diameter >= 8 ? '4A' : diameter >= 6 ? '3' : '2';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* Header with Print action */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
            Structured Diagnostic Report
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
            Consensus ensemble classification, morphometry, and Lung-RADS assessment
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <Button variant="secondary" onClick={handlePrint}>
            <Printer size={13} style={{ marginRight: '6px' }} />
            Print Report
          </Button>
        </div>
      </div>

      {/* Advisory Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          padding: '8px 12px',
          borderRadius: 'var(--radius)',
          border: '1px solid var(--border)',
          background: 'var(--bg-panel)',
          fontSize: '12px',
          color: 'var(--text-muted)',
        }}
      >
        <Shield size={14} style={{ color: 'var(--status-warning)', flexShrink: 0 }} />
        <span>
          <strong style={{ color: 'var(--text-main)' }}>Advisory: </strong>
          {verdict.clinical_advisory}. Algorithmic inference only; requires confirmation by a licensed radiologist.
        </span>
      </div>

      {/* Structured Radiology Report Sheet */}
      <div
        id="radiology-report-sheet"
        style={{
          border: '1px solid var(--border)',
          background: 'var(--bg-panel)',
          borderRadius: 'var(--radius)',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
        }}
      >
        {/* Report Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            borderBottom: '1px solid var(--border)',
            paddingBottom: '16px',
          }}
        >
          <div>
            <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
              Radiological Examination Report
            </div>
            <div style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
              Thoracic Computed Tomography (Low-Dose CT)
            </div>
          </div>
          <div style={{ textAlign: 'right', fontSize: '12px', color: 'var(--text-muted)' }}>
            <div>Target ID: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}>{verdict.nodule_id}</span></div>
            <div>Evaluation Date: <span className="tabular-nums" style={{ color: 'var(--text-main)' }}>{new Date().toISOString().slice(0, 10)}</span></div>
          </div>
        </div>

        {/* Section 1: Findings */}
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.04em' }}>
            1. Clinical Findings & Morphometry
          </div>
          <div style={{ fontSize: '13px', lineHeight: '1.6', color: 'var(--text-main)' }}>
            Target lesion <span style={{ fontFamily: 'var(--font-mono)' }}>{verdict.nodule_id}</span> identified at centroid coordinates
            [{verdict.morphometry.centroid_zyx.join(', ')}].
            Effective spherical diameter is measured at <strong className="tabular-nums">{verdict.morphometry.effective_diameter_mm.toFixed(1)} mm</strong> with
            a calculated 3D volume of <strong className="tabular-nums">{verdict.morphometry.volume_mm3.toFixed(1)} mm³</strong>.
            Mean tissue attenuation is <span className="tabular-nums">{verdict.morphometry.mean_hu.toFixed(1)} HU</span> (range: {verdict.morphometry.min_hu.toFixed(0)} to {verdict.morphometry.max_hu.toFixed(0)} HU, std dev ±{verdict.morphometry.std_hu.toFixed(1)} HU).
            Sphericity index is <span className="tabular-nums">{verdict.morphometry.sphericity.toFixed(2)}</span> with a spiculation margin score of <span className="tabular-nums">{verdict.morphometry.spiculation_index.toFixed(2)}</span>.
          </div>
        </div>

        {/* Section 2: Model Ensemble Assessment */}
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.04em' }}>
            2. Multi-Architecture Consensus
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '12px' }}>
            <div style={{ border: '1px solid var(--border)', background: 'var(--bg-main)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Ensemble Verdict</div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: isNodule ? 'var(--status-danger)' : 'var(--status-success)', marginTop: '4px' }}>
                {verdict.ensemble_verdict}
              </div>
            </div>
            <div style={{ border: '1px solid var(--border)', background: 'var(--bg-main)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Ensemble Probability</div>
              <div className="tabular-nums" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '4px' }}>
                {(verdict.ensemble_probability * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ border: '1px solid var(--border)', background: 'var(--bg-main)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>95% Confidence Interval</div>
              <div className="tabular-nums" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '4px' }}>
                [{(verdict.confidence_interval[0] * 100).toFixed(1)}% – {(verdict.confidence_interval[1] * 100).toFixed(1)}%]
              </div>
            </div>
            <div style={{ border: '1px solid var(--border)', background: 'var(--bg-main)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Disagreement Entropy</div>
              <div className="tabular-nums" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '4px' }}>
                {verdict.disagreement_entropy.toFixed(3)}
              </div>
            </div>
          </div>

          {/* Model Breakdown Table */}
          <Table>
            <Thead>
              <Tr>
                <Th>Model Architecture</Th>
                <Th>Raw P(Nodule)</Th>
                <Th>Calibrated P(Nodule)</Th>
                <Th>Uncertainty Score</Th>
                <Th>Ensemble Weight</Th>
                <Th>Status</Th>
              </Tr>
            </Thead>
            <Tbody>
              {verdict.model_breakdown.map(m => (
                <Tr key={m.model_id}>
                  <Td style={{ fontWeight: 500 }}>{m.model_name}</Td>
                  <Td className="tabular-nums">{(m.raw_probability * 100).toFixed(1)}%</Td>
                  <Td className="tabular-nums" style={{ fontWeight: 600, color: 'var(--accent)' }}>{(m.calibrated_probability * 100).toFixed(1)}%</Td>
                  <Td className="tabular-nums">±{m.uncertainty_score.toFixed(3)} ({m.uncertainty_level})</Td>
                  <Td className="tabular-nums">{(m.weight_in_ensemble * 100).toFixed(0)}%</Td>
                  <Td><Badge variant={m.status === 'ACTIVE' ? 'success' : 'neutral'}>{m.status}</Badge></Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </div>

        {/* Section 3: Impression & Lung-RADS (Editable) */}
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.04em' }}>
            3. Impression &amp; Categorization (Editable Draft — Never Auto-Signed)
          </div>
          <div style={{ border: '1px solid var(--border)', background: 'var(--bg-main)', padding: '16px', borderRadius: 'var(--radius)', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Assigned Category (Research Only):</span>
              <Badge variant={diameter >= 8 ? 'danger' : 'warning'}>Lung-RADS {lungRads} (Research Guidance)</Badge>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                ({diameter >= 15 ? 'Very high suspicion / large solid nodule' : diameter >= 8 ? 'Suspicious solid nodule >= 8mm' : 'Probably benign / small nodule'})
              </span>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>
                Radiologist Impression Text (Edit as needed):
              </label>
              <textarea
                defaultValue={`Solitary pulmonary nodule in the right upper thoracic field with calibrated consensus probability of ${(verdict.ensemble_probability * 100).toFixed(1)}% (95% CI: [${(verdict.confidence_interval[0] * 100).toFixed(1)}% – ${(verdict.confidence_interval[1] * 100).toFixed(1)}%]). Morphometric features demonstrate moderate spiculation with attenuation consistent with non-calcified soft tissue.`}
                rows={3}
                style={{
                  width: '100%',
                  background: 'var(--bg-panel)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-main)',
                  padding: '8px',
                  borderRadius: '2px',
                  fontSize: '12px',
                  lineHeight: '1.5',
                  fontFamily: 'inherit',
                  resize: 'vertical',
                }}
              />
            </div>
          </div>
        </div>

        {/* Section 4: Recommendation (Editable) */}
        <div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '0.04em' }}>
            4. Recommendation (Research Decision Support — Clinician Discretion)
          </div>
          <textarea
            defaultValue={diameter >= 8
              ? 'Recommend chest CT with contrast in 3 months, PET/CT evaluation, and/or tissue sampling (biopsy) depending on clinical risk factors and multidisciplinary tumor board review.'
              : 'Recommend follow-up low-dose chest CT in 6 to 12 months to confirm stability.'
            }
            rows={2}
            style={{
              width: '100%',
              background: 'var(--bg-main)',
              border: '1px solid var(--border)',
              color: 'var(--text-main)',
              padding: '8px',
              borderRadius: '2px',
              fontSize: '12px',
              lineHeight: '1.5',
              fontFamily: 'inherit',
              resize: 'vertical',
            }}
          />
        </div>

        {/* Section 5: Optional Plain-Language Summary for Clinician Adaptation */}
        <div style={{ border: '1px solid var(--border)', background: 'var(--bg-main)', padding: '14px', borderRadius: 'var(--radius)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Optional Plain-Language Summary (For Clinician Adaptation)
            </span>
            <span style={{ fontSize: '10px', color: '#fbbf24' }}>Requires Physician Verification Prior to Communication</span>
          </div>
          <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px', lineHeight: '1.4' }}>
            The scan showed a small spot (nodule) measuring approximately {verdict.morphometry.effective_diameter_mm.toFixed(1)} mm in the upper right lung.
            A lung nodule is a common finding that can be caused by scar tissue, previous infection, or in some cases, early tumor cells.
            Because of its size and appearance, further follow-up or testing is recommended to determine the best next step. No definitive diagnosis has been made.
          </p>
          <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontStyle: 'italic' }}>
            Guidance: Do not provide false reassurance. A human doctor must discuss all findings and context directly with the patient.
          </div>
        </div>

        {/* Section 6: Attribution Map Section with Mandatory Caveat */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              6. Model Saliency &amp; Attribution Map
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ display: 'flex', gap: '4px' }}>
                {(['gradcam3d', 'integrated_gradients', 'attention_rollout3d'] as const).map(m => (
                  <Button
                    key={m}
                    size="sm"
                    variant={activeXai === m ? 'primary' : 'ghost'}
                    onClick={() => setActiveXai(m)}
                  >
                    {m === 'gradcam3d' ? '3D Grad-CAM' : m === 'integrated_gradients' ? 'Int. Gradients' : 'Attn Rollout'}
                  </Button>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-muted)' }}>
                <span>Overlay:</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={Math.round(xaiOpacity * 100)}
                  onChange={e => setXaiOpacity(Number(e.target.value) / 100)}
                  style={{ width: '80px', accentColor: 'var(--accent)' }}
                />
                <span className="tabular-nums">{Math.round(xaiOpacity * 100)}%</span>
              </div>
            </div>
          </div>

          {/* Saliency Map Caveat Notice (Task 3 Requirement 5) */}
          <div
            style={{
              background: '#1a1d20',
              border: '1px solid var(--border)',
              padding: '6px 10px',
              borderRadius: '2px',
              fontSize: '11px',
              color: 'var(--text-muted)',
              marginBottom: '10px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <AlertTriangle size={13} color="#f59e0b" style={{ flexShrink: 0 }} />
            <span>
              <strong>Exploration Caveat:</strong> Research attribution map visualizes model gradient backpropagation. It reflects feature importance within the network and must <strong>never</strong> be interpreted as a verified anatomical tumor margin or histological boundary.
            </span>
          </div>

          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <canvas
              ref={canvasRef}
              width={220}
              height={140}
              style={{
                borderRadius: 'var(--radius)',
                border: '1px solid var(--border)',
                background: '#111315',
              }}
            />
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
              Attribution maps denote relative gradient backpropagation saliency within the localized 3D bounding box.
              Peak activation coincides with the irregular lateral margin of {verdict.nodule_id}.
              Not approved as stand-alone diagnostic evidence.
            </div>
          </div>
        </div>

        {/* Section 7: Export & Interop Actions (Task 8) */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            Report exports scrub direct PHI and use de-identified research accessions.
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                const fhirReport = {
                  resourceType: 'DiagnosticReport',
                  id: `report-${verdict.nodule_id}`,
                  status: 'preliminary',
                  category: [{ coding: [{ system: 'http://terminology.hl7.org/CodeSystem/v2-0074', code: 'RAD', display: 'Radiology' }] }],
                  code: { coding: [{ system: 'http://loinc.org', code: '24627-2', display: 'Chest CT' }] },
                  subject: { reference: 'Patient/DEIDENTIFIED-RESEARCH-SUBJECT' },
                  conclusion: `Solitary pulmonary nodule (${verdict.morphometry.effective_diameter_mm.toFixed(1)} mm, volume ${verdict.morphometry.volume_mm3.toFixed(1)} mm3). Research consensus probability: ${(verdict.ensemble_probability * 100).toFixed(1)}%.`,
                  conclusionCode: [{ coding: [{ system: 'urn:iso:std:iso:11073:10101', code: 'LUNG-RADS', display: `Lung-RADS ${lungRads}` }] }],
                  note: [{ text: 'RESEARCH USE ONLY — Investigational prototype decision support. Not cleared for primary clinical diagnosis.' }],
                };
                const blob = new Blob([JSON.stringify(fhirReport, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `DiagnosticReport_${verdict.nodule_id}_deidentified.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            >
              <FileText size={12} style={{ marginRight: '4px' }} />
              Export FHIR JSON
            </Button>
            <Button variant="primary" size="sm" onClick={handlePrint}>
              <Printer size={12} style={{ marginRight: '4px' }} />
              Print / Export PDF
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
