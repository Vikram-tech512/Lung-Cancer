import React from 'react';
import { Layers, Brain, Flame, Eye, Sparkles, FileText, CheckCircle2, AlertTriangle, ShieldCheck } from 'lucide-react';

export type XAIMode = 'SEGMENTATION' | 'ATTENTION' | 'GRAD-CAM' | 'SALIENCY';

interface CaseContext {
  id: string;
  name: string;
  verdictClass?: 'MALIGNANT' | 'BENIGN';
  predictedClass?: 'MALIGNANT' | 'BENIGN';
  verdictProb?: number;
  modelScore?: number;
  uncertainty?: string;
  lobe?: string;
  lobeSub?: string;
  anatomicMark?: string;
  area: number;
  compactness: number;
  lobularity?: number;
  cxRel?: number;
  cyRel?: number;
  centroid?: string;
  severity?: string;
  cause?: string;
  pid?: string;
}

interface AIAssistantExplanationProps {
  activeTab: XAIMode;
  onSelectTab?: (tab: XAIMode) => void;
  activeCase?: CaseContext | null;
  currentPixelX?: number | null;
  currentPixelY?: number | null;
}

interface ModalityMetadata {
  title: string;
  badge: string;
  color: string;
  icon: React.FC<any>;
  purpose: string;
  finding: (c: CaseContext, px: number, py: number) => string;
  recommendation: (isMal: boolean) => string;
}

const MODALITY_INFO: Record<XAIMode, ModalityMetadata> = {
  SEGMENTATION: {
    title: 'Volumetric Boundary Delineation',
    badge: '3D V-Net Contour',
    color: 'var(--accent-cyan)',
    icon: Layers,
    purpose: 'Delineates an algorithmically estimated lesion perimeter. Isolates candidate solid regions from aerated parenchyma. Not a validated lesion boundary unless confirmed by a radiologist.',
    finding: (c, px, py) => {
      const isMal = (c.predictedClass || c.verdictClass) === 'MALIGNANT';
      return `Segmentation mask centred near [${px}, ${py}] in the ${c.lobeSub || c.lobe || 'Right Upper Lobe'}. `
        + `Estimated lesion area: ${(c.area || 13125).toLocaleString()} px². `
        + `Margin character: ${(c.lobularity || 0.4) > 0.25 ? 'irregular (spiculated appearance)' : 'smooth and circumscribed'}. `
        + `${isMal ? 'Model output: HIGH SUSPICION.' : 'Model output: LOW SUSPICION.'} `
        + 'This is a model-estimated boundary — not a radiologist-confirmed anatomical measurement.';
    },
    recommendation: (_isMal) =>
      'RESEARCH USE ONLY — all segmentation findings require review by a qualified radiologist. '
      + 'Do not use this boundary for treatment planning or clinical measurement.'
  },
  ATTENTION: {
    title: 'Multi-Head Self-Attention Rollout',
    badge: '3D Swin Transformer',
    color: 'var(--accent-violet)',
    icon: Brain,
    purpose: 'Visualises how the 3D Swin Transformer internally routes information between spatial tokens. Attention weights indicate which regions the model weighted internally — not a direct explanation of why it made its prediction.',
    finding: (c, px, py) => {
      const score = c.modelScore ?? c.verdictProb ?? 50;
      return `Attention accumulation centred near [${px}, ${py}] in the ${c.lobe || 'Right Upper Lobe'}. `
        + `Model prediction score: ${score}%. `
        + 'High attention in a region means the model weighted that region during its internal computation — '
        + 'this does not confirm that the region contains pathology. '
        + 'Window-aware rollout used (Swin3D local-window rollout, not standard ViT CLS-token rollout).';
    },
    recommendation: (_isMal) =>
      'RESEARCH USE ONLY — attention maps show model-internal routing, not a clinical finding. '
      + 'Radiologist review of the underlying CT image is required.'
  },
  'GRAD-CAM': {
    title: 'Gradient-Weighted Class Activation',
    badge: 'Deep Layer Attribution',
    color: 'var(--accent-amber)',
    icon: Flame,
    purpose: 'Produces a coarse spatial attribution map by weighting convolutional feature activations with class-specific gradients. Indicates which coarse region most influenced the model prediction. Not a precise lesion boundary.',
    finding: (c, px, py) => {
      const cls = c.predictedClass || c.verdictClass || 'MALIGNANT';
      return `Grad-CAM attribution peak near [${px}, ${py}] for model-predicted class: ${cls}. `
        + 'The highlighted region represents the area whose convolutional features had the greatest gradient-weighted influence on the prediction score. '
        + 'Grad-CAM resolution is limited by the convolutional feature map size (typically 4×4×4 or 8×8×8 voxels) '
        + 'and is upsampled — it does not delineate precise anatomical boundaries.';
    },
    recommendation: (_isMal) =>
      'RESEARCH USE ONLY — Grad-CAM attribution is a coarse model explanation, not a lesion localisation or finding. '
      + 'Interpretation requires a qualified radiologist.'
  },
  SALIENCY: {
    title: 'Pixel-Level Gradient Saliency',
    badge: 'Input Gradient Map',
    color: 'var(--accent-emerald)',
    icon: Eye,
    purpose: 'Computes the gradient of the model output with respect to the input voxels. Highlights regions where small HU changes would most strongly affect the model score. Sensitive to fine texture differences; may highlight noise.',
    finding: (c, px, py) => {
      const cls = c.predictedClass || c.verdictClass || 'MALIGNANT';
      return `Input gradient saliency concentrated near [${px}, ${py}] in the ${c.lobe || 'Right Upper Lobe'}. `
        + 'High saliency indicates voxels whose HU values most influence the model prediction score for class: '
        + `${cls}. `
        + 'Saliency maps are noisy and should be interpreted as estimates of local input sensitivity, '
        + 'not as definitive evidence.';
    },
    recommendation: (_isMal) =>
      'RESEARCH USE ONLY — saliency maps are input-sensitivity estimates. '
      + 'They do not constitute a clinical finding. All results require radiologist review.'
  }
};

export const AIAssistantExplanation: React.FC<AIAssistantExplanationProps> = ({
  activeTab,
  onSelectTab,
  activeCase,
  currentPixelX,
  currentPixelY
}) => {
  if (!activeCase) {
    return (
      <div className="medical-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', height: '100%', boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={16} color="var(--accent-cyan)" />
          <span className="medical-header-tag">04 · Attribution &amp; Model Output Synthesis</span>
        </div>
        <div style={{ padding: '28px 16px', textAlign: 'center', color: 'var(--text-muted)', background: 'var(--bg-surface)', borderRadius: '6px', border: '1px dashed var(--border-card)', margin: 'auto 0' }}>
          <Sparkles size={24} color="var(--accent-cyan)" style={{ margin: '0 auto 8px auto', display: 'block', opacity: 0.7 }} />
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-main)' }}>Awaiting Scan Intake</div>
          <div style={{ fontSize: '11px', marginTop: '4px', maxWidth: '340px', marginInline: 'auto' }}>
            Submit a thoracic CT scan above to generate the model findings, morphometric assessment, and model-grounded explainability synthesis.
          </div>
        </div>
      </div>
    );
  }

  const current = MODALITY_INFO[activeTab] || MODALITY_INFO.SEGMENTATION;
  const Icon = current.icon;
  const isMal = (activeCase.predictedClass || activeCase.verdictClass) === 'MALIGNANT';
  const themeColor = isMal ? 'var(--accent-rose)' : 'var(--accent-emerald)';

  return (
    <div className="medical-card" style={{
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      gap: '12px',
      height: '100%',
      boxSizing: 'border-box'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={16} color="var(--accent-cyan)" />
          <span className="medical-header-tag">04 · Attribution &amp; Model Output Synthesis</span>
        </div>

        {/* Case ID badge */}
        <span style={{
          fontSize: '11px',
          fontWeight: 700,
          color: 'var(--accent-cyan)',
          fontFamily: 'JetBrains Mono, monospace',
          background: 'var(--bg-surface)',
          padding: '2px 8px',
          borderRadius: '4px',
          border: '1px solid var(--border-subtle)'
        }}>
          CASE {activeCase.pid || '#00242110D'}
        </span>
      </div>

      {/* Section 1: Clinical Case Profile */}
      <div style={{
        background: 'var(--bg-surface)',
        borderRadius: '6px',
        border: '1px solid var(--border-subtle)',
        padding: '12px 14px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '6px', marginBottom: '8px' }}>
          <div>
            <div style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
              Primary Diagnostic Impression
            </div>
            <div style={{ fontSize: '13px', fontWeight: 800, color: themeColor, marginTop: '2px' }}>
              {activeCase.severity || (isMal ? 'Lung-RADS 4B (Suspicious Nodule)' : 'Lung-RADS 2 (Benign Nodule)')}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-body)', marginTop: '2px' }}>
              {activeCase.cause || 'Invasive Bronchogenic Carcinoma'}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span className={`badge ${isMal ? 'badge-danger' : 'badge-success'}`}>
              {isMal ? 'High Risk Malignancy' : 'Low Risk Benign'}
            </span>
          </div>
        </div>

        {/* Morphometric Data Bar */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '6px',
          background: 'var(--bg-card)',
          padding: '8px 10px',
          borderRadius: '4px',
          border: '1px solid var(--border-subtle)',
          fontSize: '11px',
          marginTop: '6px'
        }}>
          <div>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Target Segment</div>
            <div style={{ fontWeight: 700, color: 'var(--text-main)' }}>{activeCase.lobeSub || 'Apical (S1)'}</div>
          </div>
          <div>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Lesion Area</div>
            <div style={{ fontWeight: 700, color: 'var(--text-main)', fontFamily: 'JetBrains Mono, monospace' }}>
              {(activeCase.area || 13125).toLocaleString()} px²
            </div>
          </div>
          <div>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Margin Character</div>
            <div style={{ fontWeight: 700, color: isMal ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
              {(activeCase.lobularity || 0.4) > 0.25 ? 'Spiculated' : 'Smooth'}
            </div>
          </div>
        </div>
      </div>

      {/* Section 2: Active Modality Evidence (Synced with active viewer tab) */}
      <div style={{
        background: 'var(--bg-surface)',
        borderRadius: '6px',
        border: '1px solid var(--border-subtle)',
        padding: '12px 14px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Icon size={14} color={current.color} />
            <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-main)' }}>
              Active AI Layer: {current.title}
            </span>
          </div>
          <span style={{
            fontSize: '9px',
            fontWeight: 700,
            color: current.color,
            background: 'var(--bg-card)',
            padding: '2px 7px',
            borderRadius: '4px',
            border: '1px solid var(--border-subtle)'
          }}>
            {current.badge}
          </span>
        </div>

        {/* Dynamic image finding */}
        <div style={{ fontSize: '11px', color: 'var(--text-body)', lineHeight: '1.45', marginBottom: '8px' }}>
          <strong style={{ color: 'var(--text-main)' }}>Segmental Finding: </strong>
          {current.finding(activeCase, currentPixelX, currentPixelY)}
        </div>

        {/* Actionable recommendation */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '11px',
          background: 'var(--bg-card)',
          padding: '6px 10px',
          borderRadius: '4px',
          border: '1px solid var(--border-subtle)',
          color: 'var(--text-muted)'
        }}>
          {isMal ? (
            <AlertTriangle size={13} color="var(--accent-amber)" style={{ flexShrink: 0 }} />
          ) : (
            <CheckCircle2 size={13} color="var(--accent-emerald)" style={{ flexShrink: 0 }} />
          )}
          <span>
            <strong style={{ color: isMal ? 'var(--accent-amber)' : 'var(--accent-emerald)' }}>Action: </strong>
            {current.recommendation(isMal)}
          </span>
        </div>
      </div>

      {/* Segment Footer Tag */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '10px',
        color: 'var(--text-muted)',
        padding: '2px 4px'
      }}>
        <span>Anatomical Coordinates: <b style={{ color: 'var(--text-main)', fontFamily: 'JetBrains Mono, monospace' }}>[{currentPixelX}, {currentPixelY}]</b></span>
        <span>Lobe: <b style={{ color: themeColor }}>{activeCase.lobe || 'Right Upper Lobe'}</b></span>
      </div>
    </div>
  );
};

export default AIAssistantExplanation;
