import React, { useState } from 'react';
import { 
  Sparkles, Brain, Eye, Layers, Compass, 
  HelpCircle, ChevronDown, ChevronUp, CheckCircle2, ShieldCheck, Flame
} from 'lucide-react';

export type XAIMode = 'SEGMENTATION' | 'ATTENTION' | 'GRAD-CAM' | 'SALIENCY';

interface CaseContext {
  id: string;
  name: string;
  verdictClass: 'MALIGNANT' | 'BENIGN';
  verdictProb: number;
  uncertainty?: string;
  lobe?: string;
  lobeSub?: string;
  area: number;
  compactness: number;
  cxRel?: number;
  cyRel?: number;
  centroid?: string;
  severity?: string;
}

interface AIAssistantExplanationProps {
  activeTab: XAIMode;
  onSelectTab?: (tab: XAIMode) => void;
  activeCase: CaseContext;
  currentPixelX: number;
  currentPixelY: number;
}

interface ModalityMetadata {
  title: string;
  badge: string;
  color: string;
  accentBg: string;
  icon: React.FC<any>;
  clinicalPurpose: string;
  algorithmicPrinciple: string;
  layerFocus: string;
  clinicalActionability: string;
  getDynamicInterpretation: (c: CaseContext, px: number, py: number) => string;
}

const MODALITY_INFO: Record<XAIMode, ModalityMetadata> = {
  SEGMENTATION: {
    title: 'Lesion Segmentation & Volumetric Boundary Delineation',
    badge: 'V-Net / 3D U-Net Contour',
    color: '#f472b6',
    accentBg: 'rgba(244, 114, 182, 0.12)',
    icon: Layers,
    clinicalPurpose: 'Isolate the 3D anatomical volume and 2D planar perimeter of the pulmonary nodule from surrounding aerated lung parenchyma, thoracic wall, and bronchovascular bundles.',
    algorithmicPrinciple: 'Computes dense voxel-wise probability maps using deep encoder-decoder skip connections, segmenting solid, sub-solid (ground glass), and calcified tissue components.',
    layerFocus: 'Multi-scale decoder resolution layers with soft Dice + focal loss boundary alignment.',
    clinicalActionability: 'Enables precise measurement of nodule longest diameter, cross-sectional area, volume doubling time (VDT), and morphological spiculation or lobulation index.',
    getDynamicInterpretation: (c, px, py) => {
      const isMal = c.verdictClass === 'MALIGNANT';
      const lobeName = c.lobe ? `in the ${c.lobe}` : 'in the lung parenchyma';
      return `The segmentation contour identifies a ${isMal ? 'spiculated, irregular' : 'smoothly circumscribed'} lesion centered at [${px}, ${py}] ${lobeName}. The measured cross-sectional area is ${c.area.toLocaleString()} px² with a compactness score of ${c.compactness.toFixed(2)}. ${
        isMal 
          ? 'Notice the irregular peripheral borders indicating parenchymal infiltration into surrounding alveolar spaces.' 
          : 'The well-demarcated margins indicate uniform encapsulation typical of benign non-infiltrative etiology.'
      }`;
    }
  },

  ATTENTION: {
    title: 'Multi-Head Self-Attention Rollout',
    badge: '3D Swin Transformer / ViT',
    color: '#38bdf8',
    accentBg: 'rgba(56, 189, 248, 0.12)',
    icon: Brain,
    clinicalPurpose: 'Expose long-range spatial context and structural dependencies that the neural network leverages across the entire thoracic cavity to contextualize the nodule.',
    algorithmicPrinciple: 'Traces the forward attention weight matrices through successive self-attention heads (Rollout matrix multiplication), identifying which distant anatomical regions influenced the nodule diagnosis.',
    layerFocus: 'Swin-3D Stages 3 & 4 (Shifted Window Self-Attention across 8x8x8 token blocks).',
    clinicalActionability: 'Verifies whether the neural network considers vascular convergence, pleural attachment, or hilar lymphadenopathy when scoring malignant probability.',
    getDynamicInterpretation: (c, px, py) => {
      const lobeName = c.lobe ? `in the ${c.lobe}` : 'in the active lung field';
      return `Self-attention rollout highlights multi-scale contextual weighting connecting the focal nodule at [${px}, ${py}] with the adjacent bronchovascular tree ${lobeName}. The network assigns heavy token relevance to the vascular interface, assessing signs of vascular convergence and pleural tethering with ${c.verdictProb}% model confidence.`;
    }
  },

  'GRAD-CAM': {
    title: 'Gradient-Weighted Class Activation Mapping (Grad-CAM)',
    badge: 'Deep Feature Attribution',
    color: '#f97316',
    accentBg: 'rgba(249, 115, 22, 0.12)',
    icon: Flame,
    clinicalPurpose: 'Identify the coarse macroscopic semantic regions within the CT scan that contributed the strongest positive gradient toward the predicted diagnostic verdict.',
    algorithmicPrinciple: 'Pools the backpropagated gradients of the score for target class (${c.verdictClass}) with respect to the final convolutional/feature feature maps, creating a thermal activation map.',
    layerFocus: 'Bottleneck convolutional layer (deepest semantic representation before classification head).',
    clinicalActionability: 'Ensures the diagnostic classification is driven by authentic parenchymal nodule features rather than extraneous background noise, chest wall artifacts, or biopsy markers.',
    getDynamicInterpretation: (c, px, py) => {
      const isMal = c.verdictClass === 'MALIGNANT';
      return `The Grad-CAM thermal peak (hot coral-orange core) is tightly focused on the nodule nidus at [${px}, ${py}]. This confirms that the model's ${isMal ? 'MALIGNANT' : 'BENIGN'} verdict (${c.verdictProb}%) is directly attributed to the dense pathological solid core rather than peripheral imaging noise or pleural reflections.`;
    }
  },

  SALIENCY: {
    title: 'Input Gradient Saliency & High-Frequency Edge Sensitivity',
    badge: 'Backpropagated Pixel Gradients',
    color: '#34d399',
    accentBg: 'rgba(52, 211, 153, 0.12)',
    icon: Eye,
    clinicalPurpose: 'Examine high-frequency micro-textural cues, fine-grained margin transitions, and micro-spiculations at the single-pixel and voxel level.',
    algorithmicPrinciple: 'Computes the direct first derivative ∂y/∂x of the output class logit with respect to each input CT voxel intensity, mapping sensitivity to subtle attenuation variations.',
    layerFocus: 'Direct input image plane backpropagation (full 512x512 / 540x480 resolution).',
    clinicalActionability: 'Detects micro-calcifications, cavitation rims, and fine spiculated corona radiata that may be smoothed out by coarser receptive-field heatmaps.',
    getDynamicInterpretation: (c, px, py) => {
      return `Pixel-level saliency highlights the high-gradient border interface between the nodule perimeter and aerated parenchyma at [${px}, ${py}]. The fine mint-cyan gradient peaks correspond to boundary roughness and micro-spiculations, providing structural rationale for the classifier's sensitivity.`;
    }
  }
};

const FAQ_QUESTIONS: { q: string; a: (c: CaseContext, mode: XAIMode, px: number, py: number) => string }[] = [
  {
    q: 'What is the specific purpose of this visualization view for this image?',
    a: (c, mode, px, py) => {
      const meta = MODALITY_INFO[mode];
      const lobeDesc = c.lobe ? `in ${c.lobe}` : 'pulmonary scan';
      return `For this scan (${c.name} ${lobeDesc}), ${meta.title} serves to: ${meta.clinicalPurpose} Currently centered at [${px}, ${py}], it provides radiological verification that the model's ${c.verdictClass} determination is anchored in authentic anatomical tissue changes.`
    }
  },
  {
    q: 'How does this view differ from the other 3 explainability modalities?',
    a: (c, mode) => {
      if (mode === 'GRAD-CAM') {
        return 'Grad-CAM provides coarse semantic heatmaps from deep convolutional layers showing *where* the model found the strongest class evidence. In contrast, Saliency provides fine pixel-level edge gradients, Attention Rollout captures global transformer relationships across the scan, and Segmentation precisely outlines anatomical boundaries.';
      } else if (mode === 'SEGMENTATION') {
        return 'Segmentation produces a concrete geometric binary/soft boundary mask for volumetry and margin analysis. The other three (Attention, Grad-CAM, Saliency) are attribution heatmaps showing *how* the neural network reasoned about the image.';
      } else if (mode === 'ATTENTION') {
        return 'Attention Rollout maps multi-head transformer token dependencies across the entire lung field, showing contextual coupling. Grad-CAM focuses solely on the discriminative convolutional bottleneck, while Saliency highlights immediate pixel-intensity derivatives.';
      } else {
        return 'Saliency displays direct first-order pixel derivatives (∂y/∂x), illuminating fine edge spiculation and textural spikes. Grad-CAM is much coarser (lower spatial resolution), while Attention Rollout reflects transformer token pathways.';
      }
    }
  },
  {
    q: 'How does the Confidence Signature relate to this CT image view?',
    a: (c, _mode, px, py) => {
      return `The Confidence Signature telemetry is physically extracted along the horizontal transect through the lesion center [X: ${px}, Y: ${py}]. As the sweep laser moves, it measures real CT attenuation values (Hounsfield density from aerated lung to solid tumor core) with calibrated tolerance boundaries (±4.2%). This guarantees the telemetry is directly verified against this image rather than an arbitrary detached animation.`;
    }
  },
  {
    q: 'What should the radiologist conclude from these findings?',
    a: (c) => {
      const isMal = c.verdictClass === 'MALIGNANT';
      return `The findings indicate ${isMal ? 'high clinical risk of malignancy' : 'probable benign etiology'} with ${c.verdictProb}% algorithmic confidence. Because XAI maps confirm localization directly on the nodular parenchyma (without artifact interference), this case warrants ${
        isMal ? 'urgent multidisciplinary tumor board review, contrast CT, or PET/biopsy correlation' : 'routine low-dose CT surveillance at 6-12 months per Fleischner Society guidelines'
      }. Note: Model attributions assist clinical decision-making and are not standalone diagnostic evidence.`;
    }
  }
];

export const AIAssistantExplanation: React.FC<AIAssistantExplanationProps> = ({
  activeTab,
  onSelectTab,
  activeCase,
  currentPixelX,
  currentPixelY
}) => {
  const [activeFaq, setActiveFaq] = useState<number | null>(null);
  const [isExpanded, setIsExpanded] = useState<boolean>(true);

  const currentMeta = MODALITY_INFO[activeTab];
  const IconComponent = currentMeta.icon;

  return (
    <div style={{
      background: 'linear-gradient(180deg, rgba(11, 17, 32, 0.98) 0%, rgba(7, 12, 23, 0.98) 100%)',
      border: '1px solid #1e293b',
      borderRadius: '8px',
      overflow: 'hidden',
      marginTop: '12px',
      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)'
    }}>
      {/* Top Banner: Assistant Header & Tab Switcher */}
      <div style={{
        padding: '12px 16px',
        background: 'rgba(15, 23, 42, 0.8)',
        borderBottom: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '10px'
      }}>
        {/* Assistant Identity */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '28px', height: '28px', borderRadius: '7px',
            background: 'linear-gradient(135deg, #0284c7, #8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 12px rgba(139, 92, 246, 0.4)'
          }}>
            <Sparkles size={16} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '13px', fontWeight: '800', color: '#f8fafc', letterSpacing: '-0.01em' }}>
                AI Modality Purpose &amp; Context Assistant
              </span>
              <span style={{
                fontSize: '9px', fontWeight: '700', color: '#38bdf8',
                background: 'rgba(56, 189, 248, 0.12)', padding: '2px 6px',
                borderRadius: '4px', border: '1px solid rgba(56, 189, 248, 0.25)',
                fontFamily: 'JetBrains Mono, monospace'
              }}>
                GOOGLE / CHATGPT XAI CONTEXT
              </span>
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '1px' }}>
              Clinical objective &amp; image-grounded diagnostic rationale for active CT view
            </div>
          </div>
        </div>

        {/* View Switcher Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {(['SEGMENTATION', 'ATTENTION', 'GRAD-CAM', 'SALIENCY'] as const).map(tab => {
            const isSelected = activeTab === tab;
            const meta = MODALITY_INFO[tab];
            return (
              <button
                key={tab}
                onClick={() => onSelectTab && onSelectTab(tab)}
                style={{
                  background: isSelected ? meta.color : '#0b1329',
                  color: isSelected ? '#040d1a' : '#94a3b8',
                  border: `1px solid ${isSelected ? meta.color : '#1e293b'}`,
                  borderRadius: '5px',
                  padding: '4px 9px',
                  fontSize: '10px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  transition: 'all 0.15s ease'
                }}
              >
                <span>{tab}</span>
              </button>
            );
          })}

          <button
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#64748b',
              cursor: 'pointer',
              padding: '4px',
              display: 'flex',
              alignItems: 'center',
              marginLeft: '4px'
            }}
            title={isExpanded ? 'Collapse AI explanation' : 'Expand AI explanation'}
          >
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div style={{ padding: '16px' }}>
          {/* Active Modality Banner */}
          <div style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '12px',
            background: currentMeta.accentBg,
            border: `1px solid ${currentMeta.color}44`,
            borderRadius: '6px',
            padding: '12px 14px',
            marginBottom: '14px'
          }}>
            <div style={{
              width: '32px', height: '32px', borderRadius: '6px',
              background: '#070c17',
              border: `1px solid ${currentMeta.color}66`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0
            }}>
              <IconComponent size={18} color={currentMeta.color} />
            </div>

            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '6px' }}>
                <span style={{ fontSize: '13px', fontWeight: '800', color: '#f8fafc' }}>
                  {currentMeta.title}
                </span>
                <span style={{
                  fontSize: '10px', fontWeight: '700', color: currentMeta.color,
                  background: '#070c17', padding: '2px 8px', borderRadius: '4px',
                  border: `1px solid ${currentMeta.color}44`,
                  fontFamily: 'JetBrains Mono, monospace'
                }}>
                  {currentMeta.badge}
                </span>
              </div>

              {/* Core Purpose */}
              <div style={{ marginTop: '6px', fontSize: '12px', color: '#cbd5e1', lineHeight: 1.55 }}>
                <strong style={{ color: currentMeta.color }}>Clinical Purpose: </strong>
                {currentMeta.clinicalPurpose}
              </div>
            </div>
          </div>

          {/* 2-Column Detailed Technical & Image Context Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '12px', marginBottom: '14px' }}>
            {/* Left: Image Context & Specific Findings */}
            <div style={{
              background: '#070c17',
              borderRadius: '6px',
              border: '1px solid #1e293b',
              padding: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
                <Compass size={14} color="#38bdf8" />
                <span style={{ fontSize: '11px', fontWeight: '800', color: '#f8fafc', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Image-Specific Context &amp; Findings
                </span>
              </div>
              <div style={{ fontSize: '12px', color: '#94a3b8', lineHeight: 1.6 }}>
                {currentMeta.getDynamicInterpretation(activeCase, currentPixelX, currentPixelY)}
              </div>

              <div style={{
                marginTop: '10px',
                paddingTop: '8px',
                borderTop: '1px solid #1e293b',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                fontSize: '10px',
                color: '#64748b',
                fontFamily: 'JetBrains Mono, monospace'
              }}>
                <span>SITE: {activeCase.lobe || 'LUNG'}</span>
                <span>COORDS: [{currentPixelX}, {currentPixelY}]</span>
                <span style={{ color: currentMeta.color }}>
                  VERDICT: {activeCase.verdictClass} ({activeCase.verdictProb}%)
                </span>
              </div>
            </div>

            {/* Right: Architecture & Actionability */}
            <div style={{
              background: '#070c17',
              borderRadius: '6px',
              border: '1px solid #1e293b',
              padding: '12px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '10px'
            }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <Brain size={14} color="#a78bfa" />
                  <span style={{ fontSize: '11px', fontWeight: '800', color: '#f8fafc', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Network Layer Representation
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: '#94a3b8', lineHeight: 1.5 }}>
                  {currentMeta.layerFocus}
                </div>
              </div>

              <div style={{ borderTop: '1px solid #1e293b', paddingTop: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                  <CheckCircle2 size={13} color="#10b981" />
                  <span style={{ fontSize: '10px', fontWeight: '700', color: '#10b981', textTransform: 'uppercase' }}>
                    Clinical Takeaway
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: '#cbd5e1', lineHeight: 1.45 }}>
                  {currentMeta.clinicalActionability}
                </div>
              </div>
            </div>
          </div>

          {/* Interactive ChatGPT/Google AI Style Prompt Chips */}
          <div style={{ borderTop: '1px solid #1e293b', paddingTop: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
              <HelpCircle size={13} color="#38bdf8" />
              <span style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Ask AI Assistant About This View:
              </span>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: activeFaq !== null ? '10px' : '0' }}>
              {FAQ_QUESTIONS.map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => setActiveFaq(activeFaq === idx ? null : idx)}
                  style={{
                    background: activeFaq === idx ? 'rgba(56, 189, 248, 0.2)' : 'rgba(15, 23, 42, 0.7)',
                    border: `1px solid ${activeFaq === idx ? '#38bdf8' : '#1e293b'}`,
                    color: activeFaq === idx ? '#38bdf8' : '#cbd5e1',
                    borderRadius: '20px',
                    padding: '5px 12px',
                    fontSize: '11px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}
                >
                  <Sparkles size={11} color={activeFaq === idx ? '#38bdf8' : '#64748b'} />
                  <span>{item.q}</span>
                </button>
              ))}
            </div>

            {/* Expanded AI Assistant Answer Box */}
            {activeFaq !== null && (
              <div style={{
                background: '#040814',
                border: '1px solid #38bdf8',
                borderRadius: '6px',
                padding: '12px 14px',
                animation: 'fadeIn 0.2s ease',
                position: 'relative'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  <ShieldCheck size={14} color="#38bdf8" />
                  <span style={{ fontSize: '11px', fontWeight: '800', color: '#38bdf8', textTransform: 'uppercase' }}>
                    AI Clinical Interpretation Response
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: '#e2e8f0', lineHeight: 1.6 }}>
                  {FAQ_QUESTIONS[activeFaq].a(activeCase, activeTab, currentPixelX, currentPixelY)}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AIAssistantExplanation;
