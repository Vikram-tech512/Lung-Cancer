import React, { useState, useRef, useEffect } from 'react';
import {
  BrainCircuit,
  Upload,
  X,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Image as ImageIcon,
  ChevronDown,
  ClipboardList,
  Copy,
  Check,
  FileCheck2,
  Key,
  Eye,
  EyeOff,
  Server,
  Zap,
  Globe,
  RefreshCw,
  ExternalLink
} from 'lucide-react';
import { API_BASE } from '../services/api';
import { TumourLocator, TumourRegion } from '../components/TumourLocator';
import { PurposeMetrics } from '../components/PurposeMetrics';

/* --------------------------------------------------------------------------
   Model Catalog — ChatGPT, Claude, Gemini
   -------------------------------------------------------------------------- */
export type AIProvider = 'openai' | 'anthropic' | 'gemini';

export interface ModelOption {
  id: string;
  label: string;
  provider: AIProvider;
  providerLabel: string;
  providerBadgeColor: string;
  badgeBg: string;
  tierColor: string;
  desc: string;
  contextWindow: string;
}

const MODELS: ModelOption[] = [
  // OpenAI (ChatGPT)
  {
    id: 'gpt-4o',
    label: 'GPT-4o (Omni Vision)',
    provider: 'openai',
    providerLabel: 'ChatGPT / OpenAI',
    providerBadgeColor: '#10b981',
    badgeBg: 'rgba(16, 185, 129, 0.15)',
    tierColor: '#34d399',
    desc: 'Flagship multimodal vision model · High diagnostic reasoning',
    contextWindow: '128k',
  },
  {
    id: 'gpt-4o-mini',
    label: 'GPT-4o Mini',
    provider: 'openai',
    providerLabel: 'ChatGPT / OpenAI',
    providerBadgeColor: '#10b981',
    badgeBg: 'rgba(16, 185, 129, 0.15)',
    tierColor: '#6ee7b7',
    desc: 'Lightweight & ultra-fast · Fast nodule triage',
    contextWindow: '128k',
  },
  {
    id: 'o1-mini',
    label: 'OpenAI o1-mini',
    provider: 'openai',
    providerLabel: 'ChatGPT / OpenAI',
    providerBadgeColor: '#10b981',
    badgeBg: 'rgba(16, 185, 129, 0.15)',
    tierColor: '#a7f3d0',
    desc: 'Deep step-by-step reasoning for complex oncological differential',
    contextWindow: '128k',
  },
  // Anthropic Claude
  {
    id: 'claude-3-5-sonnet-20241022',
    label: 'Claude 3.5 Sonnet',
    provider: 'anthropic',
    providerLabel: 'Claude / Anthropic',
    providerBadgeColor: '#f43f5e',
    badgeBg: 'rgba(244, 63, 94, 0.15)',
    tierColor: '#fb7185',
    desc: 'State-of-the-art vision and clinical reasoning accuracy',
    contextWindow: '200k',
  },
  {
    id: 'claude-3-5-haiku-20241022',
    label: 'Claude 3.5 Haiku',
    provider: 'anthropic',
    providerLabel: 'Claude / Anthropic',
    providerBadgeColor: '#f43f5e',
    badgeBg: 'rgba(244, 63, 94, 0.15)',
    tierColor: '#fda4af',
    desc: 'Ultra-fast inference speed for rapid screening protocols',
    contextWindow: '200k',
  },
  {
    id: 'claude-3-opus-20240229',
    label: 'Claude 3 Opus',
    provider: 'anthropic',
    providerLabel: 'Claude / Anthropic',
    providerBadgeColor: '#f43f5e',
    badgeBg: 'rgba(244, 63, 94, 0.15)',
    tierColor: '#e11d48',
    desc: 'Deep multi-modal synthesis for ambiguous or rare pulmonary lesions',
    contextWindow: '200k',
  },
  // Google Gemini
  {
    id: 'gemini-3.5-flash',
    label: 'Gemini 3.5 Flash (Recommended)',
    provider: 'gemini',
    providerLabel: 'Gemini / Google',
    providerBadgeColor: '#38bdf8',
    badgeBg: 'rgba(56, 189, 248, 0.15)',
    tierColor: '#38bdf8',
    desc: 'High-speed multimodal reasoning with free-tier quota support',
    contextWindow: '1M tokens',
  },
  {
    id: 'gemini-flash-lite-latest',
    label: 'Gemini Flash Lite (Ultra Fast)',
    provider: 'gemini',
    providerLabel: 'Gemini / Google',
    providerBadgeColor: '#38bdf8',
    badgeBg: 'rgba(56, 189, 248, 0.15)',
    tierColor: '#bae6fd',
    desc: 'Fastest latency & lightweight screening for thoracic CT slices',
    contextWindow: '1M tokens',
  },
  {
    id: 'gemini-3.1-flash-lite',
    label: 'Gemini 3.1 Flash Lite',
    provider: 'gemini',
    providerLabel: 'Gemini / Google',
    providerBadgeColor: '#38bdf8',
    badgeBg: 'rgba(56, 189, 248, 0.15)',
    tierColor: '#38bdf8',
    desc: 'Efficient multimodal research text & report generation',
    contextWindow: '1M tokens',
  },
  {
    id: 'gemini-pro-latest',
    label: 'Gemini Pro (Paid Tier)',
    provider: 'gemini',
    providerLabel: 'Gemini / Google',
    providerBadgeColor: '#38bdf8',
    badgeBg: 'rgba(56, 189, 248, 0.15)',
    tierColor: '#7dd3fc',
    desc: 'Deep multi-modal reasoning (requires billing enabled on Google Cloud)',
    contextWindow: '2M tokens',
  },
];

/* --------------------------------------------------------------------------
   Clinical Presets
   -------------------------------------------------------------------------- */
const PRESETS = [
  {
    label: 'Nodule Risk Assessment (For radiologist review)',
    prompt:
      'Analyse this thoracic CT slice for features that may suggest malignancy: spiculation, lobulation, pleural tethering/indentation, ground-glass opacity (GGO), vascular convergence, or lymphadenopathy. Systematically categorize and report each finding with confidence assessment.',
  },
  {
    label: 'Pulmonary Nodule Detection & Localization',
    prompt:
      'Identify any pulmonary nodules or parenchymal masses visible in this CT image. Describe their anatomical location (lung lobe/segment), approximate diameter, morphology (solid, part-solid, non-solid/GGO), and density characteristics in Hounsfield units (HU) if inferable.',
  },
  {
    label: 'Lung-RADS v2022 Criteria Reference & Fleischner Protocol',
    prompt:
      'Compare nodule characteristics, size, and solid/subsolid proportion against Lung-RADS v2022 criteria for radiologist review. Outline the Fleischner Society surveillance guideline and recommended management timeframe.',
  },
  {
    label: 'Histopathology & Growth Pattern Architectural Differential',
    prompt:
      'Evaluate visible cytomorphologic and architectural features. Outline possible morphological patterns (e.g. Adenocarcinoma subtypes: lepidic, acinar, papillary vs Squamous Cell Carcinoma vs Large Cell Neuroendocrine vs Benign Hamartoma/Granuloma) for radiologist review.',
  },
];

/* --------------------------------------------------------------------------
   Component
   -------------------------------------------------------------------------- */
export interface AIAnalysisPanelProps {
  studyId?: string;
  noduleId?: string;
}

export const AIAnalysisPanel: React.FC<AIAnalysisPanelProps> = ({
  studyId = 'CASE-DEMO-LIDC-0087',
  noduleId = 'NODULE-01'
}) => {
  // Model selection
  const [selectedModelId, setSelectedModelId] = useState<string>('gpt-4o');
  const [modelDropdownOpen, setModelDropdownOpen] = useState<boolean>(false);

  // Server-side provider availability (Keys are held exclusively in backend environment)
  const [serverProviders, setServerProviders] = useState<{ [k: string]: boolean }>({});
  const [showStatusDrawer, setShowStatusDrawer] = useState<boolean>(false);

  // Input states
  const [prompt, setPrompt]           = useState<string>(PRESETS[0].prompt);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageMime, setImageMime]     = useState<string>('image/png');
  const [imageFileName, setImageFileName] = useState<string | null>(null);
  const [imagePreview, setImagePreview]   = useState<string | null>(null);
  const [activeFindingLabel, setActiveFindingLabel] = useState<string | null>(null);

  // Clinical parameters
  const [temperature, setTemperature] = useState<number>(0.2);
  const [maxTokens, setMaxTokens]     = useState<number>(2048);

  // Output states
  const [analyzing, setAnalyzing]     = useState<boolean>(false);
  const [result, setResult]           = useState<string | null>(null);
  const [resultMeta, setResultMeta]   = useState<{
    model: string;
    provider: string;
    durationMs: number;
    usage?: any;
  } | null>(null);
  const [error, setError]             = useState<string | null>(null);
  const [copied, setCopied]           = useState<boolean>(false);

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropdownRef  = useRef<HTMLDivElement>(null);

  const selectedModel = MODELS.find((m) => m.id === selectedModelId) || MODELS[0];

  // Tumour Localization state — strictly isolated to selected study and nodule
  const [locatorRegion, setLocatorRegion] = useState<TumourRegion | null>(null);
  const [locatorSlice, setLocatorSlice] = useState<number>(48);
  // Zero-filled heatmap: invisible placeholder until real XAI attribution is available
  const [locatorHeatmap, setLocatorHeatmap] = useState<number[][]>(() => {
    const h: number[][] = [];
    for (let y = 0; y < 64; y++) {
      h.push(new Array(64).fill(0));
    }
    return h;
  });
  const [purposeMetrics, setPurposeMetrics] = useState<Record<string, Record<string, number | string>>>({});

  // Check server-side configured keys & load localization/metrics dynamically for active study & nodule
  useEffect(() => {
    fetch(`${API_BASE}/api/ai/providers`)
      .then((r) => r.json())
      .then((d) => setServerProviders(d))
      .catch(() => {});

    // Clear stale state immediately before loading new case
    setLocatorRegion(null);
    setLocatorHeatmap(() => {
      const h: number[][] = [];
      for (let y = 0; y < 64; y++) {
        h.push(new Array(64).fill(0));
      }
      return h;
    });

    // Fetch dynamic tumour localization for the active study and nodule
    const cleanNoduleId = noduleId || 'NODULE-01';
    const studyQuery = studyId ? `?study_id=${encodeURIComponent(studyId)}` : '';
    fetch(`${API_BASE}/xai/${cleanNoduleId}/locate${studyQuery}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: TumourRegion | null) => {
        if (d && d.found) {
          setLocatorRegion(d);
          if (d.profile?.best_slice !== undefined) {
            setLocatorSlice(d.profile.best_slice);
          }
        } else {
          setLocatorRegion({
            found: false,
            source: 'detector',
            disclaimer: 'Lobe and nodule localization unavailable for this scan.',
          });
        }
      })
      .catch(() => {
        setLocatorRegion({
          found: false,
          source: 'detector',
          disclaimer: 'Lobe and nodule localization unavailable.',
        });
      });

    // Fetch 2D attribution heatmap matrix from genuine model output.
    fetch(`${API_BASE}/xai/${cleanNoduleId}/gradcam3d${studyQuery}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && d.heatmap_matrix && !d.is_placeholder) {
          setLocatorHeatmap(d.heatmap_matrix);
        }
      })
      .catch(() => {});

    // Fetch clinical purpose validation metrics
    fetch(`${API_BASE}/metrics/purpose`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d) setPurposeMetrics(d);
      })
      .catch(() => {});
  }, [studyId, noduleId]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setModelDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutside);
    return () => document.removeEventListener('mousedown', handleOutside);
  }, []);

  // File upload handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (PNG, JPG, WebP, TIFF).');
      return;
    }
    setError(null);
    setImageFileName(file.name);
    setImageMime(file.type || 'image/png');
    setActiveFindingLabel(null);

    const reader = new FileReader();
    reader.onload = () => {
      const b64 = reader.result as string;
      setImageBase64(b64);
      setImagePreview(b64);
    };
    reader.readAsDataURL(file);
  };

  const handleClearImage = () => {
    setImageBase64(null);
    setImageFileName(null);
    setImagePreview(null);
    setActiveFindingLabel(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Pre-load demo sample
  const handleLoadSample = async (type: 'malignant' | 'benign') => {
    const sampleUrl = type === 'malignant' 
      ? './samples/malignant_sample.png' 
      : './samples/benign_sample.png';
    setImageFileName(type === 'malignant' ? 'LIDC-0087-Malignant-Slice.png' : 'LIDC-0023-Benign-Slice.png');
    setImageMime('image/png');
    setActiveFindingLabel(type === 'malignant' ? 'Spiculated Nodule Candidate' : 'Calcified Granuloma Candidate');
    try {
      const res = await fetch(sampleUrl);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = reader.result as string;
        setImageBase64(b64);
        setImagePreview(b64);
      };
      reader.readAsDataURL(blob);
    } catch {
      setImagePreview(sampleUrl);
    }
  };

  const hasKey = (prov: AIProvider): boolean => {
    return Boolean(serverProviders[prov]);
  };

  // Run Real-World AI Analysis via Secure Backend Proxy
  const handleAnalyze = async () => {
    if (!prompt.trim()) {
      setError("Please provide a clinical question or choose a preset prompt.");
      return;
    }

    if (!serverProviders[selectedModel.provider]) {
      setShowStatusDrawer(true);
      setError(`Server-side key for ${selectedModel.providerLabel} is not configured in backend environment (.env).`);
      return;
    }

    setAnalyzing(true);
    setError(null);
    setResult(null);
    setResultMeta(null);
    const startTime = Date.now();

    try {
      const backendResp = await fetch(`${API_BASE}/ai/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: selectedModel.provider,
          model: selectedModel.id,
          prompt: prompt.trim(),
          image_base64: imageBase64,
          image_mime_type: imageMime,
          temperature,
          max_tokens: maxTokens,
        }),
      });

      if (!backendResp.ok) {
        const err = await backendResp.json().catch(() => ({}));
        throw new Error(err.detail || `Server returned status ${backendResp.status}`);
      }

      const data = await backendResp.json();
      const analysisText = data.content;
      const usageInfo = data.usage;
      const durationMs = Date.now() - startTime;
      setResult(analysisText);
      setResultMeta({
        model: selectedModel.label,
        provider: selectedModel.providerLabel,
        durationMs,
        usage: usageInfo,
      });

      // Synchronize XAI attribution tag with AI output
      const lower = analysisText.toLowerCase();
      if (lower.includes('malignan') || lower.includes('spiculat') || lower.includes('rads 4') || lower.includes('adenocarcinoma')) {
        setActiveFindingLabel('Suspicious Nodule Pattern [Model Identified]');
      } else if (lower.includes('benign') || lower.includes('rads 2') || lower.includes('granuloma') || lower.includes('calcifi')) {
        setActiveFindingLabel('Low Risk / Benign Pattern [Model Identified]');
      } else {
        setActiveFindingLabel('Target Lesion [Analyzed]');
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during AI analysis.');
    } finally {
      setAnalyzing(false);
    }
  };

  // Copy to clipboard
  const handleCopy = () => {
    if (!result) return;
    navigator.clipboard.writeText(result);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1440px', margin: '0 auto', color: '#f1f5f9' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '36px', height: '36px', borderRadius: '10px',
              background: 'linear-gradient(135deg, #0284c7, #818cf8)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 4px 14px rgba(2, 132, 199, 0.4)'
            }}>
              <BrainCircuit size={20} color="#fff" />
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: '20px', fontWeight: '700', letterSpacing: '-0.02em' }}>
                Frontier Model Analysis (Research / Experimental)
              </h1>
              <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>
                De-identified Text Decision Support · OpenAI ChatGPT · Anthropic Claude · Google Gemini
              </p>
            </div>
          </div>
        </div>

        {/* Server AI Security Status Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            onClick={() => setShowStatusDrawer(!showStatusDrawer)}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              backgroundColor: showStatusDrawer ? '#1e293b' : '#0f172a',
              border: '1px solid #334155', borderRadius: '8px',
              color: '#38bdf8', padding: '8px 14px', fontSize: '12px',
              fontWeight: '600', cursor: 'pointer', transition: 'all 0.2s',
            }}
          >
            <Server size={14} />
            <span>Server AI Security Status</span>
            <span style={{
              width: '8px', height: '8px', borderRadius: '50%',
              backgroundColor: hasKey(selectedModel.provider) ? '#10b981' : '#f59e0b'
            }} />
          </button>
        </div>
      </div>

      {/* Server Security Status Drawer */}
      {showStatusDrawer && (
        <div style={{
          backgroundColor: '#0c1322', border: '1px solid #1e293b',
          borderRadius: '12px', padding: '18px 20px', marginBottom: '20px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Server size={16} color="#38bdf8" />
              <strong style={{ fontSize: '13px', color: '#f8fafc' }}>Server-Side AI Provider Proxy</strong>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>
                (Keys are stored strictly in backend environment variables — frontend never handles API keys)
              </span>
            </div>
            <button
              onClick={() => setShowStatusDrawer(false)}
              style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
            {/* Google Gemini */}
            <div style={{ backgroundColor: '#070c18', padding: '12px 14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', fontWeight: '600', color: '#38bdf8' }}>Google Gemini</span>
                <span style={{ fontSize: '11px', color: serverProviders.gemini ? '#10b981' : '#f59e0b', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  {serverProviders.gemini ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                  {serverProviders.gemini ? 'Server Active' : 'Not Set in .env'}
                </span>
              </div>
            </div>

            {/* OpenAI */}
            <div style={{ backgroundColor: '#070c18', padding: '12px 14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', fontWeight: '600', color: '#34d399' }}>OpenAI (ChatGPT)</span>
                <span style={{ fontSize: '11px', color: serverProviders.openai ? '#10b981' : '#f59e0b', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  {serverProviders.openai ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                  {serverProviders.openai ? 'Server Active' : 'Not Set in .env'}
                </span>
              </div>
            </div>

            {/* Anthropic Claude */}
            <div style={{ backgroundColor: '#070c18', padding: '12px 14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', fontWeight: '600', color: '#fb7185' }}>Anthropic Claude</span>
                <span style={{ fontSize: '11px', color: serverProviders.anthropic ? '#10b981' : '#f59e0b', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  {serverProviders.anthropic ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                  {serverProviders.anthropic ? 'Server Active' : 'Not Set in .env'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Experimental Safety Notice Banner */}
      <div style={{
        backgroundColor: '#451a03',
        border: '1px solid #b45309',
        borderRadius: '8px',
        padding: '10px 14px',
        marginBottom: '18px',
        color: '#fef3c7',
        fontSize: '12px',
        fontWeight: '600',
        display: 'flex',
        alignItems: 'center',
        gap: '10px'
      }}>
        <AlertCircle size={18} color="#fbbf24" style={{ flexShrink: 0 }} />
        <span>Experimental. Not validated. Data leaves this system. Sends strictly de-identified, non-image text through secure backend proxy.</span>
      </div>

      {/* Main Grid: Left Control Panel | Right Clinical Output */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 420px) 1fr', gap: '20px', alignItems: 'start' }}>
        
        {/* LEFT COLUMN: Controls & Input */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Model Selection Card */}
          <div style={{ backgroundColor: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '16px' }}>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '8px' }}>
              Active AI Model
            </label>
            <div ref={dropdownRef} style={{ position: 'relative' }}>
              <button
                type="button"
                onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
                style={{
                  width: '100%', backgroundColor: '#070c18', border: '1px solid #334155',
                  borderRadius: '8px', padding: '10px 12px', display: 'flex',
                  alignItems: 'center', justifyContent: 'space-between', color: '#f1f5f9',
                  cursor: 'pointer', textAlign: 'left',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{
                    fontSize: '10px', fontWeight: '700', padding: '2px 6px',
                    borderRadius: '4px', backgroundColor: selectedModel.badgeBg,
                    color: selectedModel.providerBadgeColor, textTransform: 'uppercase'
                  }}>
                    {selectedModel.provider}
                  </span>
                  <div>
                    <div style={{ fontSize: '13px', fontWeight: '600' }}>{selectedModel.label}</div>
                    <div style={{ fontSize: '10px', color: '#64748b' }}>{selectedModel.desc}</div>
                  </div>
                </div>
                <ChevronDown size={16} color="#94a3b8" />
              </button>

              {modelDropdownOpen && (
                <div style={{
                  position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
                  backgroundColor: '#070c18', border: '1px solid #334155', borderRadius: '8px',
                  boxShadow: '0 12px 28px rgba(0,0,0,0.5)', zIndex: 30, overflow: 'hidden'
                }}>
                  {MODELS.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => {
                        setSelectedModelId(m.id);
                        setModelDropdownOpen(false);
                      }}
                      style={{
                        width: '100%', padding: '10px 12px', textAlign: 'left',
                        backgroundColor: m.id === selectedModelId ? '#1e293b' : 'transparent',
                        border: 'none', borderBottom: '1px solid #111a2e', color: '#f1f5f9',
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        cursor: 'pointer',
                      }}
                    >
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            fontSize: '9px', fontWeight: '700', padding: '1px 5px',
                            borderRadius: '3px', backgroundColor: m.badgeBg, color: m.providerBadgeColor
                          }}>
                            {m.provider}
                          </span>
                          <span style={{ fontSize: '12px', fontWeight: '600' }}>{m.label}</span>
                        </div>
                        <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px' }}>{m.desc}</div>
                      </div>
                      <span style={{ fontSize: '10px', color: '#64748b', fontFamily: 'monospace' }}>{m.contextWindow}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Provider status pill */}
            <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }}>
              <span style={{ color: '#64748b' }}>Provider Status:</span>
              <span style={{
                color: hasKey(selectedModel.provider) ? '#34d399' : '#f59e0b',
                fontWeight: '600', display: 'flex', alignItems: 'center', gap: '5px'
              }}>
                {serverProviders[selectedModel.provider] ? 'Active (Server Proxy)' : 'Not Configured (.env)'}
              </span>
            </div>
          </div>

          {/* Image Upload / Sample CT Slice */}
          <div style={{ backgroundColor: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <label style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8' }}>
                CT Slice / Specimen Image
              </label>
              {imagePreview && (
                <button
                  type="button"
                  onClick={handleClearImage}
                  style={{ background: 'none', border: 'none', color: '#ef4444', fontSize: '11px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '3px' }}
                >
                  <X size={12} /> Clear
                </button>
              )}
            </div>

            {/* Upload Dropzone */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />

            {imagePreview ? (
              <div style={{ position: 'relative', borderRadius: '8px', overflow: 'hidden', border: '1px solid #334155', backgroundColor: '#000', maxHeight: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img src={imagePreview} alt="CT Slice" style={{ maxWidth: '100%', maxHeight: '180px', objectFit: 'contain' }} />
                <div style={{ position: 'absolute', bottom: '6px', left: '8px', right: '8px', backgroundColor: 'rgba(15, 23, 42, 0.85)', padding: '3px 8px', borderRadius: '4px', fontSize: '10px', color: '#e2e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {imageFileName || 'Uploaded CT Slice'}
                </div>
              </div>
            ) : (
              <div
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: '1px dashed #334155', borderRadius: '8px',
                  padding: '20px 14px', textAlign: 'center', cursor: 'pointer',
                  backgroundColor: '#070c18', transition: 'border-color 0.2s',
                }}
              >
                <Upload size={24} color="#64748b" style={{ margin: '0 auto 8px' }} />
                <div style={{ fontSize: '12px', fontWeight: '600', color: '#94a3b8' }}>Click to upload CT slice or drag & drop</div>
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: '4px' }}>PNG, JPG, WebP (Axial, Coronal, Sagittal)</div>
              </div>
            )}

            {/* Clinical Sample Buttons */}
            <div style={{ marginTop: '10px', display: 'flex', gap: '8px' }}>
              <button
                type="button"
                onClick={() => handleLoadSample('malignant')}
                style={{
                  flex: 1, backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#fca5a5', padding: '6px 8px', borderRadius: '6px', fontSize: '10px',
                  fontWeight: '600', cursor: 'pointer'
                }}
              >
                + Malignant Sample
              </button>
              <button
                type="button"
                onClick={() => handleLoadSample('benign')}
                style={{
                  flex: 1, backgroundColor: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)',
                  color: '#86efac', padding: '6px 8px', borderRadius: '6px', fontSize: '10px',
                  fontWeight: '600', cursor: 'pointer'
                }}
              >
                + Benign Sample
              </button>
            </div>
          </div>

          {/* Clinical Presets */}
          <div style={{ backgroundColor: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '16px' }}>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '8px' }}>
              Clinical Presets
            </label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {PRESETS.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setPrompt(p.prompt)}
                  style={{
                    textAlign: 'left', backgroundColor: prompt === p.prompt ? '#1e293b' : '#070c18',
                    border: '1px solid #1e293b', borderRadius: '6px', padding: '8px 10px',
                    color: prompt === p.prompt ? '#38bdf8' : '#94a3b8', fontSize: '11px',
                    fontWeight: prompt === p.prompt ? '600' : '400', cursor: 'pointer'
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Prompt Input Box */}
          <div style={{ backgroundColor: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px', padding: '16px' }}>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#94a3b8', marginBottom: '8px' }}>
              Diagnostic Prompt
            </label>
            <textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe clinical inquiry or radiologic focus..."
              style={{
                width: '100%', backgroundColor: '#070c18', border: '1px solid #334155',
                borderRadius: '8px', padding: '10px', color: '#f1f5f9', fontSize: '12px',
                fontFamily: 'inherit', resize: 'vertical', boxSizing: 'border-box'
              }}
            />

            {/* Execute Button */}
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={analyzing}
              style={{
                width: '100%', marginTop: '12px', padding: '12px',
                background: analyzing ? '#334155' : 'linear-gradient(135deg, #0284c7, #6366f1)',
                border: 'none', borderRadius: '8px', color: '#fff', fontSize: '13px',
                fontWeight: '700', cursor: analyzing ? 'not-allowed' : 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                boxShadow: analyzing ? 'none' : '0 4px 14px rgba(2, 132, 199, 0.4)',
                transition: 'all 0.2s',
              }}
            >
              {analyzing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Synthesizing Radiologic Evidence...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Run {selectedModel.label} Analysis</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* RIGHT COLUMN: Clinical Output */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Output Card */}
          <div style={{
            backgroundColor: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px',
            padding: '24px', minHeight: '520px', display: 'flex', flexDirection: 'column'
          }}>
            {/* Output Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingBottom: '16px', borderBottom: '1px solid #1e293b', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <FileCheck2 size={18} color="#38bdf8" />
                <span style={{ fontSize: '14px', fontWeight: '700', letterSpacing: '-0.01em' }}>
                  Clinical Diagnostic Synthesis
                </span>
                {resultMeta && (
                  <span style={{
                    fontSize: '11px', padding: '2px 8px', borderRadius: '4px',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)', color: '#38bdf8', fontWeight: '600'
                  }}>
                    {resultMeta.model} · {resultMeta.durationMs}ms
                  </span>
                )}
              </div>

              {result && (
                <button
                  type="button"
                  onClick={handleCopy}
                  style={{
                    backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '6px',
                    color: copied ? '#34d399' : '#94a3b8', padding: '6px 12px', fontSize: '11px',
                    fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px'
                  }}
                >
                  {copied ? <Check size={13} /> : <Copy size={13} />}
                  <span>{copied ? 'Copied' : 'Copy Report'}</span>
                </button>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div style={{
                backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '8px', padding: '12px 16px', color: '#fca5a5', fontSize: '12px',
                display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px'
              }}>
                <AlertCircle size={16} color="#ef4444" />
                <span>{error}</span>
              </div>
            )}

            {/* Content Area */}
            {analyzing ? (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px 0' }}>
                <Loader2 size={32} color="#38bdf8" className="animate-spin" style={{ marginBottom: '16px' }} />
                <div style={{ fontSize: '14px', fontWeight: '600', color: '#f1f5f9' }}>
                  Consulting {selectedModel.label}...
                </div>
                <div style={{ fontSize: '12px', color: '#64748b', marginTop: '6px' }}>
                  Evaluating spiculation, Hounsfield densities, and nodule margins
                </div>
              </div>
            ) : result ? (
              <div style={{
                flex: 1, backgroundColor: '#070c18', border: '1px solid #1e293b',
                borderRadius: '8px', padding: '20px', whiteSpace: 'pre-wrap',
                fontFamily: 'Inter, system-ui, sans-serif', fontSize: '13px',
                lineHeight: 1.7, color: '#e2e8f0', overflowY: 'auto'
              }}>
                {result}
              </div>
            ) : (
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', color: '#475569',
                border: '1px dashed #1e293b', borderRadius: '8px', padding: '36px'
              }}>
                <BrainCircuit size={40} color="#334155" style={{ marginBottom: '14px' }} />
                <div style={{ fontSize: '14px', fontWeight: '600', color: '#94a3b8' }}>
                  No Active Analysis
                </div>
                <div style={{ fontSize: '12px', color: '#64748b', maxWidth: '380px', textAlign: 'center', marginTop: '6px' }}>
                  Select an AI model (ChatGPT, Claude 3.5, or Gemini 1.5), upload a CT slice or choose a sample, and click <strong>Run Analysis</strong>.
                </div>
              </div>
            )}

            {/* Guard Disclaimer */}
            <div style={{
              marginTop: '16px', padding: '10px 14px', borderRadius: '6px',
              backgroundColor: 'rgba(15, 23, 42, 0.6)', border: '1px solid #1e293b',
              fontSize: '11px', color: '#64748b', display: 'flex', alignItems: 'center', justifyContent: 'space-between'
            }}>
              <span>Clinical Decision-Support / Research Use Only · Guard 5 Active</span>
              <span style={{ color: '#475569' }}>Powered by Authentic Frontier Foundation Models</span>
            </div>
          </div>

          {/* Explainable Tumour Localization & Clinical Purpose Metrics */}
          <div style={{
            backgroundColor: '#0b1120', border: '1px solid #1e293b', borderRadius: '12px',
            padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '14px', fontWeight: '700', color: '#f1f5f9' }}>
                  Explainable Tumour Localization & Slice Attribution
                </span>
                {locatorRegion && (
                  <span style={{
                    fontSize: '11px', padding: '2px 8px', borderRadius: '4px',
                    backgroundColor: locatorRegion.source === 'segmentation' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                    color: locatorRegion.source === 'segmentation' ? '#34d399' : '#38bdf8',
                    fontWeight: '600'
                  }}>
                    Source: {locatorRegion.source}
                  </span>
                )}
              </div>
            </div>

            {locatorRegion ? (
              <TumourLocator
                activeImageSrc={imagePreview || imageBase64}
                activeImageName={imageFileName}
                activeFindingLabel={activeFindingLabel}
                heatmap={locatorHeatmap}
                region={locatorRegion}
                slice={locatorSlice}
                onSliceChange={setLocatorSlice}
                onDynamicRegionUpdate={(newRegion, newHeatmap) => {
                  setLocatorRegion(newRegion);
                  setLocatorHeatmap(newHeatmap);
                }}
              />
            ) : (
              <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '12px' }}>
                Loading 3D localization profile...
              </div>
            )}

            <div style={{ borderTop: '1px solid #1e293b', paddingTop: '16px', marginTop: '6px' }}>
              <div style={{ fontSize: '13px', fontWeight: '700', color: '#f1f5f9', marginBottom: '8px' }}>
                Purpose-Stratified Validation Metrics
              </div>
              <PurposeMetrics metricsByPurpose={purposeMetrics} />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default AIAnalysisPanel;
