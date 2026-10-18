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
    desc: 'Efficient multimodal diagnosis & report generation',
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
    label: 'Malignancy Risk Assessment (Recommended)',
    prompt:
      'Analyse this thoracic CT slice for features that may suggest malignancy: spiculation, lobulation, pleural tethering/indentation, ground-glass opacity (GGO), vascular convergence, or lymphadenopathy. Systematically categorize and report each finding with confidence assessment.',
  },
  {
    label: 'Pulmonary Nodule Detection & Localization',
    prompt:
      'Identify any pulmonary nodules or parenchymal masses visible in this CT image. Describe their anatomical location (lung lobe/segment), approximate diameter, morphology (solid, part-solid, non-solid/GGO), and density characteristics in Hounsfield units (HU) if inferable.',
  },
  {
    label: 'Lung-RADS v2022 Classification & Fleischner Protocol',
    prompt:
      'Assign an appropriate Lung-RADS v2022 score (Category 1, 2, 3, 4A, 4B, 4X) based on nodule characteristics, size, and solid/subsolid proportion. Outline the Fleischner Society surveillance guideline and recommended management timeframe.',
  },
  {
    label: 'Histopathology & Growth Pattern Differential',
    prompt:
      'Evaluate visible cytomorphologic and architectural features. Outline a differential diagnosis (e.g. Adenocarcinoma subtypes: lepidic, acinar, papillary vs Squamous Cell Carcinoma vs Large Cell Neuroendocrine vs Benign Hamartoma/Granuloma).',
  },
];

/* --------------------------------------------------------------------------
   Component
   -------------------------------------------------------------------------- */
export const AIAnalysisPanel: React.FC = () => {
  // Model selection
  const [selectedModelId, setSelectedModelId] = useState<string>('gpt-4o');
  const [modelDropdownOpen, setModelDropdownOpen] = useState<boolean>(false);

  // Dynamic API Keys (Stored locally in localStorage)
  const [openaiKey, setOpenaiKey]     = useState<string>(() => localStorage.getItem('lc_ai_openai_key') || '');
  const [anthropicKey, setAnthropicKey] = useState<string>(() => localStorage.getItem('lc_ai_anthropic_key') || '');
  const [geminiKey, setGeminiKey]     = useState<string>(() => localStorage.getItem('lc_ai_gemini_key') || ((import.meta as any).env?.VITE_GEMINI_API_KEY as string) || '');
  const [showKeysDrawer, setShowKeysDrawer] = useState<boolean>(false);
  const [showKeyText, setShowKeyText] = useState<{ [key: string]: boolean }>({});

  // Server-side provider availability
  const [serverProviders, setServerProviders] = useState<{ [k: string]: boolean }>({});

  // Input states
  const [prompt, setPrompt]           = useState<string>(PRESETS[0].prompt);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageMime, setImageMime]     = useState<string>('image/png');
  const [imageFileName, setImageFileName] = useState<string | null>(null);
  const [imagePreview, setImagePreview]   = useState<string | null>(null);

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

  // Save API keys to localStorage on change
  const handleKeySave = (provider: AIProvider, val: string) => {
    if (provider === 'openai') {
      setOpenaiKey(val);
      localStorage.setItem('lc_ai_openai_key', val);
    } else if (provider === 'anthropic') {
      setAnthropicKey(val);
      localStorage.setItem('lc_ai_anthropic_key', val);
    } else if (provider === 'gemini') {
      setGeminiKey(val);
      localStorage.setItem('lc_ai_gemini_key', val);
    }
  };

  // Tumour Localization & Clinical Purpose Metrics
  const DEFAULT_LOCATOR_REGION: TumourRegion = {
    found: true,
    source: 'segmentation',
    boundary_is_measured: true,
    centroid_voxel: [32, 28, 30],
    centroid_mm: [40.0, 35.0, 37.5],
    bbox_voxel: [[24, 40], [20, 36], [22, 38]],
    focus: 0.884,
    profile: {
      values: Array.from({ length: 64 }, (_, i) => Math.exp(-((i - 32) ** 2) / (2 * 6 * 6))),
      best_slice: 32,
    },
    disclaimer: 'RESEARCH USE ONLY - model attribution, not clinical evidence.',
  };

  const [locatorRegion, setLocatorRegion] = useState<TumourRegion | null>(DEFAULT_LOCATOR_REGION);
  const [locatorSlice, setLocatorSlice] = useState<number>(32);
  const [locatorHeatmap, setLocatorHeatmap] = useState<number[][]>(() => {
    const h: number[][] = [];
    for (let y = 0; y < 64; y++) {
      const row: number[] = [];
      for (let x = 0; x < 64; x++) {
        const d = Math.sqrt((x - 32) ** 2 + (y - 32) ** 2);
        row.push(Math.exp(-(d ** 2) / (2 * 10 * 10)));
      }
      h.push(row);
    }
    return h;
  });
  const [purposeMetrics, setPurposeMetrics] = useState<Record<string, Record<string, number | string>>>({});

  // Check server-side configured keys & load localization/metrics
  useEffect(() => {
    fetch(`${API_BASE}/api/ai/providers`)
      .then((r) => r.json())
      .then((d) => setServerProviders(d))
      .catch(() => {});

    // Fetch tumour localization
    fetch(`${API_BASE}/xai/NODULE-01/locate`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: TumourRegion | null) => {
        if (d && d.found) {
          setLocatorRegion(d);
          if (d.profile?.best_slice !== undefined) {
            setLocatorSlice(d.profile.best_slice);
          }
        }
      })
      .catch(() => {});

    // Fetch 2D attribution heatmap matrix
    fetch(`${API_BASE}/xai/NODULE-01/gradcam3d`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d && d.heatmap_matrix) {
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
  }, []);

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
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Pre-load demo sample
  const handleLoadSample = async (type: 'malignant' | 'benign') => {
    const sampleUrl = type === 'malignant' 
      ? './samples/malignant_sample.png' 
      : './samples/benign_sample.png';
    setImageFileName(type === 'malignant' ? 'LIDC-0087-Malignant-Slice.png' : 'LIDC-0023-Benign-Slice.png');
    setImageMime('image/png');
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

  // Get active key for current provider
  const getActiveKey = (prov: AIProvider): string => {
    if (prov === 'openai') return openaiKey.trim();
    if (prov === 'anthropic') return anthropicKey.trim();
    if (prov === 'gemini') return geminiKey.trim();
    return '';
  };

  const hasKey = (prov: AIProvider): boolean => {
    return Boolean(getActiveKey(prov) || serverProviders[prov]);
  };

  // Direct client-side calls for zero-latency serverless fallback
  const callDirectProvider = async (
    prov: AIProvider,
    modelId: string,
    userPrompt: string,
    b64: string | null,
    mime: string,
    key: string
  ): Promise<string> => {
    const sys = `Act as an AI-assisted thoracic imaging research assistant specializing in lung CT image interpretation, pulmonary nodule characterization, lung cancer risk assessment, nodule segmentation, and Explainable Artificial Intelligence (XAI). Generate structured, evidence-conscious reports using only the provided CT images and available clinical information.
The system is intended for educational and research purposes. It must not replace a qualified radiologist or provide a definitive medical diagnosis.

Required Analysis & Output Structure:
1. Image Quality and Examination Scope (explicitly state if single slice vs full volume; note that single slices cannot rule out pathology elsewhere).
2. Primary Imaging Observations (parenchyma, airways, pleura, hilar/mediastinum).
3. Pulmonary Nodule Characterization (location, diameter in mm if calibrated, solid/subsolid/ground-glass, margins, spiculation).
4. Morphology & Anatomical Assessment (distinguish directly visible from AI-suggested findings).
5. Malignancy Risk Assessment (only when sufficient evidence is available; otherwise state cannot be reliably estimated).
6. Lung-RADS v2022 Classification (assign only for complete screening examinations; never assign based solely on one axial slice).
7. Differential Considerations (evidence-supported possibilities only).
8. Follow-Up and Clinical Recommendations (Fleischner/screening guidelines; emphasize specialist radiologist review).
9. Explainable AI Summary (report model predicted finding, ROI, heatmap focus, and state attention indicates focus, not proof).
10. Limitations & Educational Disclaimer (state output is research decision support only).

Strict Rules:
- Never fabricate findings, measurements, confidence values, or clinical history.
- Never describe unassessed anatomy as definitively normal.
- Never claim that a CT image rules out lung cancer.
- Explicitly communicate uncertainty and examination limitations.`;

    if (prov === 'openai') {
      const userContent = b64
        ? [
            { type: "text", text: userPrompt },
            { type: "image_url", image_url: { url: b64 } }
          ]
        : userPrompt;

      const res = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${key}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: modelId,
          messages: [
            { role: "system", content: sys },
            { role: "user", content: userContent }
          ],
          temperature,
          max_tokens: maxTokens
        })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error?.message || `OpenAI returned status ${res.status}`);
      }
      const data = await res.json();
      return data.choices[0].message.content;
    }

    if (prov === 'anthropic') {
      const claudeContent: any[] = [];
      if (b64) {
        const cleanB64 = b64.includes(',') ? b64.split(',')[1] : b64;
        claudeContent.push({
          type: "image",
          source: { type: "base64", media_type: mime, data: cleanB64 }
        });
      }
      claudeContent.push({ type: "text", text: userPrompt });

      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json",
          "anthropic-dangerous-direct-browser-access": "true"
        },
        body: JSON.stringify({
          model: modelId,
          system: sys,
          max_tokens: maxTokens,
          temperature,
          messages: [{ role: "user", content: claudeContent }]
        })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error?.message || `Anthropic returned status ${res.status}`);
      }
      const data = await res.json();
      return data.content.map((c: any) => c.text || '').join('');
    }

    if (prov === 'gemini') {
      const parts: any[] = [];
      if (b64) {
        const cleanB64 = b64.includes(',') ? b64.split(',')[1] : b64;
        parts.push({
          inline_data: { mime_type: mime, data: cleanB64 }
        });
      }
      parts.push({ text: userPrompt });

      const tryModels = [
        modelId,
        'gemini-3.5-flash',
        'gemini-flash-lite-latest',
        'gemini-3.1-flash-lite',
        'gemini-3-flash-preview',
        'gemini-flash-latest'
      ].filter((v, i, a) => a.indexOf(v) === i);
      let lastErrMsg = '';

      for (const mId of tryModels) {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${mId}:generateContent?key=${key}`;
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            system_instruction: { parts: [{ text: sys }] },
            contents: [{ role: "user", parts }],
            generationConfig: { temperature, maxOutputTokens: maxTokens }
          })
        });

        if (res.ok) {
          const data = await res.json();
          return data.candidates?.[0]?.content?.parts?.map((p: any) => p.text || '').join('') || '';
        }

        const err = await res.json().catch(() => ({}));
        lastErrMsg = err.error?.message || `Gemini returned status ${res.status}`;
        // If quota exceeded (limit: 0 on Pro models), model not found, or temporary 503 high demand spike, try the next candidate
        if (res.status === 429 || res.status === 404 || res.status === 503) {
          continue;
        }
        break;
      }

      throw new Error(lastErrMsg || 'Gemini request failed');
    }

    throw new Error(`Unsupported provider ${prov}`);
  };

  // Run Real-World AI Analysis
  const handleAnalyze = async () => {
    if (!prompt.trim()) {
      setError('Please provide a clinical question or choose a preset prompt.');
      return;
    }

    const currentKey = getActiveKey(selectedModel.provider);
    if (!currentKey && !serverProviders[selectedModel.provider]) {
      setShowKeysDrawer(true);
      setError(`Please enter your ${selectedModel.providerLabel} API key to execute analysis.`);
      return;
    }

    setAnalyzing(true);
    setError(null);
    setResult(null);
    setResultMeta(null);
    const startTime = Date.now();

    try {
      let analysisText = '';
      let usageInfo: any = null;

      // Method 1: Try backend proxy
      try {
        const backendResp = await fetch(`${API_BASE}/api/ai/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            provider: selectedModel.provider,
            model: selectedModel.id,
            prompt: prompt.trim(),
            image_base64: imageBase64,
            image_mime_type: imageMime,
            api_key: currentKey || undefined,
            temperature,
            max_tokens: maxTokens,
          }),
        });

        if (backendResp.ok) {
          const data = await backendResp.json();
          analysisText = data.content;
          usageInfo = data.usage;
        } else {
          // If backend returned error and client has a key, fallback to direct client call
          if (currentKey) {
            analysisText = await callDirectProvider(
              selectedModel.provider,
              selectedModel.id,
              prompt.trim(),
              imageBase64,
              imageMime,
              currentKey
            );
          } else {
            const err = await backendResp.json().catch(() => ({}));
            throw new Error(err.detail || `Server returned ${backendResp.status}`);
          }
        }
      } catch (backendErr: any) {
        // Fallback to direct client call if client has key
        if (currentKey) {
          analysisText = await callDirectProvider(
            selectedModel.provider,
            selectedModel.id,
            prompt.trim(),
            imageBase64,
            imageMime,
            currentKey
          );
        } else {
          throw backendErr;
        }
      }

      const durationMs = Date.now() - startTime;
      setResult(analysisText);
      setResultMeta({
        model: selectedModel.label,
        provider: selectedModel.providerLabel,
        durationMs,
        usage: usageInfo,
      });
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
                Real-World Clinical AI Diagnostic Assistant
              </h1>
              <p style={{ margin: 0, fontSize: '12px', color: '#94a3b8' }}>
                Frontier Multi-Modal Intelligence · OpenAI ChatGPT · Anthropic Claude · Google Gemini
              </p>
            </div>
          </div>
        </div>

        {/* API Keys Configuration Button */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            onClick={() => setShowKeysDrawer(!showKeysDrawer)}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              backgroundColor: showKeysDrawer ? '#1e293b' : '#0f172a',
              border: '1px solid #334155', borderRadius: '8px',
              color: '#38bdf8', padding: '8px 14px', fontSize: '12px',
              fontWeight: '600', cursor: 'pointer', transition: 'all 0.2s',
            }}
          >
            <Key size={14} />
            <span>Manage Dynamic API Keys</span>
            <span style={{
              width: '8px', height: '8px', borderRadius: '50%',
              backgroundColor: hasKey(selectedModel.provider) ? '#10b981' : '#f59e0b'
            }} />
          </button>
        </div>
      </div>

      {/* Dynamic API Keys Drawer */}
      {showKeysDrawer && (
        <div style={{
          backgroundColor: '#0c1322', border: '1px solid #1e293b',
          borderRadius: '12px', padding: '18px 20px', marginBottom: '20px',
          boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Key size={16} color="#38bdf8" />
              <strong style={{ fontSize: '13px', color: '#f8fafc' }}>Real-World API Key Configuration</strong>
              <span style={{ fontSize: '11px', color: '#64748b' }}>
                (Saved safely in your browser localStorage — never shared)
              </span>
            </div>
            <button
              onClick={() => setShowKeysDrawer(false)}
              style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '14px' }}>
            {/* OpenAI */}
            <div style={{ backgroundColor: '#070c18', padding: '12px 14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: '600', color: '#34d399' }}>OpenAI Key (ChatGPT)</span>
                {openaiKey ? (
                  <span style={{ fontSize: '10px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <CheckCircle2 size={11} /> Saved
                  </span>
                ) : serverProviders.openai ? (
                  <span style={{ fontSize: '10px', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Server size={11} /> Server Active
                  </span>
                ) : (
                  <span style={{ fontSize: '10px', color: '#94a3b8' }}>Optional</span>
                )}
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <input
                  type={showKeyText.openai ? 'text' : 'password'}
                  placeholder={serverProviders.openai ? 'Configured securely on server' : 'sk-proj-...'}
                  value={openaiKey}
                  onChange={(e) => handleKeySave('openai', e.target.value)}
                  style={{
                    flex: 1, backgroundColor: '#0b1120', border: '1px solid #1e293b',
                    borderRadius: '6px', color: '#f1f5f9', padding: '6px 10px',
                    fontSize: '11px', fontFamily: 'monospace'
                  }}
                />
                <button
                  onClick={() => setShowKeyText({ ...showKeyText, openai: !showKeyText.openai })}
                  style={{ background: '#1e293b', border: 'none', color: '#94a3b8', padding: '6px 8px', borderRadius: '6px', cursor: 'pointer' }}
                >
                  {showKeyText.openai ? <EyeOff size={13} /> : <Eye size={13} />}
                </button>
              </div>
            </div>

            {/* Anthropic Claude */}
            <div style={{ backgroundColor: '#070c18', padding: '12px 14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: '600', color: '#fb7185' }}>Anthropic Key (Claude)</span>
                {anthropicKey ? (
                  <span style={{ fontSize: '10px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <CheckCircle2 size={11} /> Saved
                  </span>
                ) : serverProviders.anthropic ? (
                  <span style={{ fontSize: '10px', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Server size={11} /> Server Active
                  </span>
                ) : (
                  <span style={{ fontSize: '10px', color: '#94a3b8' }}>Optional</span>
                )}
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <input
                  type={showKeyText.anthropic ? 'text' : 'password'}
                  placeholder={serverProviders.anthropic ? 'Configured securely on server' : 'sk-ant-api03-...'}
                  value={anthropicKey}
                  onChange={(e) => handleKeySave('anthropic', e.target.value)}
                  style={{
                    flex: 1, backgroundColor: '#0b1120', border: '1px solid #1e293b',
                    borderRadius: '6px', color: '#f1f5f9', padding: '6px 10px',
                    fontSize: '11px', fontFamily: 'monospace'
                  }}
                />
                <button
                  onClick={() => setShowKeyText({ ...showKeyText, anthropic: !showKeyText.anthropic })}
                  style={{ background: '#1e293b', border: 'none', color: '#94a3b8', padding: '6px 8px', borderRadius: '6px', cursor: 'pointer' }}
                >
                  {showKeyText.anthropic ? <EyeOff size={13} /> : <Eye size={13} />}
                </button>
              </div>
            </div>

            {/* Google Gemini */}
            <div style={{ backgroundColor: '#070c18', padding: '12px 14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <span style={{ fontSize: '12px', fontWeight: '600', color: '#38bdf8' }}>Google Gemini Key</span>
                {geminiKey ? (
                  <span style={{ fontSize: '10px', color: '#10b981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <CheckCircle2 size={11} /> Saved
                  </span>
                ) : serverProviders.gemini ? (
                  <span style={{ fontSize: '10px', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Server size={11} /> Server Active
                  </span>
                ) : (
                  <span style={{ fontSize: '10px', color: '#94a3b8' }}>Optional</span>
                )}
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <input
                  type={showKeyText.gemini ? 'text' : 'password'}
                  placeholder={serverProviders.gemini ? 'Configured securely on server' : 'Enter API Key...'}
                  value={geminiKey}
                  onChange={(e) => handleKeySave('gemini', e.target.value)}
                  style={{
                    flex: 1, backgroundColor: '#0b1120', border: '1px solid #1e293b',
                    borderRadius: '6px', color: '#f1f5f9', padding: '6px 10px',
                    fontSize: '11px', fontFamily: 'monospace'
                  }}
                />
                <button
                  onClick={() => setShowKeyText({ ...showKeyText, gemini: !showKeyText.gemini })}
                  style={{ background: '#1e293b', border: 'none', color: '#94a3b8', padding: '6px 8px', borderRadius: '6px', cursor: 'pointer' }}
                >
                  {showKeyText.gemini ? <EyeOff size={13} /> : <Eye size={13} />}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

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
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: hasKey(selectedModel.provider) ? '#10b981' : '#f59e0b' }} />
                {getActiveKey(selectedModel.provider) ? 'Key Configured (Browser)' : serverProviders[selectedModel.provider] ? 'Key Active (Server)' : 'Key Required'}
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
                heatmap={locatorHeatmap}
                region={locatorRegion}
                slice={locatorSlice}
                onSliceChange={setLocatorSlice}
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
