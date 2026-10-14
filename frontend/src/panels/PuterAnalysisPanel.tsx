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
  Globe,
  FileCheck2,
} from 'lucide-react';

/* --------------------------------------------------------------------------
   Puter.js declaration
   -------------------------------------------------------------------------- */
declare const puter: {
  ai: {
    chat: (
      prompt: string,
      imageUrlOrOptions?: string | object,
      options?: object
    ) => Promise<any>;
  };
  print: (msg: any) => void;
};

/* --------------------------------------------------------------------------
   Model options — GPT-6 series + legacy vision & Claude
   -------------------------------------------------------------------------- */
interface ModelOption {
  id: string;
  label: string;
  tier: string;
  tierColor: string;
  desc: string;
}

const MODELS: ModelOption[] = [
  {
    id: 'gpt-6-astra',
    label: 'GPT-6 Astra',
    tier: 'flagship',
    tierColor: '#a78bfa',
    desc: 'Flagship · Best diagnostic quality',
  },
  {
    id: 'gpt-6.1-sol',
    label: 'GPT-6.1 Sol',
    tier: 'mid',
    tierColor: '#38bdf8',
    desc: 'Mid-tier · Balanced speed & reasoning',
  },
  {
    id: 'gpt-6-luna',
    label: 'GPT-6 Luna',
    tier: 'mini',
    tierColor: '#4ade80',
    desc: 'Smallest · Fastest & lightweight',
  },
  {
    id: 'gpt-4o',
    label: 'GPT-4o Vision',
    tier: 'vision',
    tierColor: '#fb923c',
    desc: 'Multimodal vision model',
  },
  {
    id: 'claude-3-5-sonnet-20241022',
    label: 'Claude 3.5 Sonnet',
    tier: 'claude',
    tierColor: '#f472b6',
    desc: 'Anthropic · High clinical reasoning',
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
    label: 'Segmentation & Border Definition',
    prompt:
      'Describe the boundaries, margins, and geometric extent of the primary lesion visible in this slice so that a radiologist or automated algorithm could segment it. Comment on edge sharpness vs infiltrative contours.',
  },
  {
    label: 'Radiomics Imaging Biomarkers',
    prompt:
      'List imaging biomarkers relevant for radiomics extraction: lesion density distribution, internal homogeneity vs necrosis/cavitation, texture patterns, and surrounding parenchymal changes.',
  },
  {
    label: 'Comprehensive Diagnostic Report',
    prompt:
      'Generate a formal clinical radiology research impression for this CT slice. Include: 1) Findings, 2) Lung-RADS category suggestion (1-4X), 3) Recommended diagnostic follow-up.',
  },
];

/* --------------------------------------------------------------------------
   Preloaded sample CT scans
   -------------------------------------------------------------------------- */
const SAMPLE_SCANS = [
  {
    label: 'Malignant CT Scan (LIDC Sample)',
    path: '/samples/malignant_sample.png',
    badge: 'Malignant Suspicion',
    color: '#f43f5e',
  },
  {
    label: 'Benign CT Scan (LIDC Sample)',
    path: '/samples/benign_sample.png',
    badge: 'Benign Pattern',
    color: '#10b981',
  },
];

/* --------------------------------------------------------------------------
   Helper functions
   -------------------------------------------------------------------------- */
function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((res, rej) => {
    const reader = new FileReader();
    reader.onload = () => res(reader.result as string);
    reader.onerror = () => rej(reader.error);
    reader.readAsDataURL(file);
  });
}

function hexToRgb(hex: string): string {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.substring(0, 2), 16);
  const g = parseInt(clean.substring(2, 4), 16);
  const b = parseInt(clean.substring(4, 6), 16);
  return `${r}, ${g}, ${b}`;
}

/* --------------------------------------------------------------------------
   Main Component
   -------------------------------------------------------------------------- */
export const PuterAnalysisPanel: React.FC = () => {
  // Model selection
  const [model, setModel] = useState<string>(MODELS[0].id);

  // Image & prompt state
  const [inputMode, setInputMode] = useState<'upload' | 'url'>('upload');
  const [imageUrl, setImageUrl] = useState<string>('');
  const [imagePreview, setImagePreview] = useState<string>('');
  const [prompt, setPrompt] = useState<string>(PRESETS[0].prompt);
  const [showPresets, setShowPresets] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Web search tool
  const [webSearch, setWebSearch] = useState<boolean>(false);
  const [webUrl, setWebUrl] = useState<string>('');

  // Execution & feedback state
  const [loading, setLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [copied, setCopied] = useState<boolean>(false);
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Global paste support (Ctrl+V)
  useEffect(() => {
    const handlePaste = async (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            try {
              const dataUrl = await readFileAsDataURL(file);
              setImageUrl(dataUrl);
              setImagePreview(dataUrl);
              setError('');
            } catch {
              setError('Failed to process image from clipboard.');
            }
          }
        }
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const dataUrl = await readFileAsDataURL(file);
      setImagePreview(dataUrl);
      setImageUrl(dataUrl);
      setError('');
    } catch {
      setError('Failed to read the uploaded file.');
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      try {
        const dataUrl = await readFileAsDataURL(file);
        setImagePreview(dataUrl);
        setImageUrl(dataUrl);
        setError('');
      } catch {
        setError('Failed to read the dropped file.');
      }
    }
  };

  const loadSampleScan = async (samplePath: string) => {
    try {
      setLoading(true);
      setError('');
      const res = await fetch(samplePath);
      if (!res.ok) throw new Error('Could not fetch sample scan.');
      const blob = await res.blob();
      const dataUrl = await readFileAsDataURL(new File([blob], 'sample.png', { type: blob.type }));
      setImageUrl(dataUrl);
      setImagePreview(dataUrl);
    } catch (err: any) {
      setError(err?.message || 'Failed to load sample scan.');
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setImageUrl('');
    setImagePreview('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const copyResponse = () => {
    if (!response) return;
    navigator.clipboard.writeText(response);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const checkUrlIssues = (url: string) => {
    if (!url) return null;
    const lower = url.toLowerCase();
    if (
      lower.includes('bing.com/images/search') ||
      lower.includes('google.com/search') ||
      lower.includes('google.com/imgres') ||
      lower.includes('yahoo.com/search')
    ) {
      return 'The link you entered is a search engine results page, not a direct image. Search engines block AI crawlers. Please right-click the image, choose "Copy Image", and press Ctrl+V to paste it directly here, or save it and use 📁 Upload!';
    }
    return null;
  };

  // Run AI Analysis via Puter
  const analyse = async () => {
    if (!imageUrl.trim() && !webSearch) {
      setError('Please provide an image: Upload a file, paste with Ctrl+V, or select a sample scan below.');
      return;
    }
    if (!prompt.trim()) {
      setError('Please enter an analysis prompt.');
      return;
    }

    if (inputMode === 'url' && imageUrl.trim()) {
      const urlWarning = checkUrlIssues(imageUrl);
      if (urlWarning) {
        setError(urlWarning);
        return;
      }
    }

    setLoading(true);
    setError('');
    setResponse('');
    setElapsedTime(null);
    const startTime = performance.now();

    try {
      if (typeof puter === 'undefined' || !puter.ai?.chat) {
        throw new Error('Puter.js library is not loaded. Please check your internet connection.');
      }

      const fullPrompt = webUrl.trim()
        ? `${prompt.trim()}\n\nReference: ${webUrl.trim()}`
        : prompt.trim();

      const options: Record<string, any> = { model };
      if (webSearch) options.tools = [{ type: 'web_search' }];

      const result =
        imageUrl.trim() && !webSearch
          ? await puter.ai.chat(fullPrompt, imageUrl.trim(), options)
          : await puter.ai.chat(fullPrompt, options);

      if (typeof result === 'string') {
        setResponse(result);
      } else if (result?.message?.content) {
        const content = result.message.content;
        setResponse(Array.isArray(content) ? content.map((c: any) => c.text ?? '').join('') : String(content));
      } else {
        setResponse(JSON.stringify(result, null, 2));
      }

      const totalTime = ((performance.now() - startTime) / 1000).toFixed(1);
      setElapsedTime(Number(totalTime));
    } catch (err: any) {
      setError(err?.message || 'Analysis failed. Please check network connectivity.');
    } finally {
      setLoading(false);
    }
  };

  /* ---- Inline styles ---- */
  const card: React.CSSProperties = {
    background: 'rgba(11, 17, 32, 0.85)',
    border: '1px solid rgba(255,255,255,0.07)',
    borderRadius: '14px',
    padding: '20px',
    backdropFilter: 'blur(12px)',
  };

  const inputStyle: React.CSSProperties = {
    width: '100%',
    background: '#060c1a',
    border: '1px solid rgba(6,182,212,0.18)',
    borderRadius: '8px',
    color: '#e2e8f0',
    padding: '10px 14px',
    fontSize: '13px',
    fontFamily: 'Inter, sans-serif',
    outline: 'none',
    boxSizing: 'border-box',
    transition: 'border-color 0.2s ease',
  };

  const lbl: React.CSSProperties = {
    fontSize: '11px',
    fontWeight: '600',
    color: '#64748b',
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
    marginBottom: '6px',
    display: 'block',
  };

  const btnPrimary: React.CSSProperties = {
    background: loading ? 'rgba(2,132,199,0.4)' : 'linear-gradient(135deg, #0284c7, #06b6d4)',
    border: 'none',
    borderRadius: '9px',
    color: '#ffffff',
    padding: '12px 28px',
    fontSize: '13px',
    fontWeight: '700',
    letterSpacing: '0.04em',
    cursor: loading ? 'not-allowed' : 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    transition: 'all 0.2s ease',
    boxShadow: loading ? 'none' : '0 4px 18px rgba(2,132,199,0.35)',
  };

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }}>
      {/* Header Banner */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '22px' }}>
        <div
          style={{
            width: '46px',
            height: '46px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #7c3aed, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 20px rgba(124,58,237,0.35)',
            flexShrink: 0,
          }}
        >
          <BrainCircuit size={24} color="#fff" />
        </div>
        <div>
          <div style={{ fontSize: '20px', fontWeight: '800', color: '#f8fafc', letterSpacing: '-0.02em' }}>
            Lung Cancer AI Image Analysis
          </div>
          <div style={{ fontSize: '12px', color: '#64748b', marginTop: '2px' }}>
            GPT-6 Vision · Claude 3.5 · Powered by Puter.js
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
          <div
            style={{
              background: 'rgba(56,189,248,0.1)',
              border: '1px solid rgba(56,189,248,0.25)',
              borderRadius: '7px',
              padding: '5px 12px',
              fontSize: '11px',
              fontWeight: '700',
              color: '#38bdf8',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Sparkles size={13} /> Puter AI Cloud Active
          </div>
        </div>
      </div>

      {/* ── Main Two-Column Grid ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.05fr 0.95fr', gap: '20px' }}>
        {/* ── LEFT COLUMN: Input Configuration ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Image Input Card */}
          <div style={card}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={lbl}>CT Image Input</span>
              <span style={{ fontSize: '11px', color: '#38bdf8' }}>Paste (Ctrl+V) anywhere</span>
            </div>

            {/* Quick 1-Click Sample Scans */}
            <div style={{ marginBottom: '14px' }}>
              <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '6px' }}>Preloaded LIDC Sample CT Scans:</div>
              <div style={{ display: 'flex', gap: '8px' }}>
                {SAMPLE_SCANS.map((sample, i) => (
                  <button
                    key={i}
                    onClick={() => loadSampleScan(sample.path)}
                    style={{
                      flex: 1,
                      padding: '7px 10px',
                      borderRadius: '7px',
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid rgba(255,255,255,0.1)',
                      color: '#cbd5e1',
                      fontSize: '11px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      textAlign: 'left',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <FileCheck2 size={13} color={sample.color} />
                    <span style={{ fontWeight: '600' }}>{sample.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Upload vs URL Tabs */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              {(['upload', 'url'] as const).map((mode) => (
                <button
                  key={mode}
                  id={`input-mode-${mode}`}
                  onClick={() => {
                    setInputMode(mode);
                    clearImage();
                  }}
                  style={{
                    flex: 1,
                    padding: '8px',
                    borderRadius: '7px',
                    border:
                      inputMode === mode ? '1px solid rgba(6,182,212,0.5)' : '1px solid rgba(255,255,255,0.07)',
                    background: inputMode === mode ? 'rgba(6,182,212,0.12)' : 'rgba(255,255,255,0.03)',
                    color: inputMode === mode ? '#06b6d4' : '#64748b',
                    fontSize: '12px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}
                >
                  {mode === 'upload' ? '📁 Direct Upload / Drag & Drop' : '🔗 Image URL'}
                </button>
              ))}
            </div>

            {inputMode === 'upload' ? (
              <>
                <input
                  id="file-input-element"
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    ...inputStyle,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px',
                    cursor: 'pointer',
                    minHeight: '80px',
                    border: isDragging ? '1px dashed #38bdf8' : '1px dashed rgba(6,182,212,0.3)',
                    background: isDragging ? 'rgba(56,189,248,0.08)' : '#060c1a',
                    color: '#64748b',
                    textAlign: 'center',
                  }}
                >
                  <Upload size={20} color="#06b6d4" />
                  <div style={{ fontSize: '12px', color: '#cbd5e1' }}>
                    Click to select CT JPEG/PNG, or drag & drop file here
                  </div>
                  <div style={{ fontSize: '10px', color: '#475569' }}>
                    Tip: You can also copy any image from your browser/DICOM and press Ctrl+V
                  </div>
                </div>
              </>
            ) : (
              <>
                <input
                  id="image-url-input"
                  type="url"
                  placeholder="https://example.com/slice-042.jpg"
                  value={imageUrl}
                  onChange={(e) => {
                    setImageUrl(e.target.value);
                    setImagePreview(e.target.value);
                  }}
                  style={inputStyle}
                />
                <div style={{ fontSize: '11px', color: '#475569', marginTop: '6px' }}>
                  Must be a direct link ending in .jpg/.png. For web searches or protected images, use 📁 Direct Upload or Ctrl+V.
                </div>
              </>
            )}
          </div>

          {/* Analysis Prompt Card */}
          <div style={card}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <span style={lbl}>Clinical Analysis Prompt</span>
              <button
                id="presets-toggle-btn"
                onClick={() => setShowPresets((p) => !p)}
                style={{
                  background: 'rgba(6,182,212,0.1)',
                  border: '1px solid rgba(6,182,212,0.2)',
                  borderRadius: '6px',
                  color: '#06b6d4',
                  fontSize: '11px',
                  fontWeight: '600',
                  padding: '3px 9px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <ClipboardList size={12} /> Clinical Presets
                <ChevronDown
                  size={11}
                  style={{ transform: showPresets ? 'rotate(180deg)' : 'none', transition: '0.2s' }}
                />
              </button>
            </div>

            {showPresets && (
              <div
                style={{
                  marginBottom: '10px',
                  border: '1px solid rgba(6,182,212,0.15)',
                  borderRadius: '8px',
                  overflow: 'hidden',
                }}
              >
                {PRESETS.map((p, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setPrompt(p.prompt);
                      setShowPresets(false);
                    }}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: '9px 12px',
                      background: prompt === p.prompt ? 'rgba(6,182,212,0.1)' : '#070f20',
                      border: 'none',
                      borderBottom: i < PRESETS.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none',
                      color: prompt === p.prompt ? '#06b6d4' : '#94a3b8',
                      fontSize: '12px',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ fontWeight: '600', marginBottom: '2px' }}>{p.label}</div>
                    <div style={{ fontSize: '11px', color: '#475569', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.prompt}
                    </div>
                  </button>
                ))}
              </div>
            )}

            <textarea
              id="analysis-prompt-textarea"
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              style={{ ...inputStyle, resize: 'vertical', lineHeight: '1.5' }}
            />
          </div>

          {/* Model Card Picker */}
          <div style={card}>
            <span style={lbl}>AI Model</span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px', marginBottom: '14px' }}>
              {MODELS.map((m) => {
                const isActive = model === m.id;
                return (
                  <button
                    key={m.id}
                    onClick={() => setModel(m.id)}
                    style={{
                      padding: '9px 8px',
                      borderRadius: '8px',
                      border: isActive ? `1px solid ${m.tierColor}` : '1px solid rgba(255,255,255,0.07)',
                      background: isActive ? `rgba(${hexToRgb(m.tierColor)}, 0.12)` : 'rgba(255,255,255,0.02)',
                      cursor: 'pointer',
                      textAlign: 'left',
                      transition: 'all 0.18s ease',
                      boxShadow: isActive ? `0 0 10px rgba(${hexToRgb(m.tierColor)}, 0.15)` : 'none',
                    }}
                  >
                    <div
                      style={{
                        fontSize: '9px',
                        fontWeight: '700',
                        textTransform: 'uppercase',
                        letterSpacing: '0.06em',
                        color: m.tierColor,
                        marginBottom: '3px',
                      }}
                    >
                      {m.tier}
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: isActive ? '#f8fafc' : '#94a3b8', marginBottom: '2px' }}>
                      {m.label}
                    </div>
                    <div style={{ fontSize: '10px', color: '#475569', lineHeight: '1.2' }}>{m.desc}</div>
                  </button>
                );
              })}
            </div>

            {/* Puter Web Search tool */}
            <div
              style={{
                marginBottom: '14px',
                padding: '10px 12px',
                background: 'rgba(255,255,255,0.02)',
                borderRadius: '8px',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: '#f8fafc' }}>
                    <Globe size={13} style={{ display: 'inline', marginRight: '5px' }} />
                    Web Search Tool
                  </div>
                  <div style={{ fontSize: '10px', color: '#64748b' }}>Passes live web search tool to puter.ai.chat</div>
                </div>
                <button
                  onClick={() => setWebSearch((w) => !w)}
                  style={{
                    width: '40px',
                    height: '22px',
                    borderRadius: '11px',
                    background: webSearch ? 'linear-gradient(135deg, #0284c7, #06b6d4)' : 'rgba(255,255,255,0.08)',
                    border: 'none',
                    cursor: 'pointer',
                    position: 'relative',
                  }}
                >
                  <span
                    style={{
                      position: 'absolute',
                      top: '3px',
                      left: webSearch ? '20px' : '3px',
                      width: '16px',
                      height: '16px',
                      borderRadius: '50%',
                      background: '#ffffff',
                      transition: 'left 0.2s',
                    }}
                  />
                </button>
              </div>
              {webSearch && (
                <div style={{ marginTop: '10px' }}>
                  <input
                    type="url"
                    placeholder="Optional reference URL (appended to prompt)"
                    value={webUrl}
                    onChange={(e) => setWebUrl(e.target.value)}
                    style={inputStyle}
                  />
                </div>
              )}
            </div>

            {error && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '8px',
                  background: 'rgba(244,63,94,0.08)',
                  border: '1px solid rgba(244,63,94,0.25)',
                  borderRadius: '8px',
                  padding: '10px 12px',
                  color: '#fda4af',
                  fontSize: '12px',
                  marginBottom: '14px',
                  lineHeight: '1.5',
                }}
              >
                <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>{error}</span>
              </div>
            )}

            <button id="run-ai-analysis-btn" onClick={analyse} disabled={loading} style={btnPrimary}>
              {loading ? (
                <>
                  <Loader2 size={16} style={{ animation: 'spin 0.9s linear infinite' }} />
                  Analysing with {MODELS.find((m) => m.id === model)?.label}…
                </>
              ) : (
                <>
                  <Sparkles size={16} /> Run AI Analysis
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── RIGHT COLUMN: Image Preview & AI Response ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Image Preview Card */}
          <div style={{ ...card, minHeight: '230px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={lbl}>Diagnostic CT Slice Preview</span>
              {imagePreview && (
                <button
                  onClick={clearImage}
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: '#f43f5e',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    fontSize: '11px',
                  }}
                >
                  <X size={13} /> Clear
                </button>
              )}
            </div>

            {imagePreview ? (
              <div style={{ position: 'relative', textAlign: 'center' }}>
                <img
                  src={imagePreview}
                  alt="CT scan slice"
                  onError={() => setError('Image could not be rendered. Please check file format.')}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '260px',
                    borderRadius: '8px',
                    objectFit: 'contain',
                    background: '#000000',
                    border: '1px solid rgba(255,255,255,0.1)',
                  }}
                />
                <div
                  style={{
                    marginTop: '6px',
                    fontSize: '10px',
                    color: '#475569',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '6px',
                  }}
                >
                  <CheckCircle2 size={11} color="#10b981" />
                  Image verified and embedded in base64 payload
                </div>
              </div>
            ) : (
              <div
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#334155',
                  gap: '10px',
                  minHeight: '180px',
                }}
              >
                <ImageIcon size={44} strokeWidth={1} />
                <span style={{ fontSize: '12px' }}>No CT scan loaded yet</span>
                <span style={{ fontSize: '11px', color: '#1e293b' }}>
                  Upload a slice, paste (Ctrl+V), or click a preloaded sample scan
                </span>
              </div>
            )}
          </div>

          {/* AI Response Card */}
          <div style={{ ...card, flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <BrainCircuit size={16} color="#06b6d4" />
              <span style={lbl}>AI Response</span>
              {elapsedTime !== null && (
                <span
                  style={{
                    fontSize: '10px',
                    color: '#10b981',
                    background: 'rgba(16,185,129,0.1)',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontWeight: '600',
                  }}
                >
                  {elapsedTime}s elapsed
                </span>
              )}
              {response && (
                <button
                  onClick={copyResponse}
                  style={{
                    marginLeft: 'auto',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '6px',
                    color: '#94a3b8',
                    padding: '4px 8px',
                    fontSize: '11px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  {copied ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
                  {copied ? 'Copied' : 'Copy Response'}
                </button>
              )}
            </div>

            {loading && !response ? (
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '14px',
                  padding: '50px 0',
                  color: '#475569',
                }}
              >
                <div
                  style={{
                    width: '46px',
                    height: '46px',
                    borderRadius: '50%',
                    border: '3px solid rgba(6,182,212,0.15)',
                    borderTop: '3px solid #06b6d4',
                    animation: 'spin 0.9s linear infinite',
                  }}
                />
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '13px', color: '#cbd5e1', fontWeight: '600', marginBottom: '4px' }}>
                    Analyzing CT Slice features…
                  </div>
                  <div style={{ fontSize: '11px', color: '#64748b' }}>
                    Sending to {MODELS.find((m) => m.id === model)?.label}…
                  </div>
                </div>
              </div>
            ) : response ? (
              <div
                style={{
                  background: '#040711',
                  borderRadius: '10px',
                  border: '1px solid rgba(6,182,212,0.15)',
                  padding: '16px',
                  color: '#cbd5e1',
                  fontSize: '13px',
                  lineHeight: '1.75',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: '440px',
                  overflowY: 'auto',
                }}
              >
                {response}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '50px 0', color: '#334155', fontSize: '13px' }}>
                Select an image, choose a model, and click &ldquo;Run AI Analysis&rdquo;.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mandatory Clinical Disclaimer Banner */}
      <div
        style={{
          marginTop: '20px',
          background: 'rgba(244,63,94,0.05)',
          border: '1px solid rgba(244,63,94,0.18)',
          borderRadius: '8px',
          padding: '12px 16px',
          fontSize: '11px',
          color: '#64748b',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}
      >
        <AlertCircle size={15} color="#f43f5e" style={{ flexShrink: 0 }} />
        <span>
          <strong style={{ color: '#f43f5e' }}>RESEARCH USE ONLY:</strong> AI assessments are experimental and
          generated for scientific investigation. They must not replace a board-certified radiologist&rsquo;s clinical
          interpretation. Never submit un-anonymized Protected Health Information (PHI).
        </span>
      </div>
    </div>
  );
};

export default PuterAnalysisPanel;
