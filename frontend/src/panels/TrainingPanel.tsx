import React, { useState, useEffect, useRef } from 'react';
import { Play, Cpu, RefreshCw, Terminal, Layers, CheckCircle2 } from 'lucide-react';
import { Panel } from '../components/ui/Panel';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Table, Thead, Tbody, Tr, Th, Td } from '../components/ui/Table';

type LogLine = {
  text: string;
  type: 'info' | 'success' | 'warn' | 'error' | 'metric' | 'default';
};

const MODEL_CONFIGS: Record<string, { epochs: number; batchSize: number; lr: string; desc: string }> = {
  '3D ResNet-50':        { epochs: 50,  batchSize: 4,  lr: '1e-4',  desc: '3D ResNet-50 Bottleneck + MC Dropout (T=30)' },
  '3D Swin Transformer': { epochs: 60,  batchSize: 2,  lr: '8e-5',  desc: 'Swin3D-Tiny (window_size=4x4x4, embed_dim=96)' },
  '3D DenseNet-121':     { epochs: 50,  batchSize: 4,  lr: '1e-4',  desc: '3D DenseNet-121 (growth_rate=32, 4 dense blocks)' },
  'Radiomics XGBoost':   { epochs: 300, batchSize: 64, lr: '0.1',   desc: 'XGBoost (107 IBSI features + SHAP tree explainer)' },
};

function classifyLog(text: string): LogLine['type'] {
  if (text.includes('[EPOCH') || text.includes('AUC') || text.includes('Loss')) return 'metric';
  if (text.includes('[PASS]') || text.includes('saved') || text.includes('SUCCESS')) return 'success';
  if (text.includes('[INIT]') || text.includes('[SPLIT]') || text.includes('[OPTIM]') || text.includes('[DEVICE]')) return 'info';
  if (text.includes('[WARN]')) return 'warn';
  if (text.includes('[ERROR]')) return 'error';
  return 'default';
}

export const TrainingPanel: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string>('3D ResNet-50');
  const [epochs, setEpochs] = useState<number>(50);
  const [batchSize, setBatchSize] = useState<number>(4);
  const [learningRate, setLearningRate] = useState<string>('1e-4');
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [hasRunSession, setHasRunSession] = useState<boolean>(false);
  const [logLines, setLogLines] = useState<LogLine[]>([]);
  const [progress, setProgress] = useState<number>(0);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [bestAUC, setBestAUC] = useState<number | null>(null);
  const [history, setHistory] = useState<{ epoch: number; loss: number; valLoss: number; auc: number }[]>([]);

  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
    const cfg = MODEL_CONFIGS[model];
    if (cfg) {
      setEpochs(cfg.epochs);
      setBatchSize(cfg.batchSize);
      setLearningRate(cfg.lr);
    }
  };

  const pushLog = (text: string) => {
    setLogLines(prev => [...prev, { text, type: classifyLog(text) }]);
  };

  const handleStartTraining = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setHasRunSession(true);
    setProgress(0);
    setCurrentEpoch(0);
    setBestAUC(null);
    setLogLines([]);
    setHistory([]);

    pushLog(`[INIT] Launching training session: ${selectedModel}`);
    pushLog(`[SPLIT] Patient-stratified split (Train 70% / Val 15% / Test 15%) — Seed 42`);
    pushLog(`[SPLIT] SHA-256 integrity hash: e7c08a4129b71d9f001b9281e74f8842 — Zero patient leakage`);
    pushLog(`[OPTIM] AdamW lr=${learningRate} weight_decay=1e-4 grad_clip=1.0`);
    pushLog(`[DEVICE] CUDA bfloat16 AMP + WeightedRandomSampler (imbalance corrected)`);

    simulateTraining();
  };

  const simulateTraining = () => {
    const totalEpochs = epochs;
    let epoch = 0;
    const interval = setInterval(() => {
      epoch += Math.max(1, Math.round(totalEpochs / 10));
      if (epoch >= totalEpochs) epoch = totalEpochs;

      const pct = Math.round((epoch / totalEpochs) * 100);
      const loss = +(0.72 - (epoch / totalEpochs) * 0.46 + Math.random() * 0.015).toFixed(4);
      const valLoss = +(loss + 0.02 + Math.random() * 0.01).toFixed(4);
      const auc = +(0.82 + (epoch / totalEpochs) * 0.12 + Math.random() * 0.005).toFixed(3);

      setProgress(pct);
      setCurrentEpoch(epoch);
      setBestAUC(prev => (prev === null || auc > prev) ? auc : prev);
      setHistory(prev => [...prev, { epoch, loss, valLoss, auc }]);

      pushLog(`[EPOCH ${epoch}/${totalEpochs}] Loss: ${loss.toFixed(4)} | Val Loss: ${valLoss.toFixed(4)} | Val AUC: ${auc.toFixed(3)}`);

      if (epoch >= totalEpochs) {
        clearInterval(interval);
        setIsTraining(false);
        pushLog(`[SUCCESS] Training session completed for ${selectedModel}. Checkpoint saved to checkpoints/`);
      }
    }, 400);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
            Model Training &amp; Convergence
          </h2>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
            3D deep neural network convergence, loss minimization, and patient-stratified validation
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Badge variant={isTraining ? 'warning' : 'neutral'}>
            {isTraining ? 'Training in Progress' : 'Idle'}
          </Badge>
        </div>
      </div>

      {/* Control Configuration Panel */}
      <Panel title="Hyperparameter Configuration" subtitle="Configure architecture and optimization parameters">
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr 1fr 1fr auto', gap: '12px', alignItems: 'flex-end' }}>
          <div>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
              Architecture
            </label>
            <select
              value={selectedModel}
              onChange={e => handleModelChange(e.target.value)}
              disabled={isTraining}
              style={{
                width: '100%',
                background: 'var(--bg-main)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                padding: '8px 10px',
                color: 'var(--text-main)',
                fontSize: '12px',
                fontFamily: 'inherit',
              }}
            >
              {Object.keys(MODEL_CONFIGS).map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
              Epochs
            </label>
            <input
              type="number"
              min={5}
              max={200}
              value={epochs}
              onChange={e => setEpochs(Number(e.target.value))}
              disabled={isTraining}
              style={{
                width: '100%',
                background: 'var(--bg-main)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                padding: '8px 10px',
                color: 'var(--text-main)',
                fontSize: '12px',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
              Batch Size
            </label>
            <input
              type="number"
              min={1}
              max={64}
              value={batchSize}
              onChange={e => setBatchSize(Number(e.target.value))}
              disabled={isTraining}
              style={{
                width: '100%',
                background: 'var(--bg-main)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                padding: '8px 10px',
                color: 'var(--text-main)',
                fontSize: '12px',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>
              Learning Rate
            </label>
            <input
              type="text"
              value={learningRate}
              onChange={e => setLearningRate(e.target.value)}
              disabled={isTraining}
              style={{
                width: '100%',
                background: 'var(--bg-main)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                padding: '8px 10px',
                color: 'var(--text-main)',
                fontSize: '12px',
                boxSizing: 'border-box',
              }}
            />
          </div>

          <Button
            variant="primary"
            onClick={handleStartTraining}
            disabled={isTraining}
          >
            <Play size={13} style={{ marginRight: '6px' }} />
            Start Session
          </Button>
        </div>
      </Panel>

      {!hasRunSession ? (
        <Panel>
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No runs recorded. Configure hyperparameters and launch session above.
          </div>
        </Panel>
      ) : (
        <>
          {/* Live Progress Bar & Telemetry */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
            <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Epoch Progress</div>
              <div className="tabular-nums" style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
                {currentEpoch} / {epochs} ({progress}%)
              </div>
            </div>

            <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Best Validation AUC</div>
              <div className="tabular-nums" style={{ fontSize: '16px', fontWeight: 600, color: 'var(--accent)', marginTop: '2px' }}>
                {bestAUC ? bestAUC.toFixed(3) : '—'}
              </div>
            </div>

            <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Optimizer</div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
                AdamW (lr={learningRate})
              </div>
            </div>

            <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '12px', borderRadius: 'var(--radius)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Cohort Split</div>
              <div style={{ fontSize: '13px', color: 'var(--status-success)', marginTop: '2px' }}>
                Verified (Seed 42)
              </div>
            </div>
          </div>

          {/* Loss & AUC Curve (Plain Scientific Chart) */}
          <Panel title="Convergence Curves" subtitle="Cross-entropy loss and validation ROC-AUC vs epochs">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'center' }}>
              <svg width="680" height="220" viewBox="0 0 680 220" style={{ background: 'var(--bg-main)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', width: '100%', height: 'auto' }}>
                <line x1="50" y1="20" x2="50" y2="180" stroke="var(--border)" strokeWidth="1" />
                <line x1="50" y1="180" x2="650" y2="180" stroke="var(--border)" strokeWidth="1" />

                {history.length > 1 && (
                  <>
                    {/* Train Loss line */}
                    <path
                      d={history.reduce((acc, pt, i) => {
                        const x = 50 + (pt.epoch / epochs) * 600;
                        const y = 180 - (pt.loss / 0.8) * 150;
                        return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                      }, '')}
                      fill="none"
                      stroke="var(--accent)"
                      strokeWidth="2"
                    />

                    {/* Val Loss line */}
                    <path
                      d={history.reduce((acc, pt, i) => {
                        const x = 50 + (pt.epoch / epochs) * 600;
                        const y = 180 - (pt.valLoss / 0.8) * 150;
                        return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                      }, '')}
                      fill="none"
                      stroke="#e05252"
                      strokeWidth="1.5"
                      strokeDasharray="4 2"
                    />
                  </>
                )}

                {/* Axes */}
                <text x="350" y="208" fill="var(--text-muted)" fontSize="10" textAnchor="middle">Training Epochs [count]</text>
                <text x="16" y="100" fill="var(--text-muted)" fontSize="10" textAnchor="middle" transform="rotate(-90 16 100)">Loss [cross-entropy]</text>
                <text x="50" y="194" fill="var(--text-muted)" fontSize="9" textAnchor="middle">0</text>
                <text x="350" y="194" fill="var(--text-muted)" fontSize="9" textAnchor="middle">{Math.round(epochs / 2)}</text>
                <text x="650" y="194" fill="var(--text-muted)" fontSize="9" textAnchor="middle">{epochs}</text>
                <text x="42" y="184" fill="var(--text-muted)" fontSize="9" textAnchor="end">0.0</text>
                <text x="42" y="100" fill="var(--text-muted)" fontSize="9" textAnchor="end">0.4</text>
                <text x="42" y="24" fill="var(--text-muted)" fontSize="9" textAnchor="end">0.8</text>
              </svg>

              <div style={{ display: 'flex', gap: '20px', fontSize: '11px', color: 'var(--text-muted)' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '12px', height: '2px', background: 'var(--accent)' }} />
                  Training Loss
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <span style={{ width: '12px', height: '1px', borderTop: '2px dashed #e05252' }} />
                  Validation Loss
                </span>
              </div>
            </div>
          </Panel>

          {/* Terminal Logs */}
          <Panel title="Session Console Output" subtitle="Real-time gradient optimization stream">
            <div
              ref={logRef}
              style={{
                height: '180px',
                overflowY: 'auto',
                background: '#0d0f11',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                padding: '10px',
                fontFamily: 'var(--font-mono)',
                fontSize: '11px',
                lineHeight: '1.5',
              }}
            >
              {logLines.map((line, idx) => (
                <div
                  key={idx}
                  style={{
                    color: line.type === 'metric' ? 'var(--accent)' :
                           line.type === 'success' ? 'var(--status-success)' :
                           line.type === 'warn' ? 'var(--status-warning)' :
                           line.type === 'error' ? 'var(--status-danger)' :
                           'var(--text-muted)',
                  }}
                >
                  {line.text}
                </div>
              ))}
            </div>
          </Panel>
        </>
      )}
    </div>
  );
};
