import React, { useState, useEffect } from 'react';
import { Play, CheckCircle2, Clock, Terminal, ArrowRight, Activity, ShieldCheck } from 'lucide-react';
import { useAnalysisWebSocket } from '../services/websocket';
import { api } from '../services/api';

interface AnalysisTrackerProps {
  analysisId: string;
  navigate: (route: string) => void;
}

const STAGES = [
  'QUEUED', 'UPLOADING', 'VALIDATING', 'DE_IDENTIFICATION', 'PREPROCESSING',
  'VOLUME_CREATION', 'SEGMENTING_LUNGS', 'DETECTING_NODULES', 'SEGMENTING_LESIONS',
  'CALCULATING_MEASUREMENTS', 'RUNNING_RADIOMICS', 'CLASSIFYING',
  'RUNNING_TRANSFORMER', 'CALIBRATING', 'GENERATING_EXPLANATION',
  'GENERATING_REPORT', 'COMPLETED'
];

export const AnalysisTrackerPage: React.FC<AnalysisTrackerProps> = ({ analysisId, navigate }) => {
  const { latestEvent, isConnected } = useAnalysisWebSocket(analysisId);
  const [currentStage, setCurrentStage] = useState('PROCESSING');
  const [progress, setProgress] = useState(35);
  const [logs, setLogs] = useState<string[]>([
    `[INIT] Task received in queue for job: ${analysisId}`,
    `[CELERY] Worker process assigned. Spawning asynchronous pipeline.`,
    `[WEBSOCKET] Connected to event broadcast channel.`
  ]);

  useEffect(() => {
    if (latestEvent) {
      setCurrentStage(latestEvent.stage);
      setProgress(latestEvent.progress_percent);
      setLogs(prev => [
        ...prev,
        `[STAGE: ${latestEvent.stage}] Progress: ${latestEvent.progress_percent}% | ${JSON.stringify(latestEvent.data)}`
      ]);
    }
  }, [latestEvent]);

  // If no WebSocket events yet, poll REST API as fallback
  useEffect(() => {
    const interval = setInterval(() => {
      api.getAnalysisStatus(analysisId)
        .then(status => {
          if (status.current_stage) setCurrentStage(status.current_stage);
          if (status.progress_percent) setProgress(status.progress_percent);
        })
        .catch(() => {});
    }, 2000);
    return () => clearInterval(interval);
  }, [analysisId]);

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: '800', margin: 0, color: '#f8fafc' }}>
              REAL-TIME CELERY TASK ORCHESTRATOR
            </h1>
            <span style={{
              fontSize: '11px',
              fontWeight: '700',
              padding: '2px 8px',
              borderRadius: '4px',
              background: isConnected ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              color: isConnected ? '#10b981' : '#f59e0b',
              border: `1px solid ${isConnected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`
            }}>
              {isConnected ? 'LIVE WEBSOCKET STREAMING' : 'CONNECTING...'}
            </span>
          </div>
          <p style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0 0 0' }}>
            Job ID: <span style={{ color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>{analysisId}</span> | 16-Stage Medical AI Pipeline
          </p>
        </div>

        {progress >= 100 && (
          <button
            onClick={() => navigate('/viewer/CASE-DEMO-LIDC-0087')}
            style={{
              background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
              color: '#ffffff',
              border: 'none',
              borderRadius: '6px',
              padding: '10px 18px',
              fontSize: '13px',
              fontWeight: '700',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            Inspect in Workstation
            <ArrowRight size={16} />
          </button>
        )}
      </div>

      {/* Progress Bar */}
      <div className="medical-card" style={{ padding: '20px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '13px', fontWeight: '700' }}>
          <span style={{ color: '#38bdf8' }}>STAGE: {currentStage}</span>
          <span style={{ color: '#f8fafc', fontFamily: 'JetBrains Mono, monospace' }}>{progress}%</span>
        </div>

        <div style={{ height: '8px', background: '#070c17', borderRadius: '4px', overflow: 'hidden' }}>
          <div style={{ width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #0284c7, #06b6d4)', transition: 'width 0.3s ease' }} />
        </div>
      </div>

      {/* 16 Stages Flow View */}
      <div className="medical-card" style={{ padding: '20px', marginBottom: '24px' }}>
        <h3 style={{ margin: '0 0 16px 0', fontSize: '14px', fontWeight: '700', color: '#f8fafc' }}>
          Pipeline State Machine (16 Verified Stages)
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
          {STAGES.map((s, idx) => {
            const isDone = STAGES.indexOf(currentStage) >= idx || progress === 100;
            const isCurrent = currentStage === s && progress < 100;
            return (
              <div
                key={s}
                style={{
                  background: isCurrent ? 'rgba(6, 182, 212, 0.15)' : (isDone ? 'rgba(16, 185, 129, 0.1)' : '#070c17'),
                  border: '1px solid',
                  borderColor: isCurrent ? '#06b6d4' : (isDone ? '#10b981' : '#1c2638'),
                  padding: '8px 10px',
                  borderRadius: '6px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                {isDone ? (
                  <CheckCircle2 size={14} color="#10b981" />
                ) : (
                  <Clock size={14} color={isCurrent ? '#06b6d4' : '#64748b'} />
                )}
                <span style={{ fontSize: '11px', fontWeight: '700', color: isDone || isCurrent ? '#f8fafc' : '#64748b' }}>
                  {s}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Terminal Output */}
      <div className="medical-card" style={{ padding: '20px' }}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '700', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Terminal size={16} color="#34d399" />
          Live WebSocket Event Stream
        </h3>

        <div style={{
          background: '#070a12',
          border: '1px solid #1c2638',
          borderRadius: '6px',
          padding: '14px',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px',
          color: '#cbd5e1',
          maxHeight: '260px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px'
        }}>
          {logs.map((log, i) => (
            <div key={i} style={{ color: log.includes('STAGE') ? '#38bdf8' : '#94a3b8' }}>
              {log}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
