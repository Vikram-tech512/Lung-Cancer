import React, { useState, useEffect } from 'react';
import {
  Upload, Eye, Activity, BrainCircuit, Ruler, FileText,
  Database, Award, GitBranch, Cpu, Box, HardDrive, Shield,
  LucideIcon
} from 'lucide-react';

import { IntakePanel } from './panels/IntakePanel';
import { CTViewerPanel } from './panels/CTViewerPanel';
import { WorkstationPanel } from './panels/WorkstationPanel';
import { AIAnalysisPanel } from './panels/AIAnalysisPanel';
import { MorphometryPanel } from './panels/MorphometryPanel';
import { VerdictPanel } from './panels/VerdictPanel';
import { CaseLogPanel } from './panels/CaseLogPanel';
import { BenchmarkPanel } from './panels/BenchmarkPanel';
import { ExperimentsPanel } from './panels/ExperimentsPanel';
import { TrainingPanel } from './panels/TrainingPanel';
import { RegistryPanel } from './panels/RegistryPanel';
import { API_BASE } from './services/api';

export type PanelId =
  | 'intake'
  | 'viewer'
  | 'workstation'
  | 'verdict'
  | 'morphometry'
  | 'ai-analysis'
  | 'caselog'
  | 'benchmark'
  | 'experiments'
  | 'training'
  | 'registry';

interface NavItem {
  id: PanelId;
  label: string;
  category: 'clinical' | 'research';
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  // Clinical Operations
  { id: 'viewer',       label: 'Diagnostic Workstation',  category: 'clinical', icon: Eye },
  { id: 'workstation',  label: 'Case Archive',            category: 'clinical', icon: Activity },
  { id: 'intake',       label: 'DICOM Intake & QC',       category: 'clinical', icon: Upload },
  { id: 'verdict',      label: 'Radiology Report',        category: 'clinical', icon: FileText },
  { id: 'morphometry',  label: '3D Morphometry',          category: 'clinical', icon: Ruler },
  { id: 'ai-analysis',  label: 'Model Inference',         category: 'clinical', icon: BrainCircuit },
  { id: 'caselog',      label: 'Case Audit Log',          category: 'clinical', icon: Database },

  // Model & Research Evaluation
  { id: 'benchmark',    label: 'Benchmark Cohort',        category: 'research', icon: Award },
  { id: 'experiments',  label: 'Experiment Runs',         category: 'research', icon: GitBranch },
  { id: 'training',     label: 'Training & Loss',         category: 'research', icon: Cpu },
  { id: 'registry',     label: 'Model Registry',          category: 'research', icon: Box },
];

export const App: React.FC = () => {
  const [activePanel, setActivePanel] = useState<PanelId>('viewer');
  const [currentStudyId, setCurrentStudyId] = useState<string>('');
  const [currentNoduleId, setCurrentNoduleId] = useState<string>('Not available');
  const [patientId, setPatientId] = useState<string>('Not available');
  const [studyDate, setStudyDate] = useState<string>('Not available');
  const [isSyntheticStudy, setIsSyntheticStudy] = useState<boolean>(false);
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const [modelVersion, setModelVersion] = useState<string>('Not available');

  // Poll backend health status
  useEffect(() => {
    let timerId: any = null;
    const checkReady = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
          try { controller.abort(); } catch {}
        }, 2000);
        // Try /health/ready then fallback to /api/v2/health
        const r = await fetch('/health/ready', { signal: controller.signal })
          .catch(() => fetch('/api/v2/health', { signal: controller.signal }));
        clearTimeout(timeoutId);
        setBackendReady(r ? r.ok : false);
      } catch {
        setBackendReady(false);
      }
    };

    checkReady();
    timerId = setInterval(checkReady, 15000);
    // Fetch available studies on initial load
    fetch(`${API_BASE}/cases`)
      .then(r => r.ok ? r.json() : { cases: [] })
      .then(d => {
        if (d && Array.isArray(d.cases) && d.cases.length > 0) {
          setCurrentStudyId(d.cases[0].study_id);
          setIsSyntheticStudy(!!d.cases[0].is_synthetic);
        }
      })
      .catch(() => {});
    return () => {
      if (timerId) clearInterval(timerId);
    };
  }, []);

  const handleStudyLoaded = (studyId: string) => {
    setCurrentStudyId(studyId);
    setIsSyntheticStudy(true);
    setPatientId('Not available');
    setStudyDate('Not available');
    setActivePanel('viewer');
  };

  const handleInspectNodule = (noduleId: string) => {
    setCurrentNoduleId(noduleId);
    setActivePanel('verdict');
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        background: 'var(--bg-main)',
        color: 'var(--text-main)',
        fontSize: '13px',
      }}
    >
      {/* Persistent Regulatory Research Notice Strip */}
      <div
        style={{
          background: '#241a08',
          borderBottom: '1px solid #573e0e',
          color: '#fbbf24',
          fontSize: '11px',
          fontWeight: 600,
          letterSpacing: '0.04em',
          textAlign: 'center',
          padding: '3px 12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          zIndex: 100,
          flexShrink: 0,
        }}
      >
        <span>RESEARCH USE ONLY — Investigational Prototype — Not Cleared for Standalone Diagnostic Decisions</span>
        {isSyntheticStudy && (
          <span
            style={{
              background: '#3f2d08',
              border: '1px solid #d97706',
              color: '#fef3c7',
              padding: '1px 6px',
              borderRadius: '2px',
              fontSize: '10px',
              fontWeight: 700,
            }}
          >
            SYNTHETIC DEMO DATA
          </span>
        )}
      </div>

      {/* Top Bar: Case ID, Patient ID, Study Date, Modality */}
      <header
        style={{
          height: '42px',
          background: 'var(--bg-panel)',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 16px',
          flexShrink: 0,
          zIndex: 20,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, letterSpacing: '0.02em' }}>
            <span style={{ color: 'var(--accent)', fontWeight: 700 }}>RESEARCH CT</span>
            <span style={{ color: 'var(--text-muted)' }}>/</span>
            <span>Thoracic CT Workstation</span>
          </div>

          <div style={{ height: '14px', width: '1px', background: 'var(--border)' }} />

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px' }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Patient ID: </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{patientId}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Study: </span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{currentStudyId}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Date: </span>
              <span className="tabular-nums">{studyDate}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Modality: </span>
              <span>Not available</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                background: backendReady === true ? 'var(--status-success)' : backendReady === false ? 'var(--status-danger)' : 'var(--status-warning)',
              }}
            />
            <span style={{ color: 'var(--text-muted)' }}>
              Backend: {backendReady === true ? 'Connected' : backendReady === false ? 'Offline' : 'Connecting'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Body: Left Vertical Navigation + Workspace Viewport */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Vertical Navigation */}
        <nav
          style={{
            width: '210px',
            background: 'var(--bg-panel)',
            borderRight: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            padding: '10px 8px',
            overflowY: 'auto',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            {/* Clinical Section */}
            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', padding: '0 8px', marginBottom: '6px' }}>
                Clinical Diagnostics
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {NAV_ITEMS.filter(item => item.category === 'clinical').map(item => {
                  const Icon = item.icon;
                  const isActive = activePanel === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setActivePanel(item.id)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '7px 10px',
                        borderRadius: 'var(--radius)',
                        border: 'none',
                        background: isActive ? 'var(--bg-main)' : 'transparent',
                        color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                        fontSize: '12px',
                        fontWeight: isActive ? 600 : 400,
                        cursor: 'pointer',
                        textAlign: 'left',
                        outline: 'none',
                        boxShadow: isActive ? 'inset 2px 0 0 var(--accent)' : 'none',
                      }}
                    >
                      <Icon size={14} color={isActive ? 'var(--accent)' : 'var(--text-muted)'} />
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Research & Evaluation Section */}
            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', padding: '0 8px', marginBottom: '6px' }}>
                Research &amp; Benchmark
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {NAV_ITEMS.filter(item => item.category === 'research').map(item => {
                  const Icon = item.icon;
                  const isActive = activePanel === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setActivePanel(item.id)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        padding: '7px 10px',
                        borderRadius: 'var(--radius)',
                        border: 'none',
                        background: isActive ? 'var(--bg-main)' : 'transparent',
                        color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                        fontSize: '12px',
                        fontWeight: isActive ? 600 : 400,
                        cursor: 'pointer',
                        textAlign: 'left',
                        outline: 'none',
                        boxShadow: isActive ? 'inset 2px 0 0 var(--accent)' : 'none',
                      }}
                    >
                      <Icon size={14} color={isActive ? 'var(--accent)' : 'var(--text-muted)'} />
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Quick Target Indicator */}
          <div
            style={{
              padding: '8px 10px',
              background: 'var(--bg-main)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              fontSize: '11px',
            }}
          >
            <div style={{ color: 'var(--text-muted)' }}>Target Candidate</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-main)', marginTop: '2px' }}>
              {currentNoduleId}
            </div>
          </div>
        </nav>

        {/* Primary Workspace Scroll Area */}
        <main
          style={{
            flex: 1,
            overflowY: activePanel === 'viewer' ? 'hidden' : 'auto',
            padding: activePanel === 'viewer' ? 0 : '16px 20px',
            background: 'var(--bg-main)',
          }}
        >
          {activePanel === 'viewer' && (
            <CTViewerPanel
              studyId={currentStudyId}
              onInspectNodule={handleInspectNodule}
              onSelectStudy={(id) => setCurrentStudyId(id)}
            />
          )}
          {activePanel === 'workstation' && <WorkstationPanel onSelectStudy={handleStudyLoaded} />}
          {activePanel === 'intake' && <IntakePanel onStudyLoaded={handleStudyLoaded} />}
          {activePanel === 'verdict' && <VerdictPanel noduleId={currentNoduleId} studyId={currentStudyId} />}
          {activePanel === 'morphometry' && <MorphometryPanel noduleId={currentNoduleId} studyId={currentStudyId} />}
          {activePanel === 'ai-analysis' && (
            <AIAnalysisPanel
              studyId={currentStudyId}
              noduleId={currentNoduleId}
            />
          )}
          {activePanel === 'caselog' && (
            <CaseLogPanel
              onSelectCase={(caseId) => {
                setCurrentStudyId(caseId);
                setIsSyntheticStudy(true);
                setPatientId('Not available');
                setStudyDate('Not available');
                setActivePanel('viewer');
              }}
            />
          )}
          {activePanel === 'benchmark' && <BenchmarkPanel />}
          {activePanel === 'experiments' && <ExperimentsPanel />}
          {activePanel === 'training' && <TrainingPanel />}
          {activePanel === 'registry' && <RegistryPanel />}
        </main>
      </div>

      {/* Bottom Status Bar */}
      <footer
        style={{
          height: '26px',
          background: 'var(--bg-panel)',
          borderTop: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 16px',
          fontSize: '11px',
          color: 'var(--text-muted)',
          flexShrink: 0,
          zIndex: 20,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <span>
            Model Pipeline: <strong style={{ color: 'var(--text-main)' }}>{modelVersion}</strong>
          </span>
          <span style={{ color: 'var(--border)' }}>|</span>
          <span>
            DICOM Standard: <strong style={{ color: 'var(--text-main)' }}>Part 10 (PS 3.10)</strong>
          </span>
          <span style={{ color: 'var(--border)' }}>|</span>
          <span>
            Radiomics: <strong style={{ color: 'var(--text-main)' }}>Not available</strong>
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: 'var(--status-warning)' }}>
            Research use only, not for clinical decisions
          </span>
        </div>
      </footer>
    </div>
  );
};

export default App;
