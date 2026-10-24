import React, { useState, useEffect } from 'react';
import { 
  Database, Eye, Activity, Cpu, Sun, Moon,
  FileText, Info, Layers, GitBranch, Sparkles, Cloud,
  CheckCircle2, AlertTriangle, ShieldCheck
} from 'lucide-react';

import { ThemeProvider, useTheme } from './context/ThemeContext';

// 4 Primary Clinical Screens
import { WorklistScreen } from './clinical/WorklistScreen';
import { StudyViewerScreen, FindingItem } from './clinical/StudyViewerScreen';
import { ReportDraftScreen } from './clinical/ReportDraftScreen';
import { AboutModelScreen } from './clinical/AboutModelScreen';

// Research Mode Panels (Preserved and gated behind VITE_RESEARCH_MODE === 'true')
import { WorkstationPanel } from './panels/WorkstationPanel';
import { CaseLogPanel } from './panels/CaseLogPanel';
import { CTViewerPanel } from './panels/CTViewerPanel';
import { TrainingPanel } from './panels/TrainingPanel';
import { VerdictPanel } from './panels/VerdictPanel';
import { ExperimentsPanel } from './panels/ExperimentsPanel';
import { RadiomicsPanel } from './panels/RadiomicsPanel';
import { AIAnalysisPanel } from './panels/AIAnalysisPanel';
import { PuterAnalysisPanel } from './panels/PuterAnalysisPanel';

type ClinicalScreenId = 'worklist' | 'viewer' | 'report' | 'about';
type ResearchPanelId = 'workstation' | 'cases' | 'viewer-legacy' | 'training' | 'candidates' | 'experiments' | 'radiomics' | 'ai-analysis' | 'puter';

const AppContent: React.FC = () => {
  const [activeScreen, setActiveScreen] = useState<ClinicalScreenId>('worklist');
  const [researchModeActive, setResearchModeActive] = useState<boolean>(false);
  const [activeResearchPanel, setActiveResearchPanel] = useState<ResearchPanelId>('training');

  const [currentStudyId, setCurrentStudyId] = useState<string>('CASE-DEMO-LIDC-0087');
  const [studyFindings, setStudyFindings] = useState<FindingItem[]>([]);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const { theme, toggleTheme } = useTheme();

  // Environment Flags
  const showResearchMode = (import.meta as any).env?.VITE_RESEARCH_MODE === 'true';
  const showCloudAI = (import.meta as any).env?.VITE_EXPERIMENTAL_CLOUD_AI === 'true';

  // Poll API health
  useEffect(() => {
    let timerId: any = null;
    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
          try { controller.abort(); } catch {}
        }, 2500);
        const r = await fetch('/api/v2/health', { signal: controller.signal });
        clearTimeout(timeoutId);
        setApiOnline(r.ok);
      } catch {
        setApiOnline(false);
      }
    };
    checkHealth();
    timerId = setInterval(checkHealth, 20_000);
    return () => {
      if (timerId) clearInterval(timerId);
    };
  }, []);

  // Keyboard navigation between 4 clinical screens (1: Worklist, 2: Viewer, 3: Report, 4: About)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) {
        return;
      }
      if (e.key === '1') {
        setActiveScreen('worklist');
        setResearchModeActive(false);
      } else if (e.key === '2') {
        setActiveScreen('viewer');
        setResearchModeActive(false);
      } else if (e.key === '3') {
        setActiveScreen('report');
        setResearchModeActive(false);
      } else if (e.key === '4') {
        setActiveScreen('about');
        setResearchModeActive(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const clinicalNavItems: { id: ClinicalScreenId; label: string; icon: React.FC<any> }[] = [
    { id: 'worklist', label: '01 · Worklist', icon: Database },
    { id: 'viewer',   label: '02 · Study Viewer', icon: Eye },
    { id: 'report',   label: '03 · Report Draft', icon: FileText },
    { id: 'about',    label: '04 · About This Model', icon: Info },
  ];

  const researchNavItems: { id: ResearchPanelId; label: string; icon: React.FC<any> }[] = [
    { id: 'training',       label: 'Training Convergence', icon: Cpu },
    { id: 'experiments',    label: 'MLflow Runs', icon: GitBranch },
    { id: 'radiomics',      label: 'IBSI Radiomics', icon: Layers },
    { id: 'workstation',    label: 'Legacy Workstation', icon: Activity },
    { id: 'cases',          label: 'Case Audit Database', icon: Database },
    { id: 'candidates',     label: 'Candidate Output Explorer', icon: ShieldCheck },
    ...(showCloudAI ? [
      { id: 'ai-analysis' as ResearchPanelId, label: 'Frontier AI (Proxy)', icon: Sparkles },
      { id: 'puter' as ResearchPanelId,       label: 'Puter Cloud AI', icon: Cloud },
    ] : []),
  ];

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'var(--bg-main)',
      color: 'var(--text-main)',
      fontSize: '14px'
    }}>

      {/* Top Header Bar — PACS Reading Room Standard */}
      <header style={{
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '0 20px',
        height: '52px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}>
        {/* Brand & Research Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            fontSize: '14px',
            fontWeight: 700,
            letterSpacing: '0.02em',
            color: 'var(--text-main)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <span>RADIOLOGY WORKSTATION</span>
            <span style={{
              fontSize: '10px',
              padding: '2px 6px',
              borderRadius: '3px',
              background: 'rgba(217, 119, 6, 0.12)',
              color: 'var(--accent-amber)',
              border: '1px solid rgba(217, 119, 6, 0.3)',
              fontWeight: 600,
              letterSpacing: '0.04em'
            }}>
              RESEARCH USE ONLY
            </span>
          </div>
        </div>

        {/* Center / Right Metadata & Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          
          {/* Active study badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '4px 8px',
            borderRadius: '4px',
            backgroundColor: 'var(--bg-deepest)',
            border: '1px solid var(--border-subtle)',
            fontSize: '11px',
            color: 'var(--text-body)',
            fontFamily: 'ui-monospace, monospace'
          }}>
            <span style={{ color: 'var(--text-muted)' }}>Study:</span>
            <strong style={{ color: 'var(--primary)' }}>{currentStudyId}</strong>
          </div>

          {/* Reader Identity */}
          <div style={{
            fontSize: '11px',
            color: 'var(--text-muted)',
            padding: '4px 8px',
            borderRadius: '4px',
            background: 'var(--bg-deepest)',
            border: '1px solid var(--border-subtle)'
          }}>
            Reader: <strong style={{ color: 'var(--text-body)' }}>Dr. A. Vance (Rad-402)</strong>
          </div>

          {/* Engine Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '11px',
            color: 'var(--text-muted)'
          }}>
            <span style={{
              width: '6px', height: '6px', borderRadius: '50%',
              backgroundColor: apiOnline === true ? 'var(--accent-emerald)' : 'var(--accent-amber)',
            }} />
            <span>{apiOnline === true ? 'Online' : 'Standby'}</span>
          </div>

          {/* Research Mode Toggle (Only if VITE_RESEARCH_MODE === 'true') */}
          {showResearchMode && (
            <button
              onClick={() => setResearchModeActive(!researchModeActive)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '4px 10px',
                borderRadius: '4px',
                border: '1px solid',
                borderColor: researchModeActive ? 'var(--primary)' : 'var(--border-subtle)',
                background: researchModeActive ? 'var(--primary)' : 'var(--bg-deepest)',
                color: researchModeActive ? '#ffffff' : 'var(--text-body)',
                fontSize: '11px',
                fontWeight: 600,
                cursor: 'pointer'
              }}
            >
              <Cpu size={12} />
              {researchModeActive ? 'Exit Research Mode' : 'Research Mode'}
            </button>
          )}

          {/* Dark / Light Mode Toggle */}
          <button
            onClick={toggleTheme}
            className="theme-toggle-btn"
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              padding: '4px 8px',
              borderRadius: '4px',
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-deepest)',
              color: 'var(--text-body)',
              fontSize: '11px',
              cursor: 'pointer'
            }}
          >
            {theme === 'dark' ? (
              <>
                <Sun size={12} color="var(--accent-amber)" />
                <span>Light</span>
              </>
            ) : (
              <>
                <Moon size={12} color="var(--primary)" />
                <span>Dark</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* Navigation Tabs Bar */}
      <nav style={{
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '0 20px',
        display: 'flex',
        gap: '4px',
        overflowX: 'auto',
      }}>
        {!researchModeActive ? (
          // Default 4 Clinical Screens
          clinicalNavItems.map(item => {
            const Icon = item.icon;
            const isActive = activeScreen === item.id;
            return (
              <button
                key={item.id}
                id={`nav-${item.id}`}
                onClick={() => setActiveScreen(item.id)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  borderBottom: `2px solid ${isActive ? 'var(--primary)' : 'transparent'}`,
                  color: isActive ? 'var(--primary)' : 'var(--text-muted)',
                  padding: '10px 14px',
                  fontSize: '13px',
                  fontWeight: isActive ? 600 : 500,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  whiteSpace: 'nowrap'
                }}
              >
                <Icon size={14} color={isActive ? 'var(--primary)' : 'var(--text-muted)'} />
                <span>{item.label}</span>
              </button>
            );
          })
        ) : (
          // Research Subsystems Tabs (Only active when VITE_RESEARCH_MODE === 'true' & toggled)
          researchNavItems.map(item => {
            const Icon = item.icon;
            const isActive = activeResearchPanel === item.id;
            return (
              <button
                key={item.id}
                id={`nav-research-${item.id}`}
                onClick={() => setActiveResearchPanel(item.id)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  borderBottom: `2px solid ${isActive ? 'var(--primary)' : 'transparent'}`,
                  color: isActive ? 'var(--primary)' : 'var(--text-muted)',
                  padding: '10px 14px',
                  fontSize: '13px',
                  fontWeight: isActive ? 600 : 500,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  whiteSpace: 'nowrap'
                }}
              >
                <Icon size={14} color={isActive ? 'var(--primary)' : 'var(--text-muted)'} />
                <span>{item.label}</span>
              </button>
            );
          })
        )}
      </nav>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '16px 20px 48px 20px' }}>
        
        {!researchModeActive ? (
          // Clinical Mode: exactly 4 screens
          <>
            {activeScreen === 'worklist' && (
              <WorklistScreen
                onSelectStudy={(selectedId) => {
                  setCurrentStudyId(selectedId);
                  setActiveScreen('viewer');
                }}
              />
            )}

            {activeScreen === 'viewer' && (
              <StudyViewerScreen
                studyId={currentStudyId}
                onProceedToReport={(studyId, findings) => {
                  setCurrentStudyId(studyId);
                  setStudyFindings(findings);
                  setActiveScreen('report');
                }}
              />
            )}

            {activeScreen === 'report' && (
              <ReportDraftScreen
                studyId={currentStudyId}
                findings={studyFindings}
                onBackToViewer={() => setActiveScreen('viewer')}
              />
            )}

            {activeScreen === 'about' && (
              <AboutModelScreen />
            )}
          </>
        ) : (
          // Research Mode: preserved research panels
          <>
            {activeResearchPanel === 'training' && <TrainingPanel />}
            {activeResearchPanel === 'experiments' && <ExperimentsPanel />}
            {activeResearchPanel === 'radiomics' && <RadiomicsPanel studyId={currentStudyId} noduleId="NODULE-01" />}
            {activeResearchPanel === 'workstation' && <WorkstationPanel studyId={currentStudyId} navigate={() => {}} />}
            {activeResearchPanel === 'cases' && <CaseLogPanel onSelectCase={(id) => { setCurrentStudyId(id); setActiveScreen('viewer'); setResearchModeActive(false); }} />}
            {activeResearchPanel === 'candidates' && <VerdictPanel noduleId="NODULE-01" />}
            {activeResearchPanel === 'ai-analysis' && showCloudAI && <AIAnalysisPanel studyId={currentStudyId} noduleId="NODULE-01" />}
            {activeResearchPanel === 'puter' && showCloudAI && <PuterAnalysisPanel />}
          </>
        )}

      </main>

      {/* Persistent Single-Line Regulatory Footer Notice */}
      <footer style={{
        borderTop: '1px solid rgba(220, 38, 38, 0.3)',
        background: 'var(--bg-surface)',
        padding: '8px 20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '11px',
        color: 'var(--text-muted)',
        position: 'fixed',
        bottom: 0,
        width: '100%',
        boxSizing: 'border-box',
        zIndex: 40,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            background: 'rgba(220, 38, 38, 0.12)',
            color: 'var(--accent-red)',
            padding: '2px 6px',
            borderRadius: '3px',
            fontWeight: 700,
            fontSize: '10px',
            letterSpacing: '0.04em'
          }}>
            RESEARCH USE ONLY (RUO)
          </span>
          <span style={{ color: 'var(--accent-red)', fontWeight: 500 }}>
            NOT A MEDICAL DEVICE · For research benchmarking only. Not for clinical decisions.
          </span>
        </div>
        <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: '10px', color: 'var(--text-faint)' }}>
          Thoracic CT Volumetric Analysis v2.0
        </span>
      </footer>

    </div>
  );
};

export const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
};

export default App;
