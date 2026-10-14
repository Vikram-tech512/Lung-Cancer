import React, { useState, useEffect } from 'react';
import { 
  Upload, Database, Eye, Cpu,
  Ruler, Activity, ShieldAlert, WifiOff,
  GitBranch
} from 'lucide-react';

import { WorkstationPanel } from './panels/WorkstationPanel';
import { IntakePanel }      from './panels/IntakePanel';
import { CaseLogPanel }     from './panels/CaseLogPanel';
import { CTViewerPanel }    from './panels/CTViewerPanel';
import { TrainingPanel }    from './panels/TrainingPanel';
import { MorphometryPanel } from './panels/MorphometryPanel';
import { ExperimentsPanel } from './panels/ExperimentsPanel';
import { PuterAnalysisPanel } from './panels/PuterAnalysisPanel';
import { API_BASE } from './services/api';

type PanelId = 'workstation' | 'intake' | 'cases' | 'viewer' | 'training' | 'morphometry' | 'experiments' | 'puter-ai';

export const App: React.FC = () => {
  const [activePanel, setActivePanel]       = useState<PanelId>('workstation');
  const [currentStudyId, setCurrentStudyId] = useState<string>('CASE-DEMO-LIDC-0087');
  const [currentNoduleId, setCurrentNoduleId] = useState<string>('NODULE-01');
  const [apiOnline, setApiOnline]           = useState<boolean | null>(null);

  // Poll API health every 15 seconds
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2500) });
        setApiOnline(r.ok);
      } catch {
        setApiOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15_000);
    return () => clearInterval(interval);
  }, []);

  const navItems: { id: PanelId; label: string; icon: React.FC<any> }[] = [
    { id: 'workstation', label: '00 · WORKSTATION',   icon: Activity  },
    { id: 'intake',      label: '01 · INTAKE',        icon: Upload    },
    { id: 'cases',       label: '02 · CASE LOG',      icon: Database  },
    { id: 'viewer',      label: '03 · CT VIEWER',     icon: Eye       },
    { id: 'training',    label: '04 · TRAINING',      icon: Cpu       },
    { id: 'morphometry', label: '05 · MORPHOMETRY',   icon: Ruler     },
    { id: 'experiments', label: '06 · EXPERIMENTS',   icon: GitBranch },
    { id: 'puter-ai',    label: '07 · AI ANALYSIS',   icon: Cpu       },
  ];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#070a11' }}>

      {/* Guard 5 Mandatory Clinical Advisory Banner */}
      <div style={{
        background: 'linear-gradient(90deg, #7f1d1d, #881337)',
        borderBottom: '1px solid rgba(190, 18, 60, 0.6)',
        color: '#ffe4e6',
        padding: '5px 24px',
        fontSize: '11px',
        fontWeight: '700',
        letterSpacing: '0.04em',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        textTransform: 'uppercase'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldAlert size={14} color="#fda4af" />
          <span>RESEARCH USE ONLY — Not validated for clinical decision-support at this deployment site.</span>
        </div>
        <div style={{ fontSize: '10px', color: '#fecdd3', textTransform: 'none', letterSpacing: '0.02em' }}>
          Guard 5 Non-Bypassable Gate &nbsp;·&nbsp; Deployment Tier: <strong>RESEARCH_ONLY</strong>
        </div>
      </div>

      {/* Top Header Bar */}
      <header style={{
        background: '#0b1120',
        borderBottom: '1px solid #1e293b',
        padding: '0 24px',
        height: '58px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 50,
        backdropFilter: 'blur(12px)',
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '34px', height: '34px', borderRadius: '8px',
            background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 4px 14px rgba(2, 132, 199, 0.35)'
          }}>
            <Activity size={20} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: '800', letterSpacing: '-0.02em', color: '#f8fafc' }}>
              LUNG CANCER AI <span style={{ color: '#06b6d4', fontWeight: '500' }}>RESEARCH PLATFORM</span>
            </div>
            <div style={{ fontSize: '10px', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Volumetric 3D DL · Vision Transformers · IBSI Radiomics · XAI
            </div>
          </div>
        </div>

        {/* Status Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {/* API Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#94a3b8' }}>
            {apiOnline === null ? (
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#475569', display: 'inline-block' }} />
            ) : apiOnline ? (
              <span className="status-dot-live" />
            ) : (
              <WifiOff size={14} color="#f43f5e" />
            )}
            {apiOnline === null ? 'Checking API...' : apiOnline ? 'API v2.0 Online' : 'API Offline (Demo)'}
          </div>

          {/* Study context pill */}
          <div style={{
            background: 'rgba(6, 182, 212, 0.08)', border: '1px solid rgba(6, 182, 212, 0.2)',
            borderRadius: '6px', padding: '4px 12px', fontSize: '11px', color: '#64748b',
            fontFamily: 'JetBrains Mono, monospace'
          }}>
            Study: <span style={{ color: '#38bdf8' }}>{currentStudyId}</span>
          </div>

          {/* RESEARCH badge */}
          <div style={{
            background: 'rgba(244, 63, 94, 0.12)', border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: '4px', padding: '4px 10px', fontSize: '11px', fontWeight: '700',
            color: '#f43f5e', display: 'flex', alignItems: 'center', gap: '5px'
          }}>
            <ShieldAlert size={13} /> RESEARCH USE ONLY
          </div>
        </div>
      </header>

      {/* Navigation Tabs Bar */}
      <nav style={{
        background: '#0a0f1d',
        borderBottom: '1px solid #1e293b',
        padding: '0 24px',
        display: 'flex',
        gap: '2px',
        overflowX: 'auto',
      }}>
        {navItems.map(item => {
          const Icon = item.icon;
          const isActive = activePanel === item.id;
          return (
            <button
              key={item.id}
              id={`nav-${item.id}`}
              onClick={() => setActivePanel(item.id)}
              className={`nav-tab${isActive ? ' active' : ''}`}
            >
              <Icon size={14} color={isActive ? '#06b6d4' : '#475569'} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Main Panel Content Area */}
      <main style={{ flex: 1, paddingBottom: '48px' }}>

        {activePanel === 'workstation' && (
          <WorkstationPanel 
            studyId={currentStudyId} 
            navigate={(route) => {
              if (route.includes('training'))   setActivePanel('training');
              else if (route.includes('benchmark')) setActivePanel('experiments');
              else if (route.includes('registry'))  setActivePanel('experiments');
            }}
          />
        )}

        {activePanel === 'intake' && (
          <IntakePanel onStudyLoaded={(studyId) => {
            setCurrentStudyId(studyId);
            setActivePanel('viewer');
          }} />
        )}

        {activePanel === 'cases' && (
          <CaseLogPanel onSelectCase={(studyId) => {
            setCurrentStudyId(studyId);
            setActivePanel('viewer');
          }} />
        )}

        {activePanel === 'viewer' && (
          <CTViewerPanel studyId={currentStudyId} onInspectNodule={(noduleId) => {
            setCurrentNoduleId(noduleId);
            setActivePanel('morphometry');
          }} />
        )}

        {activePanel === 'training'    && <TrainingPanel />}
        {activePanel === 'morphometry' && <MorphometryPanel noduleId={currentNoduleId} />}
        {activePanel === 'experiments' && <ExperimentsPanel />}
        {activePanel === 'puter-ai'    && <PuterAnalysisPanel />}

      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid #1e293b',
        background: '#070c14',
        padding: '10px 24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '10px',
        color: '#334155',
        position: 'fixed',
        bottom: 0,
        width: '100%',
        boxSizing: 'border-box',
        zIndex: 40
      }}>
        <span>Lung Cancer AI Research Platform v2.0 · Not FDA/CE Cleared · RESEARCH USE ONLY</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>
          Study: {currentStudyId} · Nodule: {currentNoduleId}
        </span>
      </footer>

    </div>
  );
};

export default App;
