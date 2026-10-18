import React, { useState, useEffect } from 'react';
import { 
  Upload, Database, Eye, Activity, Wifi, WifiOff, Sparkles, BrainCircuit, Cpu
} from 'lucide-react';

import { WorkstationPanel } from './panels/WorkstationPanel';
import { AIAnalysisPanel }  from './panels/AIAnalysisPanel';
import { TrainingPanel }    from './panels/TrainingPanel';
import { IntakePanel }       from './panels/IntakePanel';
import { CaseLogPanel }      from './panels/CaseLogPanel';
import { CTViewerPanel }     from './panels/CTViewerPanel';

type PanelId = 'workstation' | 'ai-analysis' | 'training' | 'intake' | 'cases' | 'viewer';

export const App: React.FC = () => {
  const [activePanel, setActivePanel]       = useState<PanelId>('workstation');
  const [currentStudyId, setCurrentStudyId] = useState<string>('CASE-DEMO-LIDC-0087');
  const [currentNoduleId, setCurrentNoduleId] = useState<string>('NODULE-01');
  const [apiOnline, setApiOnline]           = useState<boolean | null>(null);

  // Poll API health every 15 seconds
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const r = await fetch('/api/v2/health', { signal: AbortSignal.timeout(2500) });
        setApiOnline(r.ok);
      } catch {
        setApiOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15_000);
    return () => clearInterval(interval);
  }, []);

  const navItems: { id: PanelId; label: string; icon: React.FC<any>; badge?: string }[] = [
    { id: 'workstation', label: '00 · WORKSTATION', icon: Activity },
    { id: 'ai-analysis', label: '01 · AI DIAGNOSTIC ENGINE (API)', icon: Sparkles, badge: 'GEMINI READY' },
    { id: 'training',    label: '02 · TRAINING', icon: Cpu },
    { id: 'intake',      label: '03 · INTAKE', icon: Upload },
    { id: 'cases',       label: '04 · CASE LOG', icon: Database },
    { id: 'viewer',      label: '05 · CT VIEWER', icon: Eye },
  ];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#070a11' }}>

      {/* Top Header Bar */}
      <header style={{
        background: '#0b1120',
        borderBottom: '1px solid #1e293b',
        padding: '0 24px',
        height: '60px',
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
            width: '36px', height: '36px', borderRadius: '10px',
            background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 4px 14px rgba(2, 132, 199, 0.35)'
          }}>
            <BrainCircuit size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: '800', letterSpacing: '-0.02em', color: '#f8fafc' }}>
              LUNG CANCER AI <span style={{ color: '#06b6d4', fontWeight: '600' }}>PLATFORM</span>
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Volumetric 3D DL · Gemini Multi-Modal Engine · XAI Localization
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs Bar */}
      <nav style={{
        background: '#0a0f1d',
        borderBottom: '1px solid #1e293b',
        padding: '0 24px',
        display: 'flex',
        gap: '4px',
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
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '12px 18px',
                border: 'none',
                background: isActive ? 'rgba(6, 182, 212, 0.12)' : 'transparent',
                borderBottom: isActive ? '2px solid #06b6d4' : '2px solid transparent',
                color: isActive ? '#f8fafc' : '#64748b',
                fontWeight: isActive ? '700' : '500',
                fontSize: '12px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <Icon size={15} color={isActive ? '#06b6d4' : '#64748b'} />
              <span>{item.label}</span>
              {item.badge && (
                <span style={{
                  fontSize: '9px',
                  fontWeight: '700',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  background: 'rgba(56, 189, 248, 0.2)',
                  color: '#38bdf8',
                  letterSpacing: '0.04em'
                }}>
                  {item.badge}
                </span>
              )}
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
              if (route.includes('training')) setActivePanel('training');
              else if (route.includes('ai') || route.includes('api')) setActivePanel('ai-analysis');
              else if (route.includes('viewer')) setActivePanel('viewer');
              else if (route.includes('cases')) setActivePanel('cases');
              else if (route.includes('intake')) setActivePanel('intake');
            }}
          />
        )}

        {activePanel === 'ai-analysis' && (
          <AIAnalysisPanel />
        )}

        {activePanel === 'training' && (
          <TrainingPanel />
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
            setActivePanel('ai-analysis');
          }} />
        )}

      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid #1e293b',
        background: '#070c14',
        padding: '10px 24px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        fontSize: '11px',
        color: '#64748b',
        position: 'fixed',
        bottom: 0,
        width: '100%',
        boxSizing: 'border-box',
        zIndex: 40
      }}>
        <span>Lung Cancer AI Platform v2.0 · Multi-Modal Clinical Intelligence</span>
      </footer>

    </div>
  );
};

export default App;
