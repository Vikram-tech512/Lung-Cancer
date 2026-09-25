import React, { useState } from 'react';
import { 
  Upload, Database, Eye, Cpu, CheckSquare, Ruler, Award, 
  GitBranch, Box, Activity, ShieldAlert 
} from 'lucide-react';

import { IntakePanel } from './panels/IntakePanel';
import { CaseLogPanel } from './panels/CaseLogPanel';
import { CTViewerPanel } from './panels/CTViewerPanel';
import { TrainingPanel } from './panels/TrainingPanel';
import { VerdictPanel } from './panels/VerdictPanel';
import { MorphometryPanel } from './panels/MorphometryPanel';
import { BenchmarkPanel } from './panels/BenchmarkPanel';
import { ExperimentsPanel } from './panels/ExperimentsPanel';
import { RegistryPanel } from './panels/RegistryPanel';

type PanelId = 'intake' | 'cases' | 'viewer' | 'training' | 'verdict' | 'morphometry' | 'benchmark' | 'experiments' | 'registry';

export const App: React.FC = () => {
  const [activePanel, setActivePanel] = useState<PanelId>('viewer');
  const [currentStudyId, setCurrentStudyId] = useState<string>('CASE-DEMO-LIDC-0087');
  const [currentNoduleId, setCurrentNoduleId] = useState<string>('NODULE-01');

  const navItems = [
    { id: 'intake', label: '01 · INTAKE', icon: Upload },
    { id: 'cases', label: '02 · CASE LOG', icon: Database },
    { id: 'viewer', label: '03 · CT VIEWER', icon: Eye },
    { id: 'training', label: '04 · TRAINING', icon: Cpu },
    { id: 'verdict', label: '05 · VERDICT', icon: CheckSquare },
    { id: 'morphometry', label: '06 · MORPHOMETRY', icon: Ruler },
    { id: 'benchmark', label: '07 · BENCHMARK', icon: Award },
    { id: 'experiments', label: '08 · EXPERIMENTS', icon: GitBranch },
    { id: 'registry', label: '09 · REGISTRY', icon: Box },
  ];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#070a11' }}>
      {/* Guard 5 Mandatory Clinical Advisory Banner */}
      <div style={{
        background: '#881337',
        borderBottom: '1px solid #be123c',
        color: '#ffe4e6',
        padding: '6px 24px',
        fontSize: '11px',
        fontWeight: '700',
        letterSpacing: '0.04em',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        textTransform: 'uppercase'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldAlert size={15} color="#fda4af" />
          <span>RESEARCH USE ONLY — not validated for this site/deployment as a clinical decision-support tool.</span>
        </div>
        <div style={{ fontSize: '10px', color: '#fecdd3', textTransform: 'none' }}>
          Deployment Tier: <strong>RESEARCH_ONLY</strong> · Guard 5 Non-Bypassable Gate Enforced
        </div>
      </div>

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
        zIndex: 50
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Activity size={20} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: '800', letterSpacing: '-0.02em', color: '#f8fafc' }}>
              LUNG CANCER AI <span style={{ color: '#06b6d4', fontWeight: '500' }}>RESEARCH PLATFORM</span>
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Volumetric 3D DL · Vision Transformers · IBSI Radiomics · XAI
            </div>
          </div>
        </div>

        {/* System Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#94a3b8' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
            API v2.0 Online
          </div>

          <div style={{
            background: 'rgba(244, 63, 94, 0.12)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: '4px',
            padding: '4px 10px',
            fontSize: '11px',
            fontWeight: '700',
            color: '#f43f5e',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <ShieldAlert size={14} />
            RESEARCH USE ONLY
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
        overflowX: 'auto'
      }}>
        {navItems.map(item => {
          const Icon = item.icon;
          const isActive = activePanel === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActivePanel(item.id as PanelId)}
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: isActive ? '2px solid #06b6d4' : '2px solid transparent',
                color: isActive ? '#f8fafc' : '#64748b',
                padding: '12px 14px',
                fontSize: '12px',
                fontWeight: isActive ? '700' : '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={15} color={isActive ? '#06b6d4' : '#64748b'} />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Main Panel Content Area */}
      <main style={{ flex: 1, paddingBottom: '40px' }}>
        {activePanel === 'intake' && (
          <IntakePanel
            onStudyLoaded={(studyId) => {
              setCurrentStudyId(studyId);
              setActivePanel('viewer');
            }}
          />
        )}

        {activePanel === 'cases' && (
          <CaseLogPanel
            onSelectCase={(studyId) => {
              setCurrentStudyId(studyId);
              setActivePanel('viewer');
            }}
          />
        )}

        {activePanel === 'viewer' && (
          <CTViewerPanel
            studyId={currentStudyId}
            onInspectNodule={(noduleId) => {
              setCurrentNoduleId(noduleId);
              setActivePanel('verdict');
            }}
          />
        )}

        {activePanel === 'training' && <TrainingPanel />}

        {activePanel === 'verdict' && <VerdictPanel noduleId={currentNoduleId} />}

        {activePanel === 'morphometry' && <MorphometryPanel noduleId={currentNoduleId} />}

        {activePanel === 'benchmark' && <BenchmarkPanel />}

        {activePanel === 'experiments' && <ExperimentsPanel />}

        {activePanel === 'registry' && <RegistryPanel />}
      </main>
    </div>
  );
};

export default App;
