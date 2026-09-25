import React, { useState, useEffect } from 'react';
import { 
  Activity, ShieldAlert, LayoutDashboard, Eye, Database, 
  Cpu, GitBranch, FolderGit2, FileText, Server, Settings, LogIn, LogOut,
  CheckSquare, BarChart2
} from 'lucide-react';
import { getAuthToken, clearAuthToken, api } from '../services/api';

interface NavbarProps {
  currentRoute: string;
  navigate: (route: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentRoute, navigate }) => {
  const [appName, setAppName] = useState('Lung Cancer AI');
  const [domain, setDomain] = useState('lungcancerai.example.com');
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    setIsLoggedIn(!!getAuthToken());
    api.getDomainSettings()
      .then(d => {
        if (d.app_name) setAppName(d.app_name);
        if (d.public_domain) setDomain(d.public_domain);
      })
      .catch(() => {});
  }, []);

  const handleLogout = () => {
    clearAuthToken();
    setIsLoggedIn(false);
    navigate('/');
  };

  const navLinks = [
    { route: '/', label: 'Home', icon: Activity },
    { route: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { route: '/viewer/CASE-DEMO-LIDC-0087', label: 'CT Workstation', icon: Eye },
    { route: '/review', label: 'Review Sign-Off', icon: CheckSquare },
    { route: '/research', label: 'Research Benchmark', icon: BarChart2 },
    { route: '/studies', label: 'Studies', icon: Database },
    { route: '/models', label: 'Models', icon: Cpu },
    { route: '/experiments', label: 'Experiments', icon: GitBranch },
    { route: '/datasets', label: 'Datasets', icon: FolderGit2 },
    { route: '/reports', label: 'Reports', icon: FileText },
    { route: '/admin', label: 'Admin Telemetry', icon: Server },
    { route: '/settings', label: 'Domain Settings', icon: Settings },
  ];

  return (
    <header style={{
      backgroundColor: '#0a0f1d',
      borderBottom: '1px solid #1c2638',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      {/* Top Banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        height: '56px',
        borderBottom: '1px solid #141e30'
      }}>
        <div 
          onClick={() => navigate('/')} 
          style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
        >
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Activity size={18} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: '800', letterSpacing: '-0.02em', color: '#f8fafc' }}>
              {appName.toUpperCase()} <span style={{ color: '#06b6d4', fontWeight: '500' }}>PLATFORM</span>
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>
              {domain} | Production 24/7
            </div>
          </div>
        </div>

        {/* Status Indicators & Auth */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: '#10b981', background: 'rgba(16, 185, 129, 0.1)', padding: '3px 8px', borderRadius: '4px', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
            API & Redis Connected
          </div>

          <div style={{
            background: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: '4px',
            padding: '3px 8px',
            fontSize: '11px',
            fontWeight: '700',
            color: '#f43f5e',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <ShieldAlert size={12} />
            RESEARCH USE ONLY
          </div>

          {isLoggedIn ? (
            <button
              onClick={handleLogout}
              style={{
                background: '#1e293b',
                color: '#cbd5e1',
                border: '1px solid #334155',
                padding: '4px 10px',
                borderRadius: '5px',
                fontSize: '11px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <LogOut size={12} />
              Logout
            </button>
          ) : (
            <button
              onClick={() => navigate('/login')}
              style={{
                background: '#0284c7',
                color: '#ffffff',
                border: 'none',
                padding: '5px 12px',
                borderRadius: '5px',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}
            >
              <LogIn size={12} />
              Login
            </button>
          )}
        </div>
      </div>

      {/* Navigation Links Bar */}
      <nav style={{
        display: 'flex',
        padding: '0 20px',
        overflowX: 'auto',
        gap: '2px'
      }}>
        {navLinks.map(item => {
          const Icon = item.icon;
          const isActive = currentRoute === item.route || (item.route.startsWith('/viewer') && currentRoute.startsWith('/viewer'));
          return (
            <button
              key={item.route}
              onClick={() => navigate(item.route)}
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: isActive ? '2px solid #06b6d4' : '2px solid transparent',
                color: isActive ? '#f8fafc' : '#64748b',
                padding: '10px 12px',
                fontSize: '12px',
                fontWeight: isActive ? '700' : '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
            >
              <Icon size={14} color={isActive ? '#06b6d4' : '#64748b'} />
              {item.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};
