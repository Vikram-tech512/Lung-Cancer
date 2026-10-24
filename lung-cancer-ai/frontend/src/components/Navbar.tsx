import React, { useState, useEffect } from 'react';
import {
  Activity, ShieldAlert, LayoutDashboard, Eye, Database,
  Cpu, GitBranch, FolderGit2, FileText, Server, Settings, LogIn, LogOut,
  CheckSquare, BarChart2, Sun, Moon
} from 'lucide-react';
import { getAuthToken, clearAuthToken, api } from '../services/api';
import { useTheme } from '../context/ThemeContext';

interface NavbarProps {
  currentRoute: string;
  navigate: (route: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentRoute, navigate }) => {
  const [appName, setAppName] = useState('Lung Cancer AI');
  const [domain, setDomain] = useState('lungcancerai.example.com');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const { theme, toggleTheme } = useTheme();

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
    { route: '/review', label: 'Review', icon: CheckSquare },
    { route: '/research', label: 'Research', icon: BarChart2 },
    { route: '/studies', label: 'Studies', icon: Database },
    { route: '/models', label: 'Models', icon: Cpu },
    { route: '/experiments', label: 'Experiments', icon: GitBranch },
    { route: '/datasets', label: 'Datasets', icon: FolderGit2 },
    { route: '/reports', label: 'Reports', icon: FileText },
    { route: '/admin', label: 'Admin', icon: Server },
    { route: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <header style={{
      backgroundColor: 'var(--bg-surface)',
      borderBottom: '1px solid var(--border)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: 'var(--shadow-card)',
      transition: 'background-color 0.3s ease, border-color 0.3s ease',
    }}>
      {/* Top Banner */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        height: '56px',
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        {/* Brand */}
        <div
          onClick={() => navigate('/')}
          style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}
        >
          <div style={{
            width: '34px',
            height: '34px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, var(--primary), var(--accent-cyan))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 8px var(--primary-glow)',
          }}>
            <Activity size={18} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: '800', letterSpacing: '-0.02em', color: 'var(--text-main)' }}>
              {appName.toUpperCase()} <span style={{ color: 'var(--accent-cyan)', fontWeight: '500' }}>PLATFORM</span>
            </div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'JetBrains Mono, monospace' }}>
              {domain}
            </div>
          </div>
        </div>

        {/* Right controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* API status */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            fontSize: '11px', color: 'var(--accent-emerald)',
            background: 'rgba(16,185,129,0.1)',
            padding: '3px 9px', borderRadius: '20px',
            border: '1px solid rgba(16,185,129,0.3)',
          }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent-emerald)', display: 'inline-block' }} />
            API Live
          </div>

          {/* Research badge */}
          <div style={{
            background: 'rgba(244, 63, 94, 0.08)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: '20px',
            padding: '3px 9px',
            fontSize: '11px',
            fontWeight: '700',
            color: 'var(--accent-rose)',
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
          }}>
            <ShieldAlert size={11} />
            Research Only
          </div>

          {/* Dark / Light toggle */}
          <button
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            style={{
              background: 'var(--bg-input)',
              border: '1px solid var(--border)',
              borderRadius: '8px',
              padding: '6px 8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              color: 'var(--text-secondary)',
              transition: 'all 0.18s ease',
            }}
          >
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          </button>

          {/* Auth */}
          {isLoggedIn ? (
            <button
              onClick={handleLogout}
              style={{
                background: 'var(--bg-input)',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border)',
                padding: '5px 11px',
                borderRadius: '7px',
                fontSize: '12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontWeight: '600',
              }}
            >
              <LogOut size={13} />
              Logout
            </button>
          ) : (
            <button
              onClick={() => navigate('/login')}
              style={{
                background: 'var(--primary)',
                color: '#ffffff',
                border: 'none',
                padding: '5px 13px',
                borderRadius: '7px',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: '0 2px 8px var(--primary-glow)',
              }}
            >
              <LogIn size={13} />
              Sign In
            </button>
          )}
        </div>
      </div>

      {/* Navigation Links Bar */}
      <nav style={{
        display: 'flex',
        padding: '0 20px',
        overflowX: 'auto',
        gap: '2px',
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
                borderBottom: isActive ? '2px solid var(--accent-cyan)' : '2px solid transparent',
                color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
                padding: '9px 11px',
                fontSize: '12px',
                fontWeight: isActive ? '700' : '500',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '5px',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease',
              }}
            >
              <Icon size={13} color={isActive ? 'var(--accent-cyan)' : 'var(--text-muted)'} />
              {item.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};
