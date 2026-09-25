import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './pages/LandingPage';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { StudiesPage } from './pages/StudiesPage';
import { CTWorkstationPage } from './pages/CTWorkstationPage';
import { AnalysisTrackerPage } from './pages/AnalysisTrackerPage';
import { ModelsPage } from './pages/ModelsPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { ReportsPage } from './pages/ReportsPage';
import { AdminPage } from './pages/AdminPage';
import { SettingsPage } from './pages/SettingsPage';
import { RadiologistReviewPage } from './pages/RadiologistReviewPage';
import { ResearchDashboardPage } from './pages/ResearchDashboardPage';

import { ClinicalDisclaimerBanner } from './components/ClinicalDisclaimerBanner';

export const App: React.FC = () => {
  const [currentRoute, setCurrentRoute] = useState<string>(window.location.pathname || '/');

  useEffect(() => {
    const onPopState = () => {
      setCurrentRoute(window.location.pathname);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const navigate = (route: string) => {
    window.history.pushState({}, '', route);
    setCurrentRoute(route);
    window.scrollTo(0, 0);
  };

  const renderPage = () => {
    if (currentRoute === '/') return <LandingPage navigate={navigate} />;
    if (currentRoute === '/login') return <LoginPage navigate={navigate} />;
    if (currentRoute === '/dashboard') return <DashboardPage navigate={navigate} />;
    if (currentRoute === '/studies') return <StudiesPage navigate={navigate} />;
    if (currentRoute.startsWith('/viewer')) {
      const parts = currentRoute.split('/');
      const studyId = parts[2] || 'CASE-DEMO-LIDC-0087';
      return <CTWorkstationPage studyId={studyId} navigate={navigate} />;
    }
    if (currentRoute.startsWith('/analysis')) {
      const parts = currentRoute.split('/');
      const analysisId = parts[2] || 'JOB-DEMO-9910';
      return <AnalysisTrackerPage analysisId={analysisId} navigate={navigate} />;
    }
    if (currentRoute === '/review' || currentRoute.startsWith('/review/')) {
      const parts = currentRoute.split('/');
      const studyId = parts[2] || 'CASE-DEMO-LIDC-0087';
      return <RadiologistReviewPage studyId={studyId} navigate={navigate} />;
    }
    if (currentRoute === '/research') return <ResearchDashboardPage navigate={navigate} />;
    if (currentRoute === '/models') return <ModelsPage />;
    if (currentRoute === '/experiments') return <ExperimentsPage />;
    if (currentRoute === '/datasets') return <DatasetsPage />;
    if (currentRoute === '/reports') return <ReportsPage />;
    if (currentRoute === '/admin') return <AdminPage />;
    if (currentRoute === '/settings') return <SettingsPage />;

    // Fallback to Workstation
    return <CTWorkstationPage studyId="CASE-DEMO-LIDC-0087" navigate={navigate} />;
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#060911' }}>
      <Navbar currentRoute={currentRoute} navigate={navigate} />
      <main style={{ flex: 1 }}>
        {renderPage()}
      </main>
      <ClinicalDisclaimerBanner />
    </div>
  );
};

export default App;
