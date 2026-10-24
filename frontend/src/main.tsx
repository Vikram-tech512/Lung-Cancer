import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('Workstation caught error:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#0a0f1d',
          color: '#f8fafc',
          padding: '24px',
          fontFamily: 'Inter, sans-serif',
          textAlign: 'center'
        }}>
          <h2 style={{ fontSize: '20px', fontWeight: 800, color: '#f43f5e', marginBottom: '8px' }}>
            Clinical Workstation Initialization Error
          </h2>
          <p style={{ color: '#94a3b8', fontSize: '13px', maxWidth: '500px', marginBottom: '16px' }}>
            {this.state.error?.message || 'An unexpected rendering error occurred.'}
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              background: '#0284c7',
              color: '#fff',
              border: 'none',
              borderRadius: '6px',
              padding: '10px 18px',
              fontSize: '13px',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            Reload Workstation
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)
