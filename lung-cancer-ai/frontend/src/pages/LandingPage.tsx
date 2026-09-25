import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, ArrowRight, Layers, Cpu, Eye, FileText, Lock, Globe } from 'lucide-react';
import { api } from '../services/api';

interface LandingPageProps {
  navigate: (route: string) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ navigate }) => {
  const [appName, setAppName] = useState('Lung Cancer AI');
  const [domain, setDomain] = useState('lungcancerai.example.com');

  useEffect(() => {
    api.getDomainSettings()
      .then(d => {
        if (d.app_name) setAppName(d.app_name);
        if (d.public_domain) setDomain(d.public_domain);
      })
      .catch(() => {});
  }, []);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '48px 24px' }}>
      {/* Hero Section */}
      <div style={{ textAlign: 'center', marginBottom: '56px' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          background: 'rgba(6, 182, 212, 0.12)',
          border: '1px solid rgba(6, 182, 212, 0.3)',
          color: '#38bdf8',
          fontSize: '12px',
          fontWeight: '600',
          padding: '6px 14px',
          borderRadius: '20px',
          marginBottom: '20px'
        }}>
          <Globe size={14} />
          <span>PRODUCTION-GRADE 24/7 MEDICAL AI RESEARCH SERVER</span>
        </div>

        <h1 style={{
          fontSize: '44px',
          fontWeight: '800',
          letterSpacing: '-0.03em',
          margin: '0 0 16px 0',
          color: '#f8fafc',
          lineHeight: '1.15'
        }}>
          {appName.toUpperCase()}
        </h1>

        <p style={{
          fontSize: '18px',
          color: '#94a3b8',
          maxWidth: '780px',
          margin: '0 auto 32px auto',
          lineHeight: '1.6'
        }}>
          Multimodal CT Analysis, Volumetric 3D Deep Learning, Vision Transformers, IBSI Radiomics, and Explainable AI Platform.
        </p>

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '16px', flexWrap: 'wrap' }}>
          <button
            onClick={() => navigate('/viewer/CASE-DEMO-LIDC-0087')}
            style={{
              background: 'linear-gradient(135deg, #0284c7, #06b6d4)',
              color: '#ffffff',
              border: 'none',
              borderRadius: '6px',
              padding: '14px 28px',
              fontSize: '14px',
              fontWeight: '700',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 4px 20px rgba(2, 132, 199, 0.4)'
            }}
          >
            GET STARTED
            <ArrowRight size={16} />
          </button>

          <button
            onClick={() => navigate('/dashboard')}
            style={{
              background: '#0f172a',
              color: '#38bdf8',
              border: '1px solid #1e293b',
              borderRadius: '6px',
              padding: '14px 28px',
              fontSize: '14px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            RESEARCH PLATFORM
          </button>

          <button
            onClick={() => navigate('/login')}
            style={{
              background: '#1e293b',
              color: '#cbd5e1',
              border: '1px solid #334155',
              borderRadius: '6px',
              padding: '14px 28px',
              fontSize: '14px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            LOGIN
          </button>
        </div>
      </div>

      {/* Feature Pillar Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '24px', marginBottom: '56px' }}>
        {[
          {
            icon: Eye,
            title: 'Volumetric CT Segmentation',
            desc: 'nnU-Net v2 and 3D U-Net parenchyma masking and candidate nodule delineation.'
          },
          {
            icon: Cpu,
            title: '3D Swin Transformers & CNNs',
            desc: 'Shifted-window 3D self-attention paired with 3D ResNet-50 and DenseNet-121.'
          },
          {
            icon: Layers,
            title: '107 IBSI Radiomic Features',
            desc: 'Standardized first-order, shape, GLCM, GLRLM, GLSZM texture analysis with XGBoost.'
          },
          {
            icon: ShieldCheck,
            title: 'Multiplanar Explainable AI',
            desc: '3D Grad-CAM, Integrated Gradients, and Swin Attention Rollout with attribution notices.'
          }
        ].map((feat, i) => {
          const Icon = feat.icon;
          return (
            <div key={i} className="medical-card" style={{ padding: '24px' }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '8px',
                background: 'rgba(2, 132, 199, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '16px'
              }}>
                <Icon size={20} color="#38bdf8" />
              </div>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '16px', fontWeight: '700', color: '#f8fafc' }}>
                {feat.title}
              </h3>
              <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8', lineHeight: '1.5' }}>
                {feat.desc}
              </p>
            </div>
          );
        })}
      </div>

      {/* Medical Research Disclaimer */}
      <div style={{
        background: 'rgba(244, 63, 94, 0.08)',
        border: '1px solid rgba(244, 63, 94, 0.25)',
        borderRadius: '8px',
        padding: '20px',
        textAlign: 'center'
      }}>
        <div style={{ fontSize: '12px', fontWeight: '700', color: '#f43f5e', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
          Mandatory Clinical Notice
        </div>
        <div style={{ fontSize: '12px', color: '#fda4af', maxWidth: '800px', margin: '0 auto' }}>
          This platform is a medical AI research and decision-support prototype. Its outputs are not a definitive diagnosis and must not replace interpretation by qualified healthcare professionals.
        </div>
      </div>
    </div>
  );
};
