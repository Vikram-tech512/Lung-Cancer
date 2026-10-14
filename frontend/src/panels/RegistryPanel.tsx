import React, { useEffect, useState } from 'react';
import { Box, CheckCircle2, AlertCircle, HardDrive, Info } from 'lucide-react';
import { ModelRegistryItem } from '../types';
import { API_BASE } from '../services/api';

export const RegistryPanel: React.FC = () => {
  const [registry, setRegistry] = useState<Record<string, ModelRegistryItem[]>>({});

  useEffect(() => {
    fetch(`${API_BASE}/registry`)
      .then(res => res.json())
      .then(data => setRegistry(data))
      .catch(() => {
        setRegistry({
          '3D ResNet-50': [
            {
              model_name: '3D ResNet-50',
              version: 'v1.0.0',
              stage: 'NOT_TRAINED',
              architecture: '3D ResNet-50 Bottleneck + MC Dropout (T=30)',
              input_dimension: '(1, 64, 64, 64)',
              metrics: { auc_roc: 0.0, sensitivity: 0.0, specificity: 0.0 },
              created_at: '2026-09-22 13:00:00',
              notes: 'Primary 3D volumetric classifier. Code implemented.'
            }
          ],
          '3D Swin Transformer': [
            {
              model_name: '3D Swin Transformer',
              version: 'v1.0.0',
              stage: 'NOT_TRAINED',
              architecture: 'Swin3D-Tiny (window_size=4x4x4, embed_dim=96)',
              input_dimension: '(1, 64, 64, 64)',
              metrics: { auc_roc: 0.0, sensitivity: 0.0, specificity: 0.0 },
              created_at: '2026-09-22 13:00:00',
              notes: 'Transformer branch extending 2D ViT research.'
            }
          ],
          'Hybrid CNN-ViT (Phase 4)': [
            {
              model_name: 'Hybrid CNN-ViT (Phase 4)',
              version: 'v1.0.0',
              stage: 'STAGING',
              architecture: 'ResNet50 stem + ViT-B/16 gated cross-attention fusion',
              input_dimension: '(3, 224, 224)',
              metrics: { auc_roc: 0.942, accuracy: 0.913, f1_score: 0.908 },
              created_at: '2026-09-22 12:00:00',
              notes: 'Trained on phase 4 slice dataset (from master_execution_log).'
            }
          ],
          '3D DenseNet-121': [
            {
              model_name: '3D DenseNet-121',
              version: 'v1.0.0',
              stage: 'NOT_TRAINED',
              architecture: '3D DenseNet-121 (growth_rate=32, 4 dense blocks)',
              input_dimension: '(1, 64, 64, 64)',
              metrics: { auc_roc: 0.0 },
              created_at: '2026-09-22 13:00:00',
              notes: 'High-parameter-efficiency volumetric baseline.'
            }
          ],
          'Radiomics XGBoost': [
            {
              model_name: 'Radiomics XGBoost',
              version: 'v1.0.0',
              stage: 'NOT_TRAINED',
              architecture: 'XGBoost (107 IBSI features + SHAP tree explainer)',
              input_dimension: '(107,)',
              metrics: { auc_roc: 0.0 },
              created_at: '2026-09-22 13:00:00',
              notes: 'Quantitative feature baseline.'
            }
          ]
        });
      });
  }, []);

  return (
    <div style={{ padding: '24px', maxWidth: '1100px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.02em', margin: 0, color: '#f8fafc' }}>
          09 · CENTRAL CLINICAL MODEL REGISTRY
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '6px' }}>
          Production artifacts, input tensor dimensions, operational stages, and validation lineage.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '20px' }}>
        {Object.entries(registry).flatMap(([_, versions]) => versions).map((m, idx) => {
          const isStaging = m.stage === 'STAGING';
          const isProduction = m.stage === 'PRODUCTION';
          return (
            <div key={idx} className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '16px', fontWeight: '700', color: '#f8fafc' }}>{m.model_name}</span>
                  <span style={{
                    fontSize: '10px',
                    fontWeight: '700',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    backgroundColor: isProduction ? 'rgba(16, 185, 129, 0.2)' : (isStaging ? 'rgba(56, 189, 248, 0.2)' : 'rgba(148, 163, 184, 0.12)'),
                    color: isProduction ? '#34d399' : (isStaging ? '#38bdf8' : '#94a3b8'),
                    border: `1px solid ${isProduction ? 'rgba(16, 185, 129, 0.4)' : (isStaging ? 'rgba(56, 189, 248, 0.4)' : 'rgba(148, 163, 184, 0.2)')}`
                  }}>
                    {m.stage}
                  </span>
                </div>

                <div style={{ fontSize: '11px', color: '#64748b', fontFamily: 'JetBrains Mono, monospace', marginBottom: '12px' }}>
                  Version: {m.version} | Created: {m.created_at}
                </div>

                <div style={{ fontSize: '12px', color: '#cbd5e1', marginBottom: '14px', background: '#0b1120', padding: '10px', borderRadius: '6px', border: '1px solid #1e293b' }}>
                  <div style={{ color: '#64748b', fontSize: '10px', textTransform: 'uppercase', marginBottom: '2px' }}>Architecture</div>
                  {m.architecture}
                </div>

                <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '12px' }}>
                  Input Tensor: <code style={{ color: '#38bdf8' }}>{m.input_dimension}</code>
                </div>

                <div style={{ fontSize: '12px', color: '#64748b', fontStyle: 'italic', marginBottom: '16px' }}>
                  "{m.notes}"
                </div>
              </div>

              <div style={{ borderTop: '1px solid #1e293b', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Validation AUC</span>
                <span style={{ fontSize: '14px', fontWeight: '700', fontFamily: 'JetBrains Mono, monospace', color: m.metrics.auc_roc ? '#34d399' : '#64748b' }}>
                  {m.metrics.auc_roc ? m.metrics.auc_roc.toFixed(3) : 'NOT RUN'}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
