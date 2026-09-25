import React, { useState, useEffect } from 'react';
import { Cpu, Box, CheckCircle2, AlertCircle, HardDrive } from 'lucide-react';
import { api } from '../services/api';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<any[]>([]);

  useEffect(() => {
    api.listModels()
      .then(setModels)
      .catch(() => {});
  }, []);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '24px', fontWeight: '800', margin: 0, color: '#f8fafc' }}>
          MODEL ARCHITECTURAL REGISTRY
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0 0 0' }}>
          Concrete 3D Deep Learning and Vision Transformer implementations with verified input tensor dimensions.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '20px' }}>
        {models.map((m, idx) => {
          const isProd = m.stage === 'PRODUCTION';
          return (
            <div key={idx} className="medical-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontSize: '16px', fontWeight: '800', color: '#f8fafc' }}>{m.model_name}</span>
                  <span style={{
                    fontSize: '10px',
                    fontWeight: '700',
                    padding: '3px 8px',
                    borderRadius: '4px',
                    backgroundColor: isProd ? 'rgba(16, 185, 129, 0.2)' : 'rgba(56, 189, 248, 0.2)',
                    color: isProd ? '#34d399' : '#38bdf8',
                    border: `1px solid ${isProd ? 'rgba(16, 185, 129, 0.4)' : 'rgba(56, 189, 248, 0.4)'}`
                  }}>
                    {m.stage}
                  </span>
                </div>

                <div style={{ fontSize: '11px', color: '#64748b', fontFamily: 'JetBrains Mono, monospace', marginBottom: '12px' }}>
                  Version: {m.version}
                </div>

                <div style={{ fontSize: '12px', color: '#cbd5e1', marginBottom: '14px', background: '#070c17', padding: '10px', borderRadius: '6px', border: '1px solid #1c2638' }}>
                  <div style={{ color: '#64748b', fontSize: '10px', textTransform: 'uppercase', marginBottom: '2px', fontWeight: '700' }}>Architecture</div>
                  {m.architecture}
                </div>

                <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '12px' }}>
                  Input Dimension: <code style={{ color: '#38bdf8' }}>{m.input_dimension}</code>
                </div>
              </div>

              <div style={{ borderTop: '1px solid #1c2638', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: '#64748b' }}>Validation AUC</span>
                <span style={{ fontSize: '14px', fontWeight: '700', fontFamily: 'JetBrains Mono, monospace', color: m.val_auc ? '#34d399' : '#64748b' }}>
                  {m.val_auc ? m.val_auc.toFixed(3) : (m.val_dice ? `Dice: ${m.val_dice.toFixed(3)}` : 'ACTIVE')}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
