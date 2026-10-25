import React from 'react';

export interface PanelProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  actions,
  children,
  className = '',
  style
}) => {
  return (
    <div className={`panel ${className}`} style={style}>
      {title && (
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div className="panel-title">{title}</div>
            {subtitle && <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>{subtitle}</div>}
          </div>
          {actions && <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>{actions}</div>}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </div>
  );
};
