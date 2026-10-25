import React from 'react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md';
  loading?: boolean;
  icon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'secondary',
  size = 'sm',
  loading = false,
  icon,
  className = '',
  disabled,
  ...props
}) => {
  const variantClass = variant === 'primary' 
    ? 'btn-primary' 
    : variant === 'danger' 
    ? 'btn-danger' 
    : 'btn-secondary';

  const sizeStyle = size === 'sm' 
    ? { padding: '4px 8px', fontSize: '12px' } 
    : { padding: '6px 12px', fontSize: '13px' };

  return (
    <button
      className={`btn ${variantClass} ${className}`}
      style={sizeStyle}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <span style={{ display: 'inline-block', width: '12px', height: '12px', border: '2px solid currentColor', borderRightColor: 'transparent', borderRadius: '50%', animation: 'spin 0.6s linear infinite' }} />
      ) : icon ? (
        <span style={{ display: 'inline-flex', alignItems: 'center' }}>{icon}</span>
      ) : null}
      {children}
    </button>
  );
};
