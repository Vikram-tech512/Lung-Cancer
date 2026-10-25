import React from 'react';

export type BadgeVariant =
  | 'pass'
  | 'warning'
  | 'fail'
  | 'neutral'
  | 'accent'
  | 'success'
  | 'danger'
  | 'info';

export interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
  title?: string;
  style?: React.CSSProperties;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'neutral',
  children,
  className = '',
  title,
  style,
}) => {
  // Normalize alias variants
  const normalizedVariant =
    variant === 'success' ? 'pass' :
    variant === 'danger' ? 'fail' :
    variant === 'info' ? 'accent' :
    variant;

  return (
    <span className={`badge badge-${normalizedVariant} ${className}`} title={title} style={style}>
      {children}
    </span>
  );
};
