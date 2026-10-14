import React, { useEffect, useRef } from 'react';

interface EKGWaveformProps {
  points?: number[];
  height?: number;
  color?: string;
}

export const EKGWaveform: React.FC<EKGWaveformProps> = ({ points, height = 48, color = '#f43f5e' }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, width, h);

    // Grid baseline
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(width, h / 2);
    ctx.stroke();

    // Draw ECG pulse line
    const data = points && points.length > 0
      ? points
      : [0, 0, 0.05, -0.05, 0, 0.1, -0.4, 0.9, -0.7, 0.3, -0.1, 0, 0.05, 0, 0, 0.05, -0.05, 0, 0.1, -0.4, 0.9, -0.7, 0.3, 0];

    const step = width / (data.length - 1);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;

    ctx.beginPath();
    data.forEach((val, i) => {
      const x = i * step;
      const y = h / 2 - val * (h * 0.42);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Reset shadow
    ctx.shadowBlur = 0;
  }, [points, height, color]);

  return (
    <div style={{ position: 'relative', width: '100%', height: `${height}px` }}>
      <canvas
        ref={canvasRef}
        width={260}
        height={height}
        style={{ display: 'block', width: '100%', height: `${height}px` }}
      />
    </div>
  );
};
