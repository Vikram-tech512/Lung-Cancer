import React, { useEffect, useRef } from 'react';

export interface HUHistogramProps {
  title?: string;
  meanHu?: number;
  meanHU?: number;
  stdHu?: number;
  stdHU?: number;
  minHu?: number;
  minHU?: number;
  maxHu?: number;
  maxHU?: number;
  huValues?: number[];
  height?: number;
  className?: string;
}

export const HUHistogram: React.FC<HUHistogramProps> = ({
  title,
  meanHu,
  meanHU,
  stdHu,
  stdHU,
  minHu,
  minHU,
  maxHu,
  maxHU,
  huValues,
  height = 90,
  className = '',
}) => {
  const effectiveMean = meanHu ?? meanHU ?? -35.0;
  const effectiveStd = stdHu ?? stdHU ?? 85.0;
  const effectiveMin = minHu ?? minHU ?? -650.0;
  const effectiveMax = maxHu ?? maxHU ?? 220.0;

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, width, h);

    // Standard thoracic HU range [-1000, +400]
    const minRange = -1000;
    const maxRange = 400;
    const rangeSpan = maxRange - minRange;
    const numBins = 50;
    const bins = new Array(numBins).fill(0);

    if (huValues && huValues.length > 0) {
      for (const val of huValues) {
        const norm = Math.max(0, Math.min(1, (val - minRange) / rangeSpan));
        const b = Math.min(numBins - 1, Math.floor(norm * numBins));
        bins[b]++;
      }
    } else {
      for (let i = 0; i < numBins; i++) {
        const binHu = minRange + (i + 0.5) * (rangeSpan / numBins);
        const z = (binHu - effectiveMean) / (effectiveStd || 80);
        bins[i] = Math.exp(-0.5 * z * z);
      }
    }

    const maxBinVal = Math.max(...bins, 0.001);

    // Baseline
    ctx.strokeStyle = '#2a2e33';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h - 18);
    ctx.lineTo(width, h - 18);
    ctx.stroke();

    // 0 HU water reference line
    const zeroX = ((0 - minRange) / rangeSpan) * width;
    ctx.strokeStyle = '#363b42';
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(zeroX, 0);
    ctx.lineTo(zeroX, h - 18);
    ctx.stroke();
    ctx.setLineDash([]);

    // Histogram bars
    const plotHeight = h - 22;
    const barWidth = width / numBins;

    for (let i = 0; i < numBins; i++) {
      const binHu = minRange + (i + 0.5) * (rangeSpan / numBins);
      const barH = (bins[i] / maxBinVal) * plotHeight;
      const x = i * barWidth;
      const y = plotHeight - barH;

      if (binHu < -600) {
        ctx.fillStyle = '#4a5568';
      } else if (binHu < 0) {
        ctx.fillStyle = '#3b82c4';
      } else if (binHu < 150) {
        ctx.fillStyle = '#3cb371';
      } else {
        ctx.fillStyle = '#d98a2c';
      }

      ctx.fillRect(x + 1, y, Math.max(1, barWidth - 1), barH);
    }

    // Mean HU vertical line
    const meanX = Math.max(0, Math.min(width, ((effectiveMean - minRange) / rangeSpan) * width));
    ctx.strokeStyle = '#e05252';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(meanX, 0);
    ctx.lineTo(meanX, plotHeight);
    ctx.stroke();

    // Ticks
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px var(--font-mono, monospace)';
    ctx.textAlign = 'left';
    ctx.fillText('-1000 HU', 4, h - 4);
    ctx.textAlign = 'center';
    ctx.fillText('0 HU', zeroX, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText('+400 HU', width - 4, h - 4);
  }, [effectiveMean, effectiveStd, effectiveMin, effectiveMax, huValues, height]);

  return (
    <div className={`hu-histogram-container ${className}`} style={{ width: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
        <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)' }}>
          {title || 'ATTENUATION DENSITY (HU HISTOGRAM)'}
        </span>
        <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          Mean: <strong className="tabular-nums" style={{ color: 'var(--text-main)' }}>{effectiveMean.toFixed(1)} HU</strong> (±{effectiveStd.toFixed(1)})
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={340}
        height={height}
        style={{ width: '100%', height: `${height}px`, display: 'block', backgroundColor: 'var(--bg-main)', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}
        aria-label="Attenuation density histogram in Hounsfield Units"
      />
    </div>
  );
};

export const EKGWaveform: React.FC<any> = (props) => {
  return <HUHistogram meanHu={props.meanHu || props.meanHU || -35.0} stdHu={props.stdHu || props.stdHU || 85.0} height={props.height || 75} />;
};
