import { useEffect, useRef } from 'react'
import { INK, SERIES, HAIR, AXIS, num } from './theme.js'
import { Tip, useTip } from './ui.jsx'

// One real window turned into features: the window as a picture, one channel as a line, the statistics read from it, and a MiniRocket-style kernel run over it.
// Data: backend evaluation/feature_demo.py.

const FONT = 'var(--font)'
const T = (props) => <text fontFamily={FONT} fontSize={11} fill={INK.secondary} {...props} />
const NEG = SERIES[0], POS = SERIES[2]

// The whole window: rows are channels, columns are time. Each channel is scaled to its own mean and spread.
export function WindowPicture({ picture, mark }) {
  const ref = useRef(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const ctx = c.getContext('2d')
    const rows = picture.length, cols = picture[0].length
    c.width = cols
    c.height = rows
    const img = ctx.createImageData(cols, rows)
    const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]
    const [nr, ng, nb] = hex(NEG), [pr, pg, pb] = hex(POS), base = [12, 20, 36]
    picture.forEach((row, i) => [...row].forEach((ch, j) => {
      const v = ch.charCodeAt(0) - 48 - 6   // steps of 0.5 standard deviations, -6 to 6
      const t = Math.min(1, Math.abs(v) / 4), [r, g, b] = v >= 0 ? [pr, pg, pb] : [nr, ng, nb], k = (i * cols + j) * 4
      img.data[k] = base[0] + (r - base[0]) * t
      img.data[k + 1] = base[1] + (g - base[1]) * t
      img.data[k + 2] = base[2] + (b - base[2]) * t
      img.data[k + 3] = 255
    }))
    ctx.putImageData(img, 0, 0)
  }, [picture])
  const rows = picture.length
  return (
    <div className="msr-picture-wrap">
      <canvas ref={ref} className="msr-picture" role="img" aria-label="One window: channels from top to bottom, time from left to right" />
      {[1, 2, 3, 4].map((l) => <span key={l} className="msr-link-label" style={{ top: `${(l - 1) * 25}%` }}>link {l}</span>)}
      {[1, 2, 3].map((l) => <i key={l} className="msr-link-line" style={{ top: `${l * 25}%` }} />)}
      {mark != null && <i className="msr-mark-line" style={{ top: `${((mark + 0.5) / (rows * 2)) * 100}%` }}><span>channel {mark}</span></i>}
    </div>
  )
}

const path = (values, x, y) => values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')

// A line over the 500 time steps, optionally with a horizontal threshold and shading where the line is above it.
export function Line({ values, color, threshold, height = 130, label, mask, pattern, offset = 0, total = values.length }) {
  const W = 620, left = 40, right = 10, top = 8, bottom = 24
  const lo = Math.min(...values, threshold ?? Infinity), hi = Math.max(...values, threshold ?? -Infinity)
  const pad = (hi - lo) * 0.08 || 1
  const x = (i) => left + ((i + offset) / (total - 1)) * (W - left - right)
  const y = (v) => top + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * (height - top - bottom)
  const ticks = [lo, (lo + hi) / 2, hi]
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" role="img" aria-label={label}>
      {ticks.map((v) => <g key={v}><line x1={left} x2={W - right} y1={y(v)} y2={y(v)} stroke={HAIR} /><T x={left - 6} y={y(v) + 4} textAnchor="end" fill={INK.muted}>{num(v, 1)}</T></g>)}
      {[0, 1, 2, 3, 4, 5].map((s) => <T key={s} x={x((s / 5) * (total - 1) - offset)} y={height - 6} textAnchor="middle" fill={INK.muted}>{s} s</T>)}
      <line x1={left} x2={W - right} y1={height - bottom} y2={height - bottom} stroke={AXIS} />
      {threshold != null && values.map((v, i) => v > threshold && <rect key={i} x={x(i) - 0.6} y={y(v)} width={1.4} height={Math.max(0, y(threshold) - y(v))} fill={color} opacity={0.35} />)}
      {mask && mask.map((on, i) => on && <rect key={i} x={x(i - offset) - 0.7} y={top} width={1.6} height={height - top - bottom} fill={color} opacity={0.22} />)}
      <path d={path(values, x, y)} fill="none" stroke={color} strokeWidth={1.6} strokeLinejoin="round" />
      {pattern && pattern.points.map((p, i) => <circle key={i} cx={x(p.i - offset)} cy={y(p.v)} r={4.5} fill={p.plus ? POS : NEG} stroke="#0C1424" strokeWidth={1.5} />)}
      {pattern && <T x={x(pattern.points[4].i - offset)} y={top + 10} textAnchor="middle" fill={INK.primary}>best match</T>}
      {threshold != null && <g><line x1={left} x2={W - right} y1={y(threshold)} y2={y(threshold)} stroke={INK.primary} strokeDasharray="5 4" /><T x={W - right} y={y(threshold) - 5} textAnchor="end" fill={INK.primary}>threshold</T></g>}
    </svg>
  )
}

// Power of the channel at each frequency, with the two bands the statistics read.
export function Spectrum({ freqs, power, bands, dominant, color, height = 180 }) {
  const { ref, bind, node } = useTip()
  const W = 620, H = height, left = 40, right = 10, top = 22, bottom = 28
  const hi = Math.max(...power) * 1.08
  const x = (f) => left + (f / 20) * (W - left - right)
  const y = (v) => top + (1 - v / hi) * (H - top - bottom)
  const bw = (W - left - right) / freqs.length
  return (
    <div ref={ref} className="msr-chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Power of one channel by frequency">
        <rect x={x(bands.low[0])} y={top} width={x(bands.low[1]) - x(bands.low[0])} height={H - top - bottom} fill="#3987e5" opacity={0.14} />
        <rect x={x(bands.high[0])} y={top} width={x(bands.high[1]) - x(bands.high[0])} height={H - top - bottom} fill="#c98500" opacity={0.10} />
        <T x={x(bands.low[0]) + 2} y={top - 8} fill="#3987e5">low band 0.5–3 Hz</T>
        <T x={x(bands.high[0]) + 40} y={top - 8} fill="#c98500">high band 3–20 Hz</T>
        {[0, 5, 10, 15, 20].map((f) => <T key={f} x={x(f)} y={H - 10} textAnchor="middle" fill={INK.muted}>{f} Hz</T>)}
        <line x1={left} x2={W - right} y1={H - bottom} y2={H - bottom} stroke={AXIS} />
        {freqs.map((f, i) => (
          <g key={f} {...bind(<Tip value={`${f} Hz`} label={`power ${num(power[i], 1)}`} />)}>
            <rect x={x(f) - bw / 2 + 0.5} y={y(power[i])} width={Math.max(1, bw - 1)} height={H - bottom - y(power[i])} fill={color} opacity={Math.abs(f - dominant) < 0.01 ? 1 : 0.55} />
            <rect x={x(f) - bw / 2} y={top} width={bw} height={H - top - bottom} fill="transparent" />
          </g>
        ))}
        <T x={x(dominant) + 8} y={top + 12} fill={INK.primary}>strongest: {dominant} Hz</T>
      </svg>
      {node}
    </div>
  )
}

export function KernelWeights({ weights }) {
  return (
    <svg viewBox="0 0 180 56" width="200" role="img" aria-label="The 9 weights of the kernel">
      <line x1={0} x2={180} y1={34} y2={34} stroke={AXIS} />
      {weights.map((w, i) => <g key={i}><rect x={i * 20 + 2} y={w > 0 ? 34 - 24 : 34} width={16} height={w > 0 ? 24 : 12} rx={3} fill={w > 0 ? POS : NEG} /><T x={i * 20 + 10} y={w > 0 ? 8 : 52} textAnchor="middle" fontSize={10}>{w > 0 ? '+2' : '−1'}</T></g>)}
    </svg>
  )
}
