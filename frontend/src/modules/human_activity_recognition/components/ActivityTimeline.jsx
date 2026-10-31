const COLORS = {
  jumping: '#22E4FF',
  running: '#FFB347',
  'seated-breathing': '#7CFFB2',
  walking: '#FF7AA2',
  wavinghand: '#C4A7FF',
  transition: '#FFD166',
  unknown: '#8B8FA8',
}

export default function ActivityTimeline({ frames = [] }) {
  if (!frames.length) return null
  const width = 640
  const height = 150
  const pad = 18
  const maxT = Math.max(...frames.map((frame) => frame.t), 1)
  const step = (width - pad * 2) / Math.max(frames.length - 1, 1)
  const points = frames.map((frame, index) => {
    const x = pad + index * step
    const confidence = Number(frame.confidence || 0)
    const y = height - pad - confidence * (height - pad * 2)
    return { x, y, frame }
  })
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="Activity confidence timeline">
      <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} stroke="var(--border-strong)" />
      <polyline fill="none" stroke="var(--accent)" strokeWidth="2" points={points.map((point) => `${point.x},${point.y}`).join(' ')} />
      {points.map((point, index) => (
        <g key={`${point.frame.t}-${index}`}>
          <circle cx={point.x} cy={point.y} r="5" fill={COLORS[point.frame.activity] || COLORS.unknown} />
          <text x={point.x} y={height - 4} textAnchor="middle" fontSize="10" fill="var(--muted)">
            {Math.round(point.frame.t)}s
          </text>
        </g>
      ))}
      <text x={pad} y={12} fontSize="11" fill="var(--muted)">confidence · max t {Math.round(maxT)}s</text>
    </svg>
  )
}
