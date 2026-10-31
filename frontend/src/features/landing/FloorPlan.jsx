import { FLOOR } from './floorplan.js'
import { TONES } from '../fusion/rules.js'

const MONO = 'JetBrains Mono, monospace'

function Rings({ x, y, delays }) {
  return delays.map((d) => (
    <circle key={d} cx={x} cy={y} r="20" fill="none" stroke={TONES.normal}>
      <animate attributeName="r" from="20" to="260" dur="4s" repeatCount="indefinite" begin={`${d}s`} />
      <animate attributeName="stroke-opacity" from="0.55" to="0" dur="4s" repeatCount="indefinite" begin={`${d}s`} />
    </circle>
  ))
}

export default function FloorPlan({ frames, fused }) {
  const { motionSource: A, activity: B, space: E } = frames
  const { box, router, node, fan, furniture, door, zones } = FLOOR
  const zone = E && zones[E.zone]
  const show = fused.humanGate && zone
  const col = TONES[fused.tone]
  const label = fused.tone === 'alert' ? 'FALL ALERT'
    : fused.tone === 'suspected' ? 'POSSIBLE FALL'
    : (B && B.activity && B.activity !== 'none' ? B.activity.toUpperCase() : 'PRESENT')
  // During a fall state the label also says what the person was doing, so that the story reads: standing up, then a possible fall.
  const sub = (fused.tone === 'alert' || fused.tone === 'suspected') && B && B.activity && B.activity !== 'none' ? B.activity.toUpperCase() : null
  const left = show && zone.x > 440
  const fanOn = A?.label === 'Fan'
  return (
    <svg viewBox="0 0 640 440" style={{ width: '100%', height: 'auto', display: 'block' }} role="img"
      aria-label="Top-down room map with a WiFi router, a CSI receiver, labelled zones and a person marker">
      <defs>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="rgba(120,160,255,0.08)" /></pattern>
        <clipPath id="room"><rect x={box.x} y={box.y} width={box.w} height={box.h} rx="14" /></clipPath>
      </defs>
      <rect x={box.x} y={box.y} width={box.w} height={box.h} rx="14" fill="url(#grid)" />
      <g clipPath="url(#room)"><Rings x={router.x} y={router.y} delays={[0, 1.33, 2.66]} /></g>
      <rect x={box.x} y={box.y} width={box.w} height={box.h} rx="14" fill="none" stroke={TONES.normal} strokeOpacity="0.55" strokeWidth="1.5" />
      <path d={`M${door.x} ${box.y + box.h}H${door.x + door.w}`} stroke="#0C1424" strokeWidth="5" />
      <path d={`M${door.x} ${box.y + box.h}A${door.w} ${door.w} 0 0 1 ${door.x + door.w} ${box.y + box.h - door.w}`} fill="none" stroke="#9FB3CC" strokeOpacity="0.4" strokeDasharray="3 5" />
      {furniture.map((f) => (
        <g key={f.name}>
          <rect x={f.x} y={f.y} width={f.w} height={f.h} rx="10" fill="rgba(139,92,246,0.14)" stroke="#8B5CF6" strokeOpacity="0.7" strokeDasharray="5 5" />
          <text x={f.x + 14} y={f.y + 22} fill="#B79BFF" fontFamily={MONO} fontSize="11">{f.name}</text>
        </g>
      ))}
      <text x={door.x + 19} y={box.y + box.h - 18} fill="#9FB3CC" fontFamily={MONO} fontSize="10">DOOR</text>
      <path d={`M${router.x} ${router.y}L${node.x} ${node.y}`} stroke={TONES.normal} strokeOpacity="0.4" strokeDasharray="4 6" />
      <rect x={router.x - 16} y={router.y - 8} width="32" height="16" rx="4" fill="#0C1424" stroke={TONES.normal} strokeWidth="1.5" />
      <text x={router.x - 30} y={router.y + 28} fill={TONES.normal} fontFamily={MONO} fontSize="10">ROUTER</text>
      <rect x={node.x - 8} y={node.y - 8} width="16" height="16" rx="3" fill="#0C1424" stroke={TONES.normal} strokeWidth="1.5" />
      <text x={node.x - 42} y={node.y + 26} fill={TONES.normal} fontFamily={MONO} fontSize="10">CSI RX</text>
      <circle cx={fan.x} cy={fan.y} r="14" fill="none" stroke={fanOn ? TONES.normal : '#9FB3CC'} strokeOpacity={fanOn ? 1 : 0.4} strokeWidth={fanOn ? 2 : 1}>
        {fanOn && <animate attributeName="r" from="14" to="24" dur="1.6s" repeatCount="indefinite" />}
      </circle>
      <text x={fan.x} y={fan.y + 4} textAnchor="middle" fill="#9FB3CC" fontFamily={MONO} fontSize="9">FAN</text>
      {show && (
        <g style={{ transform: `translate(${zone.x}px, ${zone.y}px)`, transition: 'transform 1s ease' }}>
          <circle r="18" fill="none" stroke={col} strokeOpacity="0.5">
            <animate attributeName="r" from="10" to="28" dur="2s" repeatCount="indefinite" />
            <animate attributeName="stroke-opacity" from="0.6" to="0" dur="2s" repeatCount="indefinite" />
          </circle>
          <circle r="7" fill={col} style={{ transition: 'fill .4s' }} />
          <rect x={left ? -126 : 22} y={sub ? -44 : -28} width="104" height={sub ? 42 : 26} rx="13" fill={col} fillOpacity="0.12" stroke={col} strokeOpacity="0.6" />
          <text x={left ? -74 : 74} y={sub ? -26 : -11} textAnchor="middle" fill={col} fontFamily={MONO} fontSize="11">{label}</text>
          {sub && <text x={left ? -74 : 74} y="-10" textAnchor="middle" fill="#9FB3CC" fontFamily={MONO} fontSize="10">{sub}</text>}
        </g>
      )}
      <text x="604" y="426" textAnchor="end" fill="#9FB3CC" fillOpacity="0.6" fontFamily={MONO} fontSize="9">SCHEMATIC POSITIONS</text>
    </svg>
  )
}

