import { useState } from 'react'
import { CLASSES, CLASS_COLOR, CLASS_LETTER, CLASS_TEXT, UNKNOWN_COLOR, SERIES, STATUS, SURFACE, HAIR, AXIS, INK, ramp, rampText, pct, num, shorten } from './theme.js'
import { Bullet } from './reading.jsx'
import { Legend, Tip, useTip } from './ui.jsx'

const FONT = 'var(--font)'
const T = (props) => <text fontFamily={FONT} fontSize={11} fill={INK.secondary} {...props} />

// ---------- the replay ----------

// The 24 windows of the replay, grouped by the true class; the cell shows the answer (colour and letter) and whether it is right.
const MADE_UP = 'made-up'   // grouping key of the frames that carry a `note` (the damaged windows)
const groupOf = (f) => (f.note ? MADE_UP : f.truth)

export function Timeline({ frames, index, onSelect, playing, onPlay }) {
  const { ref, bind, node } = useTip()
  // The legend lights up only the entries that describe the selected window.
  const cur = frames[index]
  const level = cur?.quality?.level, madeUp = Boolean(cur?.note), unknown = cur?.label === 'unknown'
  const legend = [
    ...CLASSES.map((c) => ({ label: `${CLASS_LETTER[c]} ${c}`, color: CLASS_COLOR[c], dim: cur?.label !== c })),
    { label: '? unknown', color: UNKNOWN_COLOR, dim: !unknown },
    { label: '✓ right', color: 'transparent', textColor: STATUS.good, dim: madeUp || unknown || cur?.label !== cur?.truth },
    { label: '✕ wrong', color: 'transparent', textColor: STATUS.critical, dim: madeUp || unknown || cur?.label === cur?.truth },
    { label: '! degraded', color: 'transparent', textColor: STATUS.warn, dim: level !== 'degraded' },
    { label: '⊘ invalid', color: 'transparent', textColor: STATUS.critical, dim: level !== 'invalid' },
  ]
  const groups = [...CLASSES, MADE_UP].map((c) => ({ c, items: frames.map((f, i) => ({ f, i })).filter((x) => groupOf(x.f) === c) })).filter((g) => g.items.length)
  return (
    <div className="msr-chart" ref={ref}>
      <div className="msr-tl-head">
        <button type="button" className="msr-play" onClick={onPlay} aria-label={playing ? 'Pause the replay' : 'Play the replay'}>{playing ? '❚❚ Pause' : '▶ Play'}</button>
        <Legend items={legend} />
      </div>
      <div className="msr-tl">
        {groups.map(({ c, items }) => (
          <div className="msr-tl-group" key={c}>
            <div className="msr-tl-label"><i style={{ background: c === MADE_UP ? INK.muted : CLASS_COLOR[c] }} />{c === MADE_UP ? 'made-up bad windows' : `true class: ${c}`}</div>
            <div className="msr-tl-cells">
              {items.map(({ f, i }) => {
                const unknown = f.label === 'unknown'
                const right = f.label === f.truth
                const level = f.quality?.level
                const madeUp = Boolean(f.note)
                const mark = level === 'invalid' ? ['⊘', STATUS.critical] : madeUp ? ['!', STATUS.warn] : unknown ? ['–', INK.secondary] : right ? ['✓', STATUS.good] : ['✕', STATUS.critical]
                const style = unknown ? { background: 'repeating-linear-gradient(45deg, #3b4a63 0 4px, #2a3550 4px 8px)', color: INK.primary } : { background: CLASS_COLOR[f.label], color: CLASS_TEXT[f.label] }
                return (
                  <button key={i} type="button" className={`msr-cell ${i === index ? 'on' : ''}`} style={style} onClick={() => onSelect(i)} aria-label={`t = ${f.t} s, answered ${f.label}, true class ${f.truth}`}
                    {...bind(<Tip value={`${f.label} ${pct(f.confidence)}`} label={`t = ${f.t} s`} lines={madeUp ? [f.note, `quality: ${level}`] : [`true class: ${f.truth}`, unknown ? 'answered unknown' : right ? 'answer is right' : 'answer is wrong']} />)}>
                    <b>{unknown ? '?' : CLASS_LETTER[f.label]}</b>
                    <span style={{ color: mark[1] }}>{mark[0]}</span>
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
      {node}
    </div>
  )
}

export function ClassBars({ probs, label }) {
  return (
    <div className="msr-pbars">
      {CLASSES.map((c) => (
        <div className="msr-pbar" key={c}>
          <span className="name"><i style={{ background: CLASS_COLOR[c] }} />{c}</span>
          <span className="track"><i style={{ width: `${Math.round((probs?.[c] || 0) * 100)}%`, background: CLASS_COLOR[c], opacity: label === c || label === 'unknown' ? 1 : 0.75 }} /></span>
          <span className="val">{Math.round((probs?.[c] || 0) * 100)}%</span>
        </div>
      ))}
    </div>
  )
}

// ---------- data ----------

// Rooms by classes: colour is the number of windows, the digit is the number of people, pets or devices recorded there.
// ---------- scores ----------

// Macro-F1 and accuracy side by side, one row per model, with both numbers written out on every row.
export function ScorePlot({ rows, splits }) {
  const { ref, bind, node } = useTip()
  const W = 760, rowH = 36, top = 34, label = 232, pw = 150, vw = 92, gap = 14
  const ax = label + 8, bx = ax + pw + vw + gap
  const H = top + rows.length * rowH + 26
  const panels = [{ key: 'macro_f1', title: 'Macro-F1', x0: ax, domain: [0, 0.7], ticks: [0, 0.25, 0.5] }, { key: 'accuracy', title: 'Accuracy', x0: bx, domain: [0, 1], ticks: [0, 0.5, 1] }]
  const xs = (pn, v) => pn.x0 + ((v - pn.domain[0]) / (pn.domain[1] - pn.domain[0])) * pw
  return (
    <div className="msr-chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Test macro-F1 and accuracy of every model, mean and spread over ${splits} splits`}>
        {panels.map((pn) => (
          <g key={pn.key}>
            <T x={pn.x0} y={16} fill={INK.primary} fontWeight={600}>{pn.title}</T>
            <T x={pn.x0 + pw + vw - 4} y={16} textAnchor="end" fill={INK.muted}>mean ± spread</T>
            {pn.ticks.map((v) => (
              <g key={v}>
                <line x1={xs(pn, v)} x2={xs(pn, v)} y1={top - 6} y2={top + rows.length * rowH} stroke={HAIR} />
                <T x={xs(pn, v)} y={top + rows.length * rowH + 16} textAnchor="middle" fill={INK.muted}>{num(v)}</T>
              </g>
            ))}
          </g>
        ))}
        {rows.map((r, i) => {
          const cy = top + i * rowH + rowH / 2
          const color = r.reference ? UNKNOWN_COLOR : CLASS_COLOR.Human
          return (
            <g key={r.name} {...bind(<Tip value={shorten(r.name) + (r.optimistic ? ' *' : '')} label={`test rows, mean ± spread of ${splits} splits`} lines={[`macro-F1 ${num(r.macro_f1[0], 3)} ± ${num(r.macro_f1[1], 3)}`, `accuracy ${num(r.accuracy[0], 3)} ± ${num(r.accuracy[1], 3)}`, ...(r.lines || [])]} />)}>
              <rect x={0} y={top + i * rowH} width={W} height={rowH} fill={r.chosen ? 'rgba(57,135,229,0.12)' : 'transparent'} />
              <T x={label} y={cy + 4} textAnchor="end" fill={r.chosen ? INK.primary : INK.secondary} fontWeight={r.chosen ? 600 : 400}>{shorten(r.name)}{r.optimistic ? ' *' : ''}{r.chosen ? ' ★' : ''}</T>
              {panels.map((pn) => {
                const [m, sd] = r[pn.key]
                return (
                  <g key={pn.key}>
                    <line x1={xs(pn, Math.max(pn.domain[0], m - sd))} x2={xs(pn, Math.min(pn.domain[1], m + sd))} y1={cy} y2={cy} stroke={INK.muted} strokeWidth={2} strokeLinecap="round" />
                    <circle cx={xs(pn, m)} cy={cy} r={6} fill={r.optimistic ? SURFACE : color} stroke={r.optimistic ? color : SURFACE} strokeWidth={2} />
                    <T x={pn.x0 + pw + vw - 4} y={cy + 4} textAnchor="end" fill={INK.primary} fontWeight={r.chosen ? 700 : 400}>{num(m)} ± {num(sd)}</T>
                  </g>
                )
              })}
              <rect x={0} y={top + i * rowH} width={W} height={rowH} fill="transparent" />
            </g>
          )
        })}
      </svg>
      <Legend items={[{ label: `mean ± spread over ${splits} splits`, color: CLASS_COLOR.Human }, { label: 'hollow dot, *: optimistic', color: SURFACE, ring: CLASS_COLOR.Human }, { label: 'reference line', color: UNKNOWN_COLOR }]} />
      {node}
    </div>
  )
}

const OUTCOME = [['right', 'right', STATUS.good, '✓'], ['wrong', 'answered as another class', STATUS.critical, '✕'], ['unknown', 'answered unknown', UNKNOWN_COLOR, '–']]

// What happens to the windows of each class: right, answered as another class, answered unknown (100% bars).
export function OutcomeBars({ rows, withRule }) {
  const { ref, bind, node } = useTip()
  const W = 640, rowH = 44, top = 8, left = 120, right = 12, H = top + rows.length * rowH + 8
  const pw = W - left - right
  return (
    <div className="msr-chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="What happens to the windows of each class">
        {rows.map((r, i) => {
          const o = withRule ? r.with_rule : r.without_rule
          const cy = top + i * rowH + rowH / 2
          let x = left
          return (
            <g key={r.class}>
              <rect x={6} y={cy - 5} width={10} height={10} rx={3} fill={CLASS_COLOR[r.class]} />
              <T x={24} y={cy + 4} fill={INK.primary} fontWeight={600}>{r.class}</T>
              {OUTCOME.map(([key, text, color, icon]) => {
                const w = o[key] * pw
                const x0 = x
                x += w
                if (w < 0.5) return null
                return (
                  <g key={key} {...bind(<Tip value={pct(o[key], 0)} label={`of the ${r.class} windows: ${text}`} />)}>
                    <rect x={x0 + 1} y={cy - 13} width={Math.max(0, w - 2)} height={26} rx={4} fill={color} opacity={key === 'unknown' ? 0.85 : 1} />
                    {w >= 46 && <text x={x0 + w / 2} y={cy + 4} textAnchor="middle" fontFamily={FONT} fontSize={12} fontWeight={700} fill="#04101C">{icon} {pct(o[key])}</text>}
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>
      <Legend items={OUTCOME.map(([, text, color]) => ({ label: text, color }))} />
      {node}
    </div>
  )
}

// The two situations side by side, each with a small picture of what was trained on and what was tested, and the share of windows answered correctly.
export function SetupCompare({ items }) {
  const { ref, bind, node } = useTip()
  return (
    <div className="msr-chart msr-setup" ref={ref}>
      {items.map((it) => (
        <div className="msr-setup-row" key={it.title} {...bind(<Tip value={`${Math.round(it.value * 100)} of 100 right`} label={it.title} lines={it.lines || []} />)}>
          <div className="msr-setup-text">
            <div className="msr-setup-title">{it.title}</div>
            <div className="msr-setup-rooms" aria-label={it.leftOut ? 'nine rooms for training, one room left out as the test' : 'all rooms used for training and testing'}>
              {Array.from({ length: 10 }, (_, i) => <i key={i} className={it.leftOut && i === 9 ? 'test' : 'train'} />)}
              <span>{it.leftOut ? '9 rooms to learn from, 1 room as the test' : 'every room to learn from; the test uses other recordings of them'}</span>
            </div>
            <div className="msr-setup-sentence">{it.sentence}</div>
          </div>
          <div className="msr-setup-bar">
            <div className="msr-setup-track">
              <i className={it.highlight ? 'hl' : ''} style={{ width: `${it.value * 100}%` }} />
              <b style={{ left: `${it.chance * 100}%` }} />
            </div>
            <div className="msr-setup-value"><strong>{Math.round(it.value * 100)}</strong> of 100 windows right</div>
          </div>
        </div>
      ))}
      <div className="msr-setup-note"><b /> the vertical line is guessing among the 4 classes: 25 of 100</div>
      {node}
    </div>
  )
}

// One bar per room left out: the share of its windows answered correctly; the squares show which classes the room holds.
export function RoomBars({ rooms }) {
  const { ref, bind, node } = useTip()
  const W = 620, rowH = 30, top = 8, left = 168, right = 64, H = top + rooms.length * rowH + 22
  const x = (v) => left + v * (W - left - right)
  return (
    <div className="msr-chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Accuracy for each room left out of the training">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}><line x1={x(v)} x2={x(v)} y1={top} y2={top + rooms.length * rowH} stroke={HAIR} /><T x={x(v)} y={top + rooms.length * rowH + 15} textAnchor="middle" fill={INK.muted}>{num(v)}</T></g>
        ))}
        {rooms.map((r, i) => {
          const cy = top + i * rowH + rowH / 2
          const w = Math.max(2, x(r.accuracy) - x(0))
          return (
            <g key={r.room} {...bind(<Tip value={`${pct(r.accuracy)} right`} label={`room ${r.room} left out`} lines={[`${r.windows.toLocaleString()} windows`, ...Object.entries(r.recall).map(([c, v]) => `${c}: ${pct(v)} right`)]} />)}>
              <T x={8} y={cy + 4} fill={INK.primary} fontWeight={600}>{r.room}</T>
              {CLASSES.map((c, k) => {
                const here = r.classes.includes(c)
                return (
                  <g key={c}>
                    <rect x={46 + k * 28} y={cy - 9} width={24} height={18} rx={4} fill={here ? CLASS_COLOR[c] : 'transparent'} stroke={here ? 'none' : HAIR} />
                    <text x={46 + k * 28 + 12} y={cy + 4} textAnchor="middle" fontFamily={FONT} fontSize={11} fontWeight={700} fill={here ? CLASS_TEXT[c] : INK.muted} opacity={here ? 1 : 0.5}>{CLASS_LETTER[c]}</text>
                  </g>
                )
              })}
              <path d={`M${x(0)},${cy - 8} H${x(0) + w - 4} Q${x(0) + w},${cy - 8} ${x(0) + w},${cy - 4} V${cy + 4} Q${x(0) + w},${cy + 8} ${x(0) + w - 4},${cy + 8} H${x(0)} Z`} fill={CLASS_COLOR.Human} />
              <T x={x(0) + w + 8} y={cy + 4} fill={INK.primary}>{num(r.accuracy)}</T>
              <rect x={0} y={top + i * rowH} width={W} height={rowH} fill="transparent" />
            </g>
          )
        })}
      </svg>
      <Legend items={CLASSES.map((c) => ({ label: `${CLASS_LETTER[c]} ${c} in the room`, color: CLASS_COLOR[c] }))} />
      {node}
    </div>
  )
}

export function ConfusionHeat({ labels, matrix, mode }) {
  const { ref, bind, node } = useTip()
  const cell = 74, left = 84, top = 58, W = left + cell * 4 + 10, H = top + cell * 4 + 12
  const rowSum = matrix.map((r) => r.reduce((a, b) => a + b, 0) || 1)
  return (
    <div className="msr-chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }} role="img" aria-label="Confusion matrix of the main model on the test rows">
        <T x={left + (cell * 4) / 2} y={14} textAnchor="middle" fill={INK.muted}>answered as</T>
        {labels.map((l, j) => <T key={l} x={left + j * cell + cell / 2} y={top - 10} textAnchor="middle" fill={INK.primary}>{l}</T>)}
        {matrix.map((row, i) => (
          <g key={labels[i]}>
            <T x={left - 10} y={top + i * cell + cell / 2 + 4} textAnchor="end" fill={INK.primary}>{labels[i]}</T>
            {row.map((v, j) => {
              const f = v / rowSum[i]
              return (
                <g key={j} {...bind(<Tip value={`${v.toLocaleString()} windows`} label={`true ${labels[i]} answered as ${labels[j]}`} lines={[`${pct(f, 1)} of the ${labels[i]} windows`]} />)}>
                  <rect x={left + j * cell + 1} y={top + i * cell + 1} width={cell - 2} height={cell - 2} rx={5} fill={ramp(0.05 + 0.95 * f)} stroke={i === j ? INK.primary : 'none'} strokeWidth={i === j ? 1 : 0} strokeOpacity={0.5} />
                  <text x={left + j * cell + cell / 2} y={top + i * cell + cell / 2 + 4} textAnchor="middle" fontFamily={FONT} fontSize={13} fontWeight={600} fill={rampText(0.05 + 0.95 * f)}>{mode === 'percent' ? pct(f) : v.toLocaleString()}</text>
                </g>
              )
            })}
          </g>
        ))}
        <T x={14} y={top + (cell * 4) / 2} textAnchor="middle" fill={INK.muted} transform={`rotate(-90 14 ${top + (cell * 4) / 2})`}>true class</T>
      </svg>
      {node}
    </div>
  )
}

// ---------- new rooms, speed ----------

// Horizontal bars of one measure. `highlight` rows are drawn in the accent hue, the others in a quiet grey-blue (emphasis, not identity).
export function HBars({ rows, domain = [0, 1], reference: refLine, format = (v) => num(v), labelWidth = 170, ariaLabel }) {
  const { ref, bind, node } = useTip()
  const W = 620, rowH = 30, top = 10, right = 70, H = top + rows.length * rowH + (refLine ? 26 : 8)
  const x = (v) => labelWidth + ((v - domain[0]) / (domain[1] - domain[0])) * (W - labelWidth - right)
  return (
    <div className="msr-chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={ariaLabel}>
        <line x1={x(domain[0])} x2={x(domain[0])} y1={top} y2={top + rows.length * rowH} stroke={AXIS} />
        {refLine && (
          <g>
            <line x1={x(refLine.value)} x2={x(refLine.value)} y1={top} y2={top + rows.length * rowH} stroke={INK.muted} strokeWidth={1} />
            <T x={x(refLine.value)} y={top + rows.length * rowH + 16} textAnchor="middle" fill={INK.muted}>{refLine.label}</T>
          </g>
        )}
        {rows.map((r, i) => {
          const cy = top + i * rowH + rowH / 2, w = Math.max(2, x(r.value) - x(domain[0]))
          return (
            <g key={r.name} {...bind(<Tip value={r.format ? r.format(r.value) : format(r.value)} label={r.name} lines={r.lines || []} />)}>
              <T x={labelWidth - 10} y={cy + 4} textAnchor="end" fill={r.highlight ? INK.primary : INK.secondary} fontWeight={r.highlight ? 600 : 400}>{r.name}</T>
              <path d={`M${x(domain[0])},${cy - 9} H${x(domain[0]) + w - 4} Q${x(domain[0]) + w},${cy - 9} ${x(domain[0]) + w},${cy - 5} V${cy + 5} Q${x(domain[0]) + w},${cy + 9} ${x(domain[0]) + w - 4},${cy + 9} H${x(domain[0])} Z`} fill={r.highlight ? CLASS_COLOR.Human : '#3b4a63'} />
              <T x={x(domain[0]) + w + 8} y={cy + 4} fill={INK.primary}>{r.format ? r.format(r.value) : format(r.value)}</T>
              <rect x={0} y={top + i * rowH} width={W} height={rowH} fill="transparent" />
            </g>
          )
        })}
      </svg>
      {node}
    </div>
  )
}

// ---------- training ----------

const splitColor = (k) => SERIES[Number(k.split('_')[1]) % SERIES.length]
const fmtValue = (v, kind) => (kind === 'linear' ? (v >= 0.01 && v < 1000 ? String(v) : v.toExponential(0).replace('e-', 'e-').replace('e+', 'e')) : String(v))
const niceRange = (lo, hi, pad = 0.03, step = 0.05) => [Math.max(0, Math.floor((lo - pad) / step) * step), Math.ceil((hi + pad) / step) * step]

function Plot({ W = 620, H = 290, left = 52, right = 22, top = 16, bottom = 56, xs, ys, xTicks, yTicks, xFormat, yFormat, xLabel, yLabel, children }) {
  const px = (v) => left + xs(v) * (W - left - right)
  const py = (v) => top + (1 - ys(v)) * (H - top - bottom)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${yLabel} against ${xLabel}`}>
      {yTicks.map((v) => <g key={v}><line x1={left} x2={W - right} y1={py(v)} y2={py(v)} stroke={HAIR} /><T x={left - 8} y={py(v) + 4} textAnchor="end" fill={INK.muted}>{yFormat(v)}</T></g>)}
      {xTicks.map((v) => <T key={v} x={px(v)} y={H - bottom + 18} textAnchor="middle" fill={INK.muted}>{xFormat(v)}</T>)}
      <line x1={left} x2={W - right} y1={H - bottom} y2={H - bottom} stroke={AXIS} />
      <T x={(left + W - right) / 2} y={H - 8} textAnchor="middle" fill={INK.secondary}>{xLabel}</T>
      <T x={14} y={(top + H - bottom) / 2} textAnchor="middle" fill={INK.secondary} transform={`rotate(-90 14 ${(top + H - bottom) / 2})`}>{yLabel}</T>
      {children(px, py)}
    </svg>
  )
}

const SplitLegend = ({ splits, extra = [] }) => (
  <Legend items={[...splits.map((k) => ({ label: k.replace('_', ' '), color: splitColor(k), line: true })), ...extra]} />
)

// The values of the setting that was tried (log scale) against the validation macro-F1, one line per split. Ring: the value that was chosen. Hollow: the fit stopped at the iteration limit.
export function TuningChart({ view }) {
  const { ref, bind, node } = useTip()
  const splits = Object.keys(view.tuning)
  const all = splits.flatMap((k) => view.tuning[k])
  const values = [...new Set(all.map((r) => r.value))].sort((a, b) => a - b)
  const lx = (v) => Math.log10(v)
  const x0 = lx(values[0]), x1 = lx(values[values.length - 1])
  const [ylo, yhi] = niceRange(Math.min(...all.map((r) => r.macro_f1)), Math.max(...all.map((r) => r.macro_f1)))
  const yTicks = []
  for (let v = ylo; v <= yhi + 1e-9; v += 0.05) yTicks.push(Number(v.toFixed(2)))
  const linear = view.kind === 'linear'
  return (
    <div className="msr-chart" ref={ref}>
      <Plot xs={(v) => (x1 === x0 ? 0.5 : (lx(v) - x0) / (x1 - x0))} ys={(v) => (v - ylo) / (yhi - ylo)} xTicks={values} yTicks={yTicks} xFormat={(v) => fmtValue(v, view.kind)} yFormat={(v) => num(v)}
        xLabel={`${view.param.label} (log scale)`} yLabel="validation macro-F1">
        {(px, py) => splits.map((k) => {
          const rows = view.tuning[k]
          return (
            <g key={k}>
              {rows.length > 1 && <polyline fill="none" stroke={splitColor(k)} strokeWidth={2} strokeLinejoin="round" points={rows.map((r) => `${px(r.value)},${py(r.macro_f1)}`).join(' ')} />}
              {rows.map((r) => {
                const chosen = r.value === view.chosen[k]
                const hollow = linear && r.converged === false
                return (
                  <g key={r.value} {...bind(<Tip value={num(r.macro_f1, 3)} label={`${k.replace('_', ' ')} · ${view.param.name} ${fmtValue(r.value, view.kind)}${chosen ? ' · chosen' : ''}`}
                    lines={[`accuracy ${num(r.accuracy, 3)}`, `fit time ${Math.round(r.seconds)} s`, ...(linear ? [`${r.iterations} iterations${r.converged ? ', converged' : ', stopped at the limit'}`] : [])]} />)}>
                    {chosen && <circle cx={px(r.value)} cy={py(r.macro_f1)} r={9} fill="none" stroke={INK.primary} strokeWidth={2} />}
                    <circle cx={px(r.value)} cy={py(r.macro_f1)} r={5} fill={hollow ? SURFACE : splitColor(k)} stroke={hollow ? splitColor(k) : SURFACE} strokeWidth={2} />
                    <circle cx={px(r.value)} cy={py(r.macro_f1)} r={13} fill="transparent" />
                  </g>
                )
              })}
            </g>
          )
        })}
      </Plot>
      <SplitLegend splits={splits} extra={[{ label: 'ring: chosen', color: SURFACE, ring: INK.primary }, ...(linear ? [{ label: 'hollow: stopped at the iteration limit', color: SURFACE, ring: INK.secondary }] : [])]} />
      {node}
    </div>
  )
}

// How the fit develops: a forest as trees are added, a Logistic Regression as its weights are adjusted.
export function FitCurve({ view, mode, at }) {
  const { ref, bind, node } = useTip()
  const splits = Object.keys(view.curve)
  const forest = view.kind === 'forest'
  const xKey = forest ? 'trees' : 'iterations'
  const all = splits.flatMap((k) => view.curve[k])
  const xmax = Math.max(...all.map((p) => p[xKey]))
  const lx = (v) => Math.log10(Math.max(1, v))
  const values = all.map((p) => p[mode])
  const lossMode = mode === 'val_log_loss'
  const [ylo, yhi] = lossMode ? niceRange(Math.min(...values), Math.max(...values), 0.02, 0.1) : niceRange(Math.min(...values), Math.max(...values), 0.02, 0.05)
  const step = lossMode ? 0.2 : 0.05
  const yTicks = []
  for (let v = ylo; v <= yhi + 1e-9; v += step) yTicks.push(Number(v.toFixed(2)))
  const xTicks = forest ? [1, 3, 10, 30, 100, 300] : [0, 50, 100, 150, 200, 250, 300]
  const xs = (v) => (forest ? lx(v) / lx(xmax) : v / xmax)
  const label = { macro_f1: 'validation macro-F1', accuracy: 'validation accuracy', val_log_loss: 'validation loss (lower is better)' }[mode]
  return (
    <div className="msr-chart" ref={ref}>
      <Plot xs={xs} ys={(v) => (v - ylo) / (yhi - ylo)} xTicks={xTicks} yTicks={yTicks} xFormat={(v) => String(v)} yFormat={(v) => num(v)} xLabel={forest ? 'trees that vote (log scale)' : 'iterations'} yLabel={label}>
        {(px, py) => [at != null && <line key="at" x1={px(at)} x2={px(at)} y1={py(yhi)} y2={py(ylo)} stroke={INK.secondary} strokeWidth={1.5} strokeDasharray="4 3" />, ...splits.map((k) => {
          const pts = view.curve[k]
          const last = pts[pts.length - 1]
          const converged = !forest && view.fits[k].converged
          return (
            <g key={k}>
              <polyline fill="none" stroke={splitColor(k)} strokeWidth={2} strokeLinejoin="round" points={pts.map((p) => `${px(p[xKey])},${py(p[mode])}`).join(' ')} />
              {pts.map((p) => (
                <g key={p[xKey]} {...bind(<Tip value={mode === 'val_log_loss' ? num(p.val_log_loss, 3) : num(p[mode], 3)} label={`${k.replace('_', ' ')} · ${p[xKey]} ${forest ? 'trees' : 'iterations'}`}
                  lines={forest ? [`macro-F1 ${num(p.macro_f1, 3)}`, `accuracy ${num(p.accuracy, 3)}`] : [`validation loss ${num(p.val_log_loss, 3)}`, `training loss ${num(p.train_log_loss, 3)}`, `macro-F1 ${num(p.macro_f1, 3)}`]} />)}>
                  <circle cx={px(p[xKey])} cy={py(p[mode])} r={p === last && !forest ? 6 : 3.5} fill={p === last && !forest && !converged ? SURFACE : splitColor(k)} stroke={p === last && !forest && !converged ? splitColor(k) : SURFACE} strokeWidth={2} />
                  <circle cx={px(p[xKey])} cy={py(p[mode])} r={11} fill="transparent" />
                </g>
              ))}
            </g>
          )
        })]}
      </Plot>
      <SplitLegend splits={splits} extra={forest ? [] : [{ label: 'last point: filled = converged, hollow = stopped at the iteration limit', color: SURFACE, ring: INK.secondary }]} />
      {node}
    </div>
  )
}

// ---------- the unknown rule ----------

// Validation rows: what happens when more of the least confident windows are answered unknown. The slider moves along the curve.
export function TradeoffChart({ points, idx, onIdx, chosenShare }) {
  const { ref, bind, node } = useTip()
  const W = 640, H = 300, left = 44, right = 104, top = 16, bottom = 40
  const xmax = Math.max(...points.map((p) => p.share))
  const x = (v) => left + (v / xmax) * (W - left - right)
  const y = (v) => top + (H - top - bottom) * (1 - v)
  const series = [{ key: 'accuracy_kept', label: 'accuracy of the answered', short: 'accuracy', color: SERIES[0] }, { key: 'macro_f1_kept', label: 'macro-F1 of the answered', short: 'macro-F1', color: SERIES[1] }, { key: 'errors_rejected', label: 'errors turned into unknown', short: 'errors caught', color: SERIES[2] }]
  const cur = points[idx]
  const nearest = (clientX, rect) => {
    const px = ((clientX - rect.left) / rect.width) * W
    let best = 0
    points.forEach((p, i) => { if (Math.abs(x(p.share) - px) < Math.abs(x(points[best].share) - px)) best = i })
    return best
  }
  return (
    <div className="msr-chart" ref={ref}>
      <div className="msr-readout">
        <div><b>{pct(cur.share)}</b><span>of the windows answered unknown</span></div>
        {series.map((s) => <div key={s.key}><b style={{ borderBottom: `3px solid ${s.color}` }}>{s.key === 'accuracy_kept' || s.key === 'macro_f1_kept' ? num(cur[s.key]) : pct(cur[s.key])}</b><span>{s.label}</span></div>)}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Accuracy, macro-F1 and errors turned into unknown against the share of windows answered unknown (validation rows)"
        onPointerDown={(e) => onIdx(nearest(e.clientX, e.currentTarget.getBoundingClientRect()))}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => <g key={v}><line x1={left} x2={W - right} y1={y(v)} y2={y(v)} stroke={HAIR} /><T x={left - 8} y={y(v) + 4} textAnchor="end" fill={INK.muted}>{num(v)}</T></g>)}
        {points.map((p) => <T key={p.share} x={x(p.share)} y={H - 18} textAnchor="middle" fill={INK.muted}>{pct(p.share)}</T>)}
        <T x={(left + W - right) / 2} y={H - 3} textAnchor="middle" fill={INK.muted}>share of the windows answered unknown</T>
        {series.map((s) => (
          <g key={s.key}>
            <polyline fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" points={points.map((p) => `${x(p.share)},${y(p[s.key])}`).join(' ')} />
            {points.map((p, i) => <circle key={i} cx={x(p.share)} cy={y(p[s.key])} r={i === idx ? 6 : 4} fill={s.color} stroke={SURFACE} strokeWidth={2} />)}
            <T x={W - right + 8} y={y(points[points.length - 1][s.key]) + 4} fill={INK.secondary}>{s.short}</T>
          </g>
        ))}
        <line x1={x(cur.share)} x2={x(cur.share)} y1={top} y2={H - bottom} stroke={INK.primary} strokeOpacity={0.55} />
        <line x1={x(chosenShare)} x2={x(chosenShare)} y1={top} y2={top + 10} stroke={STATUS.warn} strokeWidth={3} />
        <T x={x(chosenShare) + 6} y={top + 9} fill={STATUS.warn}>chosen {pct(chosenShare)}</T>
        {points.map((p, i) => <rect key={i} x={x(p.share) - 14} y={top} width={28} height={H - top - bottom} fill="transparent" {...bind(<Tip value={`${pct(p.share)} unknown`} label="validation rows" lines={[`accuracy ${num(p.accuracy_kept)}`, `macro-F1 ${num(p.macro_f1_kept)}`, `errors turned into unknown ${pct(p.errors_rejected)}`]} />)} />)}
      </svg>
      <input className="msr-slider" type="range" min={0} max={points.length - 1} step={1} value={idx} onChange={(e) => onIdx(+e.target.value)} aria-label="Share of the windows answered unknown" />
      <Legend items={series.map((s) => ({ label: s.label, color: s.color, line: true }))} />
      {node}
    </div>
  )
}

// ---------- experiments, limits ----------

const VERDICT = { good: ['✓', STATUS.good], neutral: ['–', STATUS.neutral], warn: ['!', STATUS.warn], critical: ['✕', STATUS.critical], info: ['i', STATUS.info] }

export function ExperimentList({ cards }) {
  const [open, setOpen] = useState(null)
  return (
    <div className="msr-acc">
      {cards.map((c, i) => {
        const [icon, color] = VERDICT[c.tone] || VERDICT.neutral
        return (
          <div className={`msr-acc-row ${open === i ? 'open' : ''}`} key={c.name}>
            <button type="button" aria-expanded={open === i} onClick={() => setOpen(open === i ? null : i)}>
              <span className="name">{c.name}</span>
              <span className="verdict" style={{ color }}><b style={{ borderColor: color }}>{icon}</b>{c.verdict}</span>
              <span className="res">{c.result}</span>
              <span className="chev">{open === i ? '−' : '+'}</span>
            </button>
            {open === i && (
              <div className="msr-acc-body msr-exp">
                {[['why', 'WHY WE TRIED IT'], ['did', 'WHAT WE DID'], ['found', 'RESULT'], ['so', 'CONCLUSION']].map(([key, label]) => (
                  <section key={key}>
                    <div className="msr-reading-kicker">{label}</div>
                    <ul>{c[key].map((t, j) => <Bullet key={j} text={t} />)}</ul>
                  </section>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function LimitList({ items }) {
  const [open, setOpen] = useState(0)
  return (
    <div className="msr-acc limits">
      {items.map((l, i) => {
        const [icon, color] = VERDICT[l.tone] || VERDICT.info
        return (
          <div className={`msr-acc-row ${open === i ? 'open' : ''}`} key={l.title}>
            <button type="button" aria-expanded={open === i} onClick={() => setOpen(open === i ? null : i)}>
              <span className="verdict" style={{ color }}><b style={{ borderColor: color }}>{icon}</b></span>
              <span className="name">{l.title}</span>
              <span className="chev">{open === i ? '−' : '+'}</span>
            </button>
            {open === i && <div className="msr-acc-body">{l.detail}</div>}
          </div>
        )
      })}
    </div>
  )
}
