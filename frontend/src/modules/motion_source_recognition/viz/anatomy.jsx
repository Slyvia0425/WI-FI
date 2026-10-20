import { useEffect, useState } from 'react'
import { CLASSES, CLASS_COLOR, INK, SERIES, SURFACE, HAIR, num } from './theme.js'
import { PLAY_INTERVAL_MS } from '../config.js'
import { Segmented, Tip, useTip } from './ui.jsx'

// What is inside a model, and how it is trained: one real decision tree and the vote of the trees (Random Forest), or the weights and the formula (Logistic Regression).
// All numbers come from backend evaluation/model_anatomy.py (split 0, the saved models); the example windows are validation windows, one per class.

const FONT = 'var(--font)'
const T = (props) => <text fontFamily={FONT} fontSize={11} fill={INK.secondary} {...props} />
const top = (p) => p.indexOf(Math.max(...p))

export function useReplay(view) {
  const a = view.anatomy
  const steps = a.kind === 'forest' ? view.curve[a.split].map((p) => p.trees) : a.stages.map((s) => s.iterations)
  const [ex, setEx] = useState(0)
  const [step, setStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (!playing) return undefined
    const id = setInterval(() => setStep((s) => { if (s >= steps.length - 1) { setPlaying(false); return s } return s + 1 }), PLAY_INTERVAL_MS)
    return () => clearInterval(id)
  }, [playing, steps.length])
  const toggle = () => { if (!playing && step >= steps.length - 1) setStep(0); setPlaying(!playing) }
  return { ex, setEx, step, setStep, playing, toggle, steps, at: steps[step] }
}

export const ExamplePicker = ({ value, onChange, label = 'Example window, true class' }) => (
  <div className="msr-ex"><span>{label}</span><Segmented label="Example" value={value} onChange={onChange} options={CLASSES.map((c, i) => ({ value: i, label: c }))} /></div>
)

function MixBar({ x, y, w, dist, h = 8 }) {
  let at = x
  return dist.map((d, i) => { const ww = Math.max(0, d * w - 2); const r = <rect key={i} x={at} y={y} width={ww} height={h} rx={2} fill={CLASS_COLOR[CLASSES[i]]} />; at += d * w; return r })
}

// ---------- Random Forest: one tree ----------

const POS = { 0: [360, 18], 1: [180, 112], 2: [540, 112], 3: [90, 206], 4: [270, 206], 5: [450, 206], 6: [630, 206] }
const BW = 170, BH = 56
const POS_AT = (id) => POS[id]

function TreeDiagram({ a, ex }) {
  const { ref, bind, node } = useTip()
  const t = a.tree
  const byId = Object.fromEntries(t.nodes.map((n) => [n.id, n]))
  const path = new Set(t.paths[CLASSES[ex]])
  const color = CLASS_COLOR[CLASSES[ex]]
  return (
    <div className="msr-chart" ref={ref}>
      <svg viewBox="0 0 720 310" width="100%" role="img" aria-label="The top of one decision tree">
        {t.nodes.filter((n) => n.left != null).map((n) => [['left', 'yes: at most'], ['right', 'no: more']].map(([side, word]) => {
          const [px, py] = POS[n.id], [cx, cy] = POS[n[side]]
          const on = path.has(n.id) && path.has(n[side])
          return (
            <g key={n.id + side}>
              <line x1={px} y1={py + BH} x2={cx} y2={cy} stroke={on ? color : HAIR} strokeWidth={on ? 3 : 1.5} />
              <T x={(px + cx) / 2 + (side === 'left' ? -8 : 8)} y={(py + BH + cy) / 2 + 3} textAnchor={side === 'left' ? 'end' : 'start'} fill={INK.muted}>{side === 'left' ? 'yes' : 'no'}</T>
            </g>
          )
        }))}
        {t.nodes.map((n) => {
          const [x, y] = POS[n.id]
          const on = path.has(n.id)
          return (
            <g key={n.id} {...bind(<Tip value={n.feature} label={`goes left when at most ${n.threshold}`} lines={[`${n.samples.toLocaleString()} training windows reach this point`, ...CLASSES.map((c, i) => `${c} ${num(n.dist[i] * 100, 0)}%`), 'shares are class-weighted']} />)}>
              <rect x={x - BW / 2} y={y} width={BW} height={BH} rx={10} fill="rgba(255,255,255,.04)" stroke={on ? color : HAIR} strokeWidth={on ? 2.5 : 1.5} />
              <T x={x} y={y + 16} textAnchor="middle" fill={INK.primary} fontSize={10.5}>{n.feature}</T>
              <T x={x} y={y + 31} textAnchor="middle" fontSize={10.5}>≤ {n.threshold} ?</T>
              <MixBar x={x - BW / 2 + 10} y={y + 40} w={BW - 20} dist={n.dist} />
            </g>
          )
        })}
        {[3, 4, 5, 6].map((id) => { const [x, y] = POS[id]; return <T key={id} x={x} y={y + BH + 16} textAnchor="middle" fill={INK.muted}>⋮ splits further</T> })}
        <T x={360} y={300} textAnchor="middle" fill={INK.muted}>… until every leaf holds at least {a.settings.min_samples_leaf} windows. Colour bar: the class mix of the windows that reach the box.</T>
      </svg>
      {node}
    </div>
  )
}
const PoS = (id) => POS[id]

// ---------- Logistic Regression: the formula ----------

function Formula({ a }) {
  const last = a.stages[a.stages.length - 1]
  const term = (c, j) => ({ f: a.features[j], w: last.weights[c][j] })
  const rows = CLASSES.map((c, i) => ({ c, b: last.bias[i], terms: a.features.map((_, j) => term(c, j)).sort((x, y) => Math.abs(y.w) - Math.abs(x.w)).slice(0, 3) }))
  const sgn = (w) => (w < 0 ? '−' : '+')
  return (
    <div>
      <div className="msr-flow">
        <div className="msr-flow-box"><b>Window</b><span>{a.settings.features.toLocaleString()} numbers</span></div>
        <em>→</em>
        <div className="msr-flow-box"><b>Standardise</b><span>put every number on the same scale</span></div>
        <em>→</em>
        <div className="msr-flow-box"><b>4 scores</b><span>one formula per class</span></div>
        <em>→</em>
        <div className="msr-flow-box"><b>4 shares</b><span>the scores turned into percentages that add up to 100%</span></div>
      </div>
      <div className="msr-sub2">THE 4 FORMULAS (learned weights, {a.split.replace('_', ' ')})</div>
      <div className="msr-eq">
        {rows.map((r) => (
          <div className="msr-eq-row" key={r.c}>
            <b style={{ color: CLASS_COLOR[r.c] }}>{r.c} score</b>
            <code>= {num(r.b, 2)}{r.terms.map((t) => ` ${sgn(t.w)} ${num(Math.abs(t.w), 3)} × ${t.f}`).join('')} + … ({(a.settings.features - 3).toLocaleString()} more terms)</code>
          </div>
        ))}
      </div>
      <div className="msr-hint">Every class has its own weights, so its own formula. The window with the highest score wins. Only the 3 largest terms of each formula are shown.</div>
    </div>
  )
}

export function StructureDiagram({ view, ex }) {
  const a = view.anatomy
  return a.kind === 'forest' ? <TreeDiagram a={a} ex={ex} /> : <Formula a={a} />
}

// ---------- the replay ----------

function Bars({ probs, ex, answer }) {
  return (
    <div className="msr-vote">
      {CLASSES.map((c, i) => (
        <div className="msr-vote-row" key={c}>
          <span>{c}</span>
          <div className="msr-vote-track"><i style={{ width: `${Math.max(0.5, probs[i] * 100)}%`, background: CLASS_COLOR[c], opacity: i === answer ? 1 : 0.55 }} /></div>
          <b>{num(probs[i] * 100, 0)}%</b>
        </div>
      ))}
      <div className="msr-vote-answer">{CLASSES[answer]} {answer === ex ? <span className="ok">✓ right</span> : <span className="bad">✕ wrong</span>}</div>
    </div>
  )
}

function ForestReplay({ view, r }) {
  const a = view.anatomy
  const { ref, bind, node } = useTip()
  const trees = a.examples[r.ex].tree_probs
  const n = r.at
  const mean = CLASSES.map((_, c) => trees.slice(0, n).reduce((s, p) => s + p[c], 0) / n)
  const f1 = view.curve[a.split].find((p) => p.trees === n)
  return (
    <div className="msr-replay-body" ref={ref}>
      <div>
        <div className="msr-replay-title">Each tree votes; the forest averages the votes</div>
        <svg viewBox="0 0 520 190" width="100%" role="img" aria-label="Votes of the trees">
          {trees.map((p, i) => (
            <rect key={i} x={(i % 30) * 17.2} y={Math.floor(i / 30) * 17.2} width={15} height={15} rx={3} fill={CLASS_COLOR[CLASSES[top(p)]]} opacity={i < n ? 1 : 0.14}
              {...bind(<Tip value={`Tree ${i + 1}: ${CLASSES[top(p)]}`} label={i < n ? 'is voting' : 'has not voted yet'} lines={CLASSES.map((c, k) => `${c} ${num(p[k] * 100, 0)}%`)} />)} />
          ))}
        </svg>
        <div className="msr-hint">One square = one tree, coloured by its own answer. Lit squares have voted.</div>
      </div>
      <div>
        <div className="msr-replay-title">After {n} {n === 1 ? 'tree' : 'trees'}</div>
        <Bars probs={mean} ex={r.ex} answer={top(mean)} />
        {f1 && <div className="msr-hint">All validation windows, first {n} trees: macro-F1 {num(f1.macro_f1, 3)}</div>}
      </div>
      {node}
    </div>
  )
}

const POSNEG = [SERIES[0], SERIES[2]]
function weightFill(v, max) {
  const t = Math.min(1, Math.sqrt(Math.abs(v) / max))
  return v >= 0 ? `color-mix(in srgb, ${POSNEG[1]} ${Math.round(t * 100)}%, transparent)` : `color-mix(in srgb, ${POSNEG[0]} ${Math.round(t * 100)}%, transparent)`
}

function LinearReplay({ view, r }) {
  const a = view.anatomy
  const { ref, bind, node } = useTip()
  const st = a.stages[r.step]
  const max = Math.max(...a.stages.flatMap((s) => CLASSES.flatMap((c) => s.weights[c].map(Math.abs))))
  const probs = st.examples[r.ex]
  const k = a.features.length
  const cw = 34
  return (
    <div className="msr-replay-body stack" ref={ref}>
      <div>
        <div className="msr-replay-title">The weights of the 4 formulas, step by step</div>
        <svg viewBox={`0 0 ${70 + k * cw} ${CLASSES.length * 30 + 30}`} width="100%" role="img" aria-label="Weights of the largest features for each class">
          {CLASSES.map((c, i) => (
            <g key={c}>
              <T x={0} y={i * 30 + 19} fill={INK.primary}>{c}</T>
              {a.features.map((f, j) => {
                const v = st.weights[c][j]
                return <rect key={f} x={64 + j * cw} y={i * 30 + 4} width={cw - 3} height={26} rx={4} fill={weightFill(v, max)} stroke={HAIR} {...bind(<Tip value={num(v, 3)} label={`weight of ${c} on ${f}`} lines={[`after ${st.iterations} iterations`, 'plus: pushes towards this class; minus: away from it']} />)} />
              })}
            </g>
          ))}
          {a.features.map((f, j) => <T key={f} x={64 + j * cw + (cw - 3) / 2} y={CLASSES.length * 30 + 20} textAnchor="middle" fill={INK.muted} fontSize={10}>{j + 1}</T>)}
        </svg>
        <div className="msr-legend">
          <span><i style={{ background: POSNEG[0], width: 10, height: 10, borderRadius: 3 }} />negative weight</span>
          <span><i style={{ background: POSNEG[1], width: 10, height: 10, borderRadius: 3 }} />positive weight</span>
        </div>
        <div className="msr-hint">One row per class = one formula. Each column is one feature (the {k} with the largest weights; hover for the name). Stronger colour = larger weight. The weights belong to the model, so they are the same for every window; only the probabilities on the right depend on the example window.</div>
      </div>
      <div>
        <div className="msr-replay-title">After {st.iterations} {st.iterations === 1 ? 'iteration' : 'iterations'}</div>
        <Bars probs={probs} ex={r.ex} answer={top(probs)} />
        <div className="msr-hint">All validation windows: macro-F1 {num(st.macro_f1, 3)}</div>
      </div>
      {node}
    </div>
  )
}

export function Replay({ view, r }) {
  const forest = view.anatomy.kind === 'forest'
  return (
    <div>
      {forest ? <ForestReplay view={view} r={r} /> : <LinearReplay view={view} r={r} />}
      <div className="msr-replay-bar">
        <button type="button" className="msr-rplay" onClick={r.toggle} aria-label={r.playing ? 'Pause' : 'Play'}>{r.playing ? '❚❚' : '▶'}</button>
        <input type="range" min={0} max={r.steps.length - 1} value={r.step} onChange={(e) => { r.setStep(Number(e.target.value)); }} aria-label={forest ? 'Trees that have voted' : 'Iterations'} />
        <span className="mono">{r.at} {forest ? 'trees' : 'iterations'}</span>
      </div>
    </div>
  )
}
