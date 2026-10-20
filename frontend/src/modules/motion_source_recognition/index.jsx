import { useEffect, useState } from 'react'
import { useModuleData } from '../../api/replay.js'
import { Notice } from '../../components/ui.jsx'
import { MODULES } from '../registry.js'
import { CLASS_COLOR, CLASS_LETTER, CLASS_TEXT, pct, num, shorten } from './viz/theme.js'
import { ChartCard, Segmented, Tabs } from './viz/ui.jsx'
import { Reading, readingFor } from './viz/reading.jsx'
import { Line, Spectrum, KernelWeights, WindowPicture } from './viz/features.jsx'
import { StructureDiagram, Replay, ExamplePicker, useReplay } from './viz/anatomy.jsx'
import { ClassBars, ConfusionHeat, ExperimentList, FitCurve, HBars, LimitList, OutcomeBars, RoomBars, ScorePlot, SetupCompare, Timeline, TradeoffChart, TuningChart } from './viz/charts.jsx'
import { DEFAULT_TRAINING_MODEL, PLAY_INTERVAL_MS, STAT_TEXT } from './config.js'
import './motion.css'

// Page of this module. Everything it shows comes from data/replay.json (written by backend evaluation/export_replay.py); the charts are in viz/.
const SUMMARY = 'Which kind of source is moving: Human, Pet, IRobot or Fan? Unknown when evidence is weak.'

const STAGES = [
  { key: 'preprocessing', title: 'Data preprocessing', kicker: 'STAGE 1' },
  { key: 'training', title: 'Model training', kicker: 'STAGE 2' },
  { key: 'experiments', title: 'Experiments & comparison', kicker: 'STAGE 3' },
  { key: 'evaluation', title: 'Evaluation', kicker: 'STAGE 4' },
]

function Stage({ id, n, kicker, title, plan, children }) {
  const [open, setOpen] = useState(false)
  const done = (plan || []).filter((p) => p.s === 'done').length
  return (
    <section className="stage" id={id}>
      <div className="stage-head">
        <div className="stage-num">{n}</div>
        <div>
          <div className="kicker dim" style={{ padding: 0 }}>{kicker}</div>
          <h2>{title}</h2>
        </div>
      </div>
      {plan?.length > 0 && (
        <div className="msr-stage-tools">
          <button type="button" className="msr-plan-btn" aria-expanded={open} onClick={() => setOpen(!open)}>PLAN · {done}/{plan.length} DONE {open ? '▴' : '▾'}</button>
        </div>
      )}
      {open && <ul className="msr-plan-list">{plan.map((p, i) => <li key={i}>{p.s === 'done' ? '✓ ' : '· '}{p.t}</li>)}</ul>}
      {children}
    </section>
  )
}

function Preview({ data, dash }) {
  const frames = dash.preview_frames
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (!playing) return undefined
    const id = setInterval(() => setIdx((i) => (i + 1) % frames.length), PLAY_INTERVAL_MS)
    return () => clearInterval(id)
  }, [playing, frames.length])
  const f = frames[Math.min(idx, frames.length - 1)]
  const unknown = f.label === 'unknown'
  const lean = f.probs && Object.entries(f.probs).sort((a, b) => b[1] - a[1])[0]
  const right = f.label === f.truth
  const madeUp = Boolean(f.note)
  const realPerClass = frames.filter((x) => !x.note).length / new Set(frames.filter((x) => !x.note).map((x) => x.truth)).size
  const madeUpCount = frames.filter((x) => x.note).length
  const key = (e) => {
    if (e.key === 'ArrowRight') { setPlaying(false); setIdx((i) => Math.min(frames.length - 1, i + 1)) }
    if (e.key === 'ArrowLeft') { setPlaying(false); setIdx((i) => Math.max(0, i - 1)) }
  }
  return (
    <section className="card glow" onKeyDown={key}>
      <div className="sec-head">
        <h3>OUTPUT PREVIEW</h3>
        <span className="mono dim">window {idx + 1} of {frames.length} · t = {f.t} s</span>
      </div>
      <div className="msr-top">
        <div className="msr-answer">
          <div className="msr-sub">The module answers</div>
          <div className="msr-big">{f.label}<small>confidence {pct(f.confidence)}</small></div>
          <div className="msr-chips">
            <span className={`msr-pill ${f.quality?.level || 'ok'}`}>quality: {f.quality?.level || 'ok'}</span>
            {madeUp
              ? <span className="msr-verdict">made-up window: {f.note}</span>
              : <span className={`msr-verdict ${unknown ? 'unknown' : right ? 'right' : 'wrong'}`}>true class {f.truth} · {unknown ? 'answered unknown' : right ? 'right ✓' : 'wrong ✕'}</span>}
          </div>
          {f.quality?.reasons?.length > 0 && <div className="msr-hint">Reason: {f.quality.reasons.join(', ')}. {f.quality.level === 'invalid' ? 'The model is not run.' : 'The answer is less reliable.'}</div>}
          {unknown && lean && <div className="msr-hint">Best guess {lean[0]} ({pct(lean[1])}), not sure enough.</div>}
        </div>
        <div>
          <div className="msr-sub">Probability of each class</div>
          {f.probs ? <ClassBars probs={f.probs} label={f.label} /> : <div className="msr-nop">No probabilities: the window is not usable, so the answer is unknown with confidence 0.</div>}
        </div>
      </div>
      <Timeline frames={frames} index={idx} onSelect={(i) => { setPlaying(false); setIdx(i) }} playing={playing} onPlay={() => setPlaying(!playing)} />
      <div className="msr-hint" style={{ marginTop: 10 }}>{realPerClass} real windows per class, then {madeUpCount} made-up bad windows to show the quality levels. Click one, or use ← →.</div>
    </section>
  )
}

// The data in numbers, on one card: size and window shape on top, then how the classes and the rooms are spread. All from replay.json (table, data tiles, room_class).
function DataProfile({ dash, data }) {
  const demo = dash.feature_demo
  const table = data.preprocessing.table.map((r) => ({ ...r, n: Number(String(r.Windows).replace(/,/g, '')) }))
  const tile = (label) => dash.data_tiles.find((t) => t.label === label)?.value
  const total = table.reduce((s, r) => s + r.n, 0)
  const rooms = dash.room_class.map((r) => ({ room: r.room, windows: r.cells.reduce((s, c) => s + c.windows, 0), cells: r.cells }))
  const maxClass = Math.max(...table.map((r) => r.n)), maxRoom = Math.max(...rooms.map((r) => r.windows))
  return (
    <section className="card msr-card msr-profile">
      <div className="msr-headline">
        {[['WINDOWS', tile('WINDOWS')], ['RECORDINGS', tile('RECORDINGS')], ['ROOMS', tile('ROOMS')], ['INDIVIDUALS', table.reduce((s, r) => s + r.Individuals, 0)]].map(([l, v]) => (
          <div key={l}><b>{v}</b><span>{l}</span></div>
        ))}
        <p>Each window: <b>{demo.steps / demo.sampling_rate_hz} s</b> of Wi-Fi amplitude, <b>{demo.channels} channels</b> ({demo.antenna_links} antenna links × {demo.channels / demo.antenna_links}) × <b>{demo.steps} steps</b></p>
      </div>
      <div className="msr-pcols">
        <div>
          <div className="msr-pgroup-title">WINDOWS PER CLASS</div>
          {table.map((r) => (
            <div className="msr-brow" key={r.Class} title={`${r.Recordings} recordings in ${r.Rooms} rooms`}>
              <span>{r.Class}</span>
              <div className="msr-btrack"><i style={{ width: `${(r.n / maxClass) * 100}%`, background: CLASS_COLOR[r.Class] }} /></div>
              <b>{r.Windows}</b>
              <em>{pct(r.n / total)}</em>
              <small>{r.Individuals} individuals</small>
            </div>
          ))}
        </div>
        <div>
          <div className="msr-pgroup-title">WINDOWS PER ROOM · H Human, P Pet, R IRobot, F Fan</div>
          {rooms.map((r) => (
            <div className="msr-brow room" key={r.room}>
              <span>{r.room}</span>
              <div className="msr-btrack"><i style={{ width: `${(r.windows / maxRoom) * 100}%`, background: '#5b7aa6' }} /></div>
              <b>{r.windows.toLocaleString()}</b>
              <div className="msr-room-chips">
                {r.cells.map((c) => <span key={c.class} className={c.windows ? '' : 'off'} style={c.windows ? { background: CLASS_COLOR[c.class], color: CLASS_TEXT[c.class] } : undefined} title={c.windows ? `${c.class}: ${c.windows.toLocaleString()} windows, ${c.individuals} individuals` : `${c.class}: not recorded`}>{CLASS_LETTER[c.class]}</span>)}
              </div>
            </div>
          ))}
        </div>
      </div>
      {dash.insight && <div className="msr-insight"><b>!</b><span>{dash.insight}</span></div>}
    </section>
  )
}

// Two ways to turn one window into a row of numbers. One real window per class; the pictures are drawn from backend evaluation/feature_demo.py.
function FeatureEngineering({ demo }) {
  const [ci, setCi] = useState(0)
  const [ki, setKi] = useState(0)
  const d = demo.classes[ci]
  const perLink = demo.channels / demo.antenna_links
  const color = CLASS_COLOR[d.class]
  const k = d.kernels[ki]
  const matches = Array.from({ length: d.signal.length }, (_, i) => (k.output[i - k.offset] ?? -Infinity) > k.threshold)
  const t0 = k.output.indexOf(Math.max(...k.output)) + k.offset
  const mean = d.signal.reduce((a, v) => a + v, 0) / d.signal.length
  const amp = Math.sqrt(d.signal.reduce((a, v) => a + (v - mean) ** 2, 0) / d.signal.length)
  const best = { points: k.weights.map((w, j) => ({ i: Math.min(d.signal.length - 1, Math.max(0, t0 + (j - 4) * k.dilation)), v: mean + (w > 0 ? 1 : -0.5) * amp, plus: w > 0 })) }
  return (
    <div>
      <section className="card msr-card">
        <div className="msr-card-head">
          <div><h3>Start: one window</h3><div className="msr-hint">{(demo.channels * demo.steps).toLocaleString()} numbers per window: too many, so each window becomes a shorter row of numbers.</div></div>
          <div className="msr-card-tools"><ExamplePicker value={ci} onChange={setCi} label="Example window" /></div>
        </div>
        <div className="msr-start">
          <div>
            <div className="msr-replay-title">The whole window</div>
            <WindowPicture picture={d.picture} mark={demo.channel} />
            <div className="msr-hint">Rows: {demo.antenna_links} links × {perLink} subcarriers. Columns: time. Pink: below average. Purple: above.</div>
          </div>
          <div>
            <div className="msr-replay-title">Channel {demo.channel} = link {Math.floor(demo.channel / perLink) + 1}, subcarrier {(demo.channel % perLink) + 1}</div>
            <Line values={d.signal} color={color} label={`Channel ${demo.channel} over 5 seconds`} />
            <div className="msr-hint">Both methods work on lines like this, channel by channel.</div>
          </div>
        </div>
      </section>
      <div className="msr-grid2 even msr-methods" style={{ marginTop: 16 }}>
        <section className="card msr-card">
          <div className="msr-card-head"><div><h3>Method 1 · Statistics</h3><div className="msr-hint">Summarise each channel with 10 numbers.</div></div></div>
          <div className="msr-replay-title">1 · How strong is each rhythm?</div>
          <Spectrum freqs={demo.frequencies} power={d.power} bands={demo.bands} dominant={d.statistics.dominant_freq} color={color} height={215} />
          <div className="msr-replay-title" style={{ marginTop: 10 }}>2 · Read 10 numbers from the channel</div>
          <div className="msr-stats">
            {STAT_TEXT.map(([key, text]) => <div key={key}><span>{text}</span><b>{num(d.statistics[key], Math.abs(d.statistics[key]) >= 100 ? 0 : 2)}</b></div>)}
          </div>
          <div className="msr-result">3 · {demo.statistics_per_channel} numbers × {demo.channels} channels = <b>{demo.statistics_features.toLocaleString()}</b> numbers per window.</div>
        </section>
        <section className="card msr-card">
          <div className="msr-card-head">
            <div><h3>Method 2 · MiniRocket</h3><div className="msr-hint">Count how often the signal looks like a small pattern.</div></div>
            <div className="msr-card-tools"><Segmented label="Pattern" value={ki} onChange={setKi} options={d.kernels.map((x, i) => ({ value: i, label: `Pattern ${x.name}` }))} /></div>
          </div>
          <div className="msr-replay-title">1 · The pattern: high at +, low at −</div>
          <div className="msr-pattern-row">
            <KernelWeights weights={k.weights} />
            <div className="msr-hint">{k.dilation === 1 ? '9 neighbouring steps.' : `9 steps, ${k.dilation} apart.`}</div>
          </div>
          <div className="msr-replay-title" style={{ marginTop: 10 }}>2 · Slide it along the signal</div>
          <Line values={d.signal} color={color} mask={matches} pattern={best} height={140} label={`Signal with the places where pattern ${k.name} matches`} />
          <div className="msr-replay-title" style={{ marginTop: 10 }}>3 · Match score at each position</div>
          <Line values={k.output} offset={k.offset} total={d.signal.length} color={color} threshold={k.threshold} height={100} label={`Match score of pattern ${k.name}`} />
          <div className="msr-result">Strong at <b>{pct(k.share_above)}</b> of positions = <b>one</b> of the <b>{demo.minirocket_features.toLocaleString()}</b> numbers per window.</div>
        </section>
      </div>
    </div>
  )
}

function Training({ dash }) {
  const views = dash.training_view
  const [key, setKey] = useState(DEFAULT_TRAINING_MODEL)
  const view = views.find((v) => v.key === key) || views[0]
  return <TrainingModel key={view.key} view={view} views={views} onKey={setKey} />
}

function TrainingModel({ view, views, onKey }) {
  const [mode, setMode] = useState({ forest: 'macro_f1', linear: 'macro_f1' })
  const r = useReplay(view)
  const splits = Object.keys(view.tuning)
  const forest = view.kind === 'forest'
  const m = mode[view.kind]
  const fmt = (v) => (view.kind === 'linear' && (v < 0.01 || v >= 1000) ? v.toExponential(0).replace('e+', 'e') : String(v))
  const tuningTable = splits.flatMap((k) => view.tuning[k].map((p) => ({ Split: k.replace('_', ' '), [view.param.label]: fmt(p.value), 'Macro-F1': num(p.macro_f1, 3), Accuracy: num(p.accuracy, 3), 'Time (s)': Math.round(p.seconds),
    ...(forest ? {} : { Iterations: p.iterations, Converged: p.converged ? 'yes' : 'no' }), Chosen: p.value === view.chosen[k] ? '✓' : '' })))
  const curveTable = splits.flatMap((k) => view.curve[k].map((p) => (forest
    ? { Split: k.replace('_', ' '), Trees: p.trees, 'Macro-F1': num(p.macro_f1, 3), Accuracy: num(p.accuracy, 3) }
    : { Split: k.replace('_', ' '), Iterations: p.iterations, 'Validation loss': num(p.val_log_loss, 3), 'Training loss': num(p.train_log_loss, 3), 'Macro-F1': num(p.macro_f1, 3) })))
  const options = forest ? [{ value: 'macro_f1', label: 'Macro-F1' }, { value: 'accuracy', label: 'Accuracy' }] : [{ value: 'macro_f1', label: 'Macro-F1' }, { value: 'val_log_loss', label: 'Loss' }]
  return (
    <div>
      <Tabs value={view.key} onChange={onKey} tabs={views.map((v) => ({ value: v.key, label: shorten(v.name) + (v.optimistic ? ' *' : '') }))} />
      <div className="msr-blurb"><b>{view.name}</b><span>{view.features} · {view.blurb}</span></div>

      <div className="msr-sub2">A · HOW THE MODEL COMPUTES</div>
      <section className="card msr-card">
        <div className="msr-card-head">
          <div><h3>{forest ? 'One decision tree' : 'How a window becomes an answer'}</h3><div className="msr-hint">{forest ? 'A window answers one question per box, going down the tree.' : 'Each class has its own formula. Training finds the weights in them.'}</div></div>
          {forest && <div className="msr-card-tools"><ExamplePicker value={r.ex} onChange={r.setEx} /></div>}
        </div>
        <StructureDiagram view={view} ex={r.ex} />
      </section>

      <div className="msr-sub2">B · TRAINING: WHAT GETS ADJUSTED</div>
      <div className="msr-grid2 even">
        <section className="card msr-card">
          <div className="msr-card-head">
            <div><h3>{forest ? 'Trees voting' : 'Weights and probabilities'}</h3><div className="msr-hint">{forest ? `${view.anatomy.settings.trees} trees are grown. Press play to add them one by one.` : 'Weights start at zero. Press play to watch them change.'} {view.anatomy.split.replace('split_', 'Split ')}, saved model.</div></div>
            <div className="msr-card-tools"><ExamplePicker value={r.ex} onChange={r.setEx} /></div>
          </div>
          <Replay view={view} r={r} />
        </section>
        <ChartCard title={forest ? 'Score while the forest grows' : 'Score while the weights are fitted'} hint={forest ? 'Validation score when only the first n trees vote. The line marks the replay.' : 'Refitted at the chosen C, stopped at each step. The line marks the replay.'} table={curveTable}
          controls={<Segmented label="Show" value={m} onChange={(v) => setMode({ ...mode, [view.kind]: v })} options={options} />}>
          <FitCurve view={view} mode={m} at={r.at} />
        </ChartCard>
      </div>

      <div className="msr-sub2">SETTINGS · WHAT EACH ONE DOES</div>
      <div className="msr-params">
        {view.parameters.map((p, i) => (
          <div className="msr-param" key={p.name}>
                        <div>
              <b>{p.name}</b>
              <span className={`msr-param-tag ${p.tuned ? 'tuned' : ''}`}>{p.tuned ? 'tuned' : 'fixed'}</span>
              <div className="msr-param-where">{p.where}</div>
              <div>{p.effect}</div>
              <div className="msr-param-value">{settingValue(view, p, i, fmt, splits)}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="msr-sub2">C · TUNING: TRAIN EACH CANDIDATE, SCORE IT, KEEP THE BEST</div>
      <ol className="msr-steps">
        <li><b>1</b>Pick candidate values of {forest ? 'the smallest leaf' : 'C'}</li>
        <li><b>2</b>Train one model per value, on the training windows</li>
        <li><b>3</b>Score each model on the validation windows</li>
        <li><b>4</b>Keep the value with the best score</li>
      </ol>
      <ChartCard title={`Scores of the candidates (${view.param.label})`} hint={view.param.help} table={tuningTable}>
        <TuningChart view={view} />
      </ChartCard>

      <div className="msr-final">
        <div className="msr-final-title">D · FINAL SETTINGS</div>
        <div className="msr-tiles three">
          {splits.map((k) => (
            <div className="tile" key={k}>
              <div className="tile-label">{k.replace('_', ' ').toUpperCase()}</div>
              <div className="tile-value">{view.param.name === 'C' ? `C = ${fmt(view.chosen[k])}` : `leaf ${view.chosen[k]}`}</div>
              <div className="tile-note">
                {forest ? `${view.anatomy.settings.trees} trees` : `${view.fits[k].iterations} iterations ${view.fits[k].converged ? '✓ converged' : '(stopped at the limit)'}`} · macro-F1 {num(view.fits[k].validation_macro_f1, 3)}
              </div>
            </div>
          ))}
          <div className="tile">
            <div className="tile-label">TRAINED ON</div>
            <div className="tile-value">{view.fits.split_0.train_windows.toLocaleString()}</div>
            <div className="tile-note">windows · {view.fits.split_0.features.toLocaleString()} features · {view.megabytes} MB</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// The value a setting has in this model: the tuned one per split, or the fixed one.
function settingValue(view, p, i, fmt, splits) {
  const s = view.anatomy.settings
  if (p.tuned) return `${view.param.name === 'C' ? 'C = ' : ''}${splits.map((k) => fmt(view.chosen[k])).join(' · ')}${view.param.name === 'C' ? '' : ' windows'} (${splits.map((k) => k.replace('split_', '')).join(' · ')})`
  if (view.kind === 'forest') return [null, `${s.trees} trees`, `${s.features_per_split} of ${s.features.toLocaleString()} features`, Object.entries(s.class_weights).map(([c, w]) => `${c} ×${w}`).join(' · ')][i]
  return [null, `at most ${s.max_iter} iterations`, 'all features', Object.entries(s.class_weights).map(([c, w]) => `${c} ×${w}`).join(' · ')][i]
}

// A chart with its short reading on the right.
function EvalRow({ reading, children }) {
  return (
    <div className="msr-evalrow">
      <div>{children}</div>
      {reading && <Reading reading={reading} />}
    </div>
  )
}

function Evaluation({ data, dash }) {
  const [tab, setTab] = useState('scores')
  const [withRule, setWithRule] = useState(false)
  const [cmMode, setCmMode] = useState('percent')
  const chosenIdx = Math.max(0, dash.tradeoff.findIndex((p) => Math.abs(p.share - dash.tradeoff_chosen_share) < 1e-9))
  const [idx, setIdx] = useState(chosenIdx)

  const scoreRows = [
    ...dash.models.map((m) => ({ name: m.name, macro_f1: m.test.macro_f1, accuracy: m.test.accuracy, chosen: m.chosen, optimistic: m.optimistic,
      lines: [`validation macro-F1 ${num(m.validation.macro_f1[0], 3)}, accuracy ${num(m.validation.accuracy[0], 3)}`, ...(m.optimistic ? ['features designed after looking at the whole dataset'] : [])] })),
    ...dash.references.map((r) => ({ name: r.name, macro_f1: r.test.macro_f1, accuracy: r.test.accuracy, reference: true })),
  ]
  const classTable = dash.outcome.map((o) => ({ Class: o.class, 'Right (every window answered)': pct(o.without_rule.right), 'Right (with unknown rule)': pct(o.with_rule.right), 'Wrong (with unknown rule)': pct(o.with_rule.wrong), 'Unknown (with unknown rule)': pct(o.with_rule.unknown) }))
  const cm = dash.confusion
  const cmTable = cm.matrix.map((row, i) => ({ 'True class': cm.labels[i], ...Object.fromEntries(cm.labels.map((l, j) => [`Answered ${l}`, row[j].toLocaleString()])) }))
  const roomsTable = [...dash.compare.map((c) => ({ Setting: c.name, Accuracy: num(c.value), Detail: c.detail })),
    ...dash.rooms.map((r) => ({ Setting: `Room ${r.room} left out`, Accuracy: num(r.accuracy), Detail: `${r.windows.toLocaleString()} windows: ${r.classes.join(', ')}` }))]
  const ruleTable = dash.unknown_rule.map((u) => ({ Split: u.split.replace('_', ' '), T: u.temperature, Threshold: u.threshold, Unknown: pct(u.unknown), 'Accuracy, all': num(u.accuracy_all, 3), 'Accuracy, answered': num(u.accuracy_answered, 3) }))
  const speedTable = dash.speed.map((s) => ({ Model: s.name, 'Whole path (ms)': s.ms, 'Size to ship (MB)': s.megabytes }))
  const maxMs = Math.max(...dash.speed.map((s) => s.ms))
  const reading = readingFor(tab, dash, data, idx)

  return (
    <div>
      <Tabs value={tab} onChange={setTab} tabs={[
        { value: 'scores', label: 'Test scores' }, { value: 'classes', label: 'Per class' }, { value: 'confusion', label: 'Confusion' }, { value: 'rooms', label: 'New rooms' },
        { value: 'unknown', label: 'Unknown rule' }, { value: 'speed', label: 'Speed' }, { value: 'limits', label: 'Limits & interface' }]} />
      {tab === 'scores' && <EvalRow reading={reading}>
        <ChartCard title="Test scores by model" hint={`Test rows. Mean ± spread over ${dash.n_splits} splits. ★ chosen.`} table={data.metrics.table}>
          <ScorePlot splits={dash.n_splits} rows={scoreRows} />
        </ChartCard>
      </EvalRow>}
      {tab === 'classes' && <EvalRow reading={reading}>
        <ChartCard title="What happens to the windows of each class" hint={`Main model, test rows, mean of ${dash.n_splits} splits.`} table={classTable}
          controls={<Segmented label="Unknown rule" value={withRule ? 'with' : 'without'} onChange={(v) => setWithRule(v === 'with')} options={[{ value: 'without', label: 'Every window answered' }, { value: 'with', label: 'With unknown rule' }]} />}>
          <OutcomeBars rows={dash.outcome} withRule={withRule} />
        </ChartCard>
      </EvalRow>}
      {tab === 'confusion' && <EvalRow reading={reading}>
        <ChartCard title="What each class is answered as" hint={`Main model, test rows, the ${dash.n_splits} splits added.`} table={cmTable}
          controls={<Segmented label="Show" value={cmMode} onChange={setCmMode} options={[{ value: 'percent', label: '% of row' }, { value: 'count', label: 'Windows' }]} />}>
          <ConfusionHeat labels={cm.labels} matrix={cm.matrix} mode={cmMode} />
        </ChartCard>
      </EvalRow>}
      {tab === 'rooms' && <EvalRow reading={reading}>
        <ChartCard title="Known rooms against a new room" hint="Accuracy on the test windows. A room left out also takes away its people, pets and devices." table={roomsTable}>
          <SetupCompare items={[
            { title: 'Rooms and people the model has seen', sentence: 'Trained on part of the recordings of every room, tested on the rest.', value: dash.compare[0].value, chance: 0.25, lines: [dash.compare[0].detail] },
            { title: 'A room it has never seen', sentence: 'One whole room, with its people, pets and devices, is held out of the training and used as the test. All 10 rooms in turn.', value: dash.compare[2].value, chance: 0.25, leftOut: true, highlight: true, lines: [dash.compare[2].detail] },
          ]} />
          <div className="msr-sub2">Room by room</div>
          <div className="msr-hint" style={{ margin: '0 0 8px' }}>Each bar: that room is the test, the other 9 are for learning. Squares: the classes recorded in it.</div>
          <RoomBars rooms={dash.rooms} />
        </ChartCard>
      </EvalRow>}
      {tab === 'unknown' && <EvalRow reading={reading}>
        <ChartCard title="Answer unknown when not sure" hint="Validation rows. Drag the slider; the rule uses the marked share." table={ruleTable}>
          <TradeoffChart points={dash.tradeoff} idx={idx} onIdx={setIdx} chosenShare={dash.tradeoff_chosen_share} />
        </ChartCard>
      </EvalRow>}
      {tab === 'speed' && <EvalRow reading={reading}>
        <ChartCard title="One window, raw signal to answer" hint="Milliseconds on one CPU thread. A window arrives every 5 000 ms." table={speedTable}>
          <HBars ariaLabel="Milliseconds per window" domain={[0, Math.ceil(maxMs / 10) * 10]} labelWidth={210} format={(v) => `${v.toFixed(0)} ms`}
            rows={dash.speed.map((s) => ({ name: shorten(s.name), value: s.ms, highlight: /Logistic Regression on MiniRocket/.test(s.name), lines: [`${s.megabytes} MB to ship`] }))} />
        </ChartCard>
      </EvalRow>}
      {tab === 'limits' && (
        <div className="msr-grid2">
          <ChartCard title="Limits" hint="Click a line for the detail."><LimitList items={dash.limits} /></ChartCard>
          <ChartCard title="What the other modules get" hint="One frame per window.">
            <dl className="msr-fields">{dash.fields.map((f) => <div key={f.field}><dt>{f.field}</dt><dd>{f.meaning}</dd></div>)}</dl>
          </ChartCard>
        </div>
      )}
    </div>
  )
}

export default function MotionSourceRecognition() {
  const mod = MODULES.find((m) => m.id === 'motion_source_recognition')
  const { data, error, loading } = useModuleData(mod)
  const dash = data?.dashboard
  const go = (key) => document.getElementById(`${mod.id}-${key}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  const plan = mod.plan
  const ready = data && dash
  return (
    <div className="mpage">
      <header className="mhead">
        <div className="mhead-text">
          <div className="kicker">SENSING MODULE {mod.order} OF {MODULES.length}</div>
          <h1>{mod.title}</h1>
          <p className="lead">{SUMMARY}</p>
        </div>
        <span className="state-pill" style={{ color: data?.mock ? '#FFB347' : '#22E4FF', background: data?.mock ? 'rgba(255,179,71,.12)' : 'rgba(34,228,255,.10)' }}>
          {loading ? 'LOADING' : error ? 'NO DATA' : data?.mock ? 'PLACEHOLDER RESULTS' : 'LIVE DATA'}
        </span>
      </header>
      {error && <Notice kind="warn">Couldn't load this module's data ({error.message}).</Notice>}
      {data?.mock && <Notice kind="warn">Results and replay frames are placeholders.</Notice>}
      {ready && (
        <>
          <Preview data={data} dash={dash} />
          <nav className="stagenav" aria-label="Stages">
            {STAGES.map((s, i) => <button key={s.key} type="button" onClick={() => go(s.key)}><b>{i + 1}</b>{s.title}</button>)}
          </nav>

          <Stage id={`${mod.id}-preprocessing`} n={1} kicker="STAGE 1" title="Data preprocessing" plan={plan.preprocessing}>
            <div className="msr-sub2">PART 1 · DATA PROFILE</div>
            <DataProfile dash={dash} data={data} />
            <div className="msr-sub2">PART 2 · FEATURE ENGINEERING</div>
            <FeatureEngineering demo={dash.feature_demo} />
          </Stage>

          <Stage id={`${mod.id}-training`} n={2} kicker="STAGE 2" title="Model training" plan={plan.training}>
            <Training dash={dash} data={data} />
          </Stage>

          <Stage id={`${mod.id}-experiments`} n={3} kicker="STAGE 3" title="Experiments & comparison" plan={plan.experiments}>
            <ChartCard title="What was tried" hint="Click a line for the question and the numbers." table={data.experiments.table}>
              <ExperimentList cards={data.experiments.cards} />
            </ChartCard>
          </Stage>

          <Stage id={`${mod.id}-evaluation`} n={4} kicker="STAGE 4" title="Evaluation" plan={plan.evaluation}>
            <Evaluation data={data} dash={dash} />
          </Stage>
        </>
      )}
    </div>
  )
}
