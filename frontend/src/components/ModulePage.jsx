import { useState } from 'react'
import { MODULES } from '../modules/registry.js'
import { useModuleData } from '../api/replay.js'
import { Notice } from './ui.jsx'
import { MetricTable, ConfusionMatrix } from './charts.jsx'

const STAGES = [
  { key: 'preprocessing', title: 'Data preprocessing', kicker: 'STAGE 1' },
  { key: 'training', title: 'Model training', kicker: 'STAGE 2' },
  { key: 'experiments', title: 'Experiments & comparison', kicker: 'STAGE 3' },
  { key: 'evaluation', title: 'Evaluation', kicker: 'STAGE 4' },
]

const STATUS = { planned: 'PLANNED', doing: 'IN PROGRESS', done: 'DONE' }

function Plan({ items }) {
  if (!items?.length) return null
  return (
    <ul className="plan">
      {items.map((it, i) => (
        <li key={i}>
          <span className={`st ${it.s}`}>{STATUS[it.s] || STATUS.planned}</span>
          <span>{it.t}</span>
        </li>
      ))}
    </ul>
  )
}

function Tiles({ items }) {
  return (
    <div className="tiles">
      {items.map((t) => (
        <div className="tile" key={t.label}>
          <div className="tile-label">{t.label}</div>
          <div className="tile-value">{t.value}</div>
        </div>
      ))}
    </div>
  )
}

function Empty({ what }) {
  return <div className="empty">{what}</div>
}

// Optional starting point for a module page. Shows the module's current output, then
// the four stages every module shares: data preprocessing, model training,
// experiments and comparison, evaluation. Each stage has a plan checklist, data-driven
// blocks read from the module JSON, and a `stages[key]` slot for your own JSX.
//
// Props: id (the module folder name), summary, renderFrame(frame, ctx), stages { preprocessing, training, experiments, evaluation }.
// Plan: src/modules/<id>/plan.js. Results blocks read these optional keys of src/modules/<id>/data/replay.json:
//   preprocessing { tiles, table }, training { table }, experiments { table }, metrics { table, confusion }.
export default function ModulePage({ id, summary, renderFrame, stages = {} }) {
  const mod = MODULES.find((m) => m.id === id)
  const { data, error, loading } = useModuleData(mod)
  const [idx, setIdx] = useState(0)
  const plan = mod.plan

  const frames = data?.frames ?? []
  const frame = frames[Math.min(idx, frames.length - 1)] ?? null

  const go = (key) => document.getElementById(`${id}-${key}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  const blocks = {
    preprocessing: (
      <>
        {data?.preprocessing?.tiles && <Tiles items={data.preprocessing.tiles} />}
        {data?.preprocessing?.table
          ? <div className="tblwrap"><MetricTable rows={data.preprocessing.table} /></div>
          : !data?.preprocessing?.tiles && <Empty what={`Dataset facts appear here. Add "preprocessing": { "tiles": [...], "table": [...] } to src/modules/${id}/data/replay.json`} />}
      </>
    ),
    training: data?.training?.table
      ? <div className="tblwrap"><MetricTable rows={data.training.table} /></div>
      : <Empty what={`Training runs appear here. Add "training": { "table": [...] } to src/modules/${id}/data/replay.json`} />,
    experiments: data?.experiments?.table
      ? <div className="tblwrap"><MetricTable rows={data.experiments.table} /></div>
      : <Empty what={`Model comparison appears here. Add "experiments": { "table": [...] } to src/modules/${id}/data/replay.json`} />,
    evaluation: (
      <>
        {data?.metrics?.table && <div className="tblwrap"><MetricTable rows={data.metrics.table} /></div>}
        {data?.metrics?.confusion && <div className="tblwrap"><ConfusionMatrix labels={data.metrics.confusion.labels} matrix={data.metrics.confusion.matrix} /></div>}
        {!data?.metrics && <Empty what={`Metrics and confusion matrix appear here. Add "metrics": { "table": [...], "confusion": {...} } to src/modules/${id}/data/replay.json`} />}
      </>
    ),
  }

  return (
    <div className="mpage">
      <header className={`mhead ${mod.danger ? 'danger' : ''}`}>
        <div className="mhead-text">
          <div className="kicker">SENSING MODULE {mod.order} OF {MODULES.length}</div>
          <h1>{mod.title}</h1>
          {summary && <p className="lead">{summary}</p>}
          {mod.owner && <p className="sm">Owner: {mod.owner}</p>}
        </div>
        <span className="state-pill" style={{ color: data?.mock ? '#FFB347' : '#22E4FF', background: data?.mock ? 'rgba(255,179,71,.12)' : 'rgba(34,228,255,.10)' }}>
          {loading ? 'LOADING' : error ? 'NO DATA' : data?.mock ? 'PLACEHOLDER RESULTS' : 'RECORDED RESULTS'}
        </span>
      </header>

      {error && <Notice kind="warn">Couldn't load this module's data ({error.message}). Check src/modules/{id}/data/replay.json or VITE_BACKEND_URL in .env.local.</Notice>}
      {data?.mock && <Notice kind="warn">Results and replay frames are placeholders. Replace src/modules/{id}/data/replay.json or export from the backend.</Notice>}

      {frame && renderFrame && (
        <section className="card glow">
          <div className="sec-head">
            <h3>OUTPUT PREVIEW</h3>
            <span className="mono dim">{data.timeline_kind === 'presentation_only' ? `${frames.length} independent segments · sample ${idx + 1}/${frames.length}` : `${frames.length} frames · t = ${frame.t}s · ${idx + 1}/${frames.length}`} · source: {data._source}</span>
          </div>
          <div className="replay-ctl">
            <input type="range" min="0" max={frames.length - 1} value={idx} onChange={(e) => setIdx(+e.target.value)} aria-label="Frame" />
          </div>
          {renderFrame(frame, { frames, idx })}
        </section>
      )}

      <nav className="stagenav" aria-label="Stages">
        {STAGES.map((s, i) => (
          <button key={s.key} onClick={() => go(s.key)}><b>{i + 1}</b>{s.title}</button>
        ))}
      </nav>

      {STAGES.map((s, i) => (
        <section className="stage" id={`${id}-${s.key}`} key={s.key}>
          <div className="stage-head">
            <div className="stage-num">{i + 1}</div>
            <div>
              <div className="kicker dim" style={{ padding: 0 }}>{s.kicker}</div>
              <h2>{s.title}</h2>
            </div>
          </div>
          <div className="stage-body">
            <div className="card">
              <h3>PLAN</h3>
              <Plan items={plan[s.key]} />
            </div>
            <div className="card">
              <h3>RESULTS</h3>
              {blocks[s.key]}
              {stages[s.key]}
            </div>
          </div>
        </section>
      ))}
    </div>
  )
}
