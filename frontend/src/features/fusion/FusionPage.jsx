import { useMemo } from 'react'
import { MODULES } from '../../modules/registry.js'
import { useAllModules, framesByRole } from '../../api/replay.js'
import { fuse, TONES } from './rules.js'
import { RULES, CHECKS, IDEAS } from './content.js'
import { Notice, DISCLAIMER } from '../../components/ui.jsx'

function Cell({ tone, children }) {
  return <td style={tone ? { color: TONES[tone] } : undefined}>{children}</td>
}

export default function FusionPage() {
  const { all: recorded, ready } = useAllModules(MODULES)
  const all = useMemo(() => ({ ...recorded, space_proximity_sensing: null }), [recorded])
  const times = useMemo(() => {
    const s = new Set()
    Object.values(all).forEach((d) => d && d.frames.forEach((f) => s.add(f.t)))
    return [...s].sort((a, b) => a - b)
  }, [all])

  const rows = times.map((t) => {
    const f = framesByRole(MODULES, all, t)
    return { t, f, fused: fuse(f) }
  })
  const missing = MODULES.filter((m) => m.role !== 'space' && ready && !all[m.id]).map((m) => m.title)
  const areas = [...new Set(IDEAS.map((i) => i.area))]

  return (
    <div className="mpage">
      <header className="mhead">
        <div className="mhead-text">
          <div className="kicker">SYSTEM · FUSION</div>
          <h1>Fusion & improvement</h1>
          <p className="lead">How the five module outputs are combined into a room state, what the replay shows, and what to improve next.</p>
        </div>
      </header>
      <Notice>{DISCLAIMER}</Notice>
      {missing.length > 0 && <Notice kind="warn">No data from {missing.join(', ')}. They show as "no data" below and the rest keep working.</Notice>}
      <section className="card" style={{ marginBottom: 24 }}>
        <h3>INPUT PROVENANCE</h3>
        <p className="sm">B and C share HP input segments. A and D use independent recordings. E runs a separate static-object simulation and is excluded from this replay timeline. This is an interface demonstration, without synchronised joint labels.</p>
        <div className="tblwrap"><table className="mt">
          <thead><tr><th>Module</th><th>Data shown</th><th>Frames</th><th>Joint recording</th></tr></thead>
          <tbody>{MODULES.map((m) => <tr key={m.id}>
            <td>{m.title}</td><td>{m.role === 'space' ? 'Independent signal simulation' : !all[m.id] ? 'Unavailable' : all[m.id].mock ? 'Synthetic / placeholder fixture' : `${all[m.id].dataset || 'Recorded'} ${all[m.id].split || 'results'}`}</td>
            <td>{m.role === 'space' ? 'Interactive' : all[m.id]?.frames?.length ?? 0}</td><td>{m.role === 'space' ? 'Excluded from replay fusion' : 'Not established'}</td>
          </tr>)}</tbody>
        </table></div>
      </section>

      <section className="stage">
        <div className="stage-head"><div className="stage-num">1</div><div><div className="kicker dim" style={{ padding: 0 }}>RESULT</div><h2>Fusion replay</h2></div></div>
        <div className="card">
          <div className="tblwrap">
            <table className="mt">
              <thead><tr><th>t (s)</th>{MODULES.map((m) => <th key={m.id}>{m.title}</th>)}<th>Room state</th></tr></thead>
              <tbody>
                {rows.map(({ t, f, fused }) => (
                  <tr key={t}>
                    <td className="mono">{t}</td>
                    {MODULES.map((m) => <td key={m.id}>{f[m.role] ? m.summarize(f[m.role]) : 'no data'}</td>)}
                    <Cell tone={fused.tone}>{fused.room}</Cell>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="stage">
        <div className="stage-head"><div className="stage-num">2</div><div><div className="kicker dim" style={{ padding: 0 }}>METHOD</div><h2>Rule-based fusion</h2></div></div>
        <div className="stage-body">
          <div className="card">
            <h3>RULES</h3>
            <div className="tblwrap"><table className="mt"><thead><tr><th>Condition</th><th>System action</th></tr></thead>
              <tbody>{RULES.map(([a, b]) => <tr key={a}><td>{a}</td><td>{b}</td></tr>)}</tbody></table></div>
            <p className="sm" style={{ marginTop: 12 }}>A transparent rule layer, not a trained fusion network. Low-quality input never triggers a strong conclusion.</p>
          </div>
          <div className="card">
            <h3>SYSTEM-LEVEL CHECKS</h3>
            <div className="tblwrap"><table className="mt"><thead><tr><th>Check</th><th>What is measured</th></tr></thead>
              <tbody>{CHECKS.map(([a, b]) => <tr key={a}><td>{a}</td><td>{b}</td></tr>)}</tbody></table></div>
          </div>
        </div>
      </section>

      <section className="stage">
        <div className="stage-head"><div className="stage-num">3</div><div><div className="kicker dim" style={{ padding: 0 }}>NEXT</div><h2>Improvement suggestions</h2></div></div>
        <div className="ideas">
          {areas.map((a) => (
            <div className="card" key={a}>
              <h3>{a.toUpperCase()}</h3>
              <ul className="idea-list">
                {IDEAS.filter((i) => i.area === a).map((i) => (
                  <li key={i.t}><div>{i.t}</div><div className="sm">{i.why}</div></li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <p className="sm">Draft list. The team should edit it once results are in.</p>
      </section>
    </div>
  )
}
