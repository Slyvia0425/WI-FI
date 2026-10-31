import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { MODULES } from '../../modules/registry.js'
import { framesByRole } from '../../api/replay.js'
import DEMO from './demo.json'
import { fuse, TONES } from '../fusion/rules.js'
import FloorPlan from './FloorPlan.jsx'

const DETAILS = {
  motion_source_recognition: { description: 'Identify whether movement comes from a person, pet, robot or fan.', letter: 'A', output: 'Human · Pet · Robot · Fan', action: 'Explore signal sources', icon: '◉' },
  human_activity_recognition: { description: 'Explore activity predictions and confidence for recorded signal samples.', letter: 'B', output: 'Activity class & confidence', action: 'Review activity predictions', icon: '↗' },
  fall_detection: { description: 'Review fall scores, decision thresholds and difficult cases.', letter: 'C', output: 'Fall score & decision', action: 'Inspect fall detection', icon: '⌁' },
  breathing_detection: { description: 'Replay breathing signals and inspect evidence, rate and signal quality.', letter: 'D', output: 'Breathing evidence & rate', action: 'Open breathing monitor', icon: '∿' },
  space_proximity_sensing: { letter: 'E', output: 'Occupied zones & object counts', action: 'Try spatial sensing', icon: '▧' },
}

export default function LandingPage() {
  // The animation runs on its own invented data (demo.json), not on the output of the modules.
  const all = useMemo(() => Object.fromEntries(MODULES.map((m) => [m.id, DEMO[m.id] ? { frames: DEMO[m.id] } : null])), [])
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(true)
  const started = useRef(false)
  const timer = useRef(null)
  const [services, setServices] = useState(null)
  useEffect(() => {
    let live = true
    const refresh = () => fetch('/api/integration').then(r => { if (!r.ok) throw Error(); return r.json() }).then(d => { if (live) setServices(d) }).catch(() => { if (live) setServices(null) })
    refresh()
    const id = setInterval(refresh, 15000)
    return () => { live = false; clearInterval(id) }
  }, [])

  const times = useMemo(() => {
    const s = new Set()
    Object.values(all).forEach((d) => d && d.frames.forEach((f) => s.add(f.t)))
    return [...s].sort((a, b) => a - b)
  }, [all])

  useEffect(() => {
    if (!times.length || started.current) return
    started.current = true
    const msr = MODULES.find((m) => m.role === 'motionSource')
    const humanT = msr && all[msr.id] && all[msr.id].frames.find((f) => f.label === 'Human')
    setTime(humanT ? humanT.t : times[0])
  }, [times])
  const idx = Math.max(0, times.indexOf(time))

  useEffect(() => {
    if (!playing) return
    timer.current = setInterval(() => {
      setTime((t) => {
        const i = times.indexOf(t)
        return times[(i + 1) % times.length]
      })
    }, 2200)
    return () => clearInterval(timer.current)
  }, [playing, times])

  const frames = framesByRole(MODULES, all, time)
  const fused = fuse(frames)
  const col = TONES[fused.tone]

  return (
    <>
      <section className="hero">
        <div className="hero-text">
          <div className="kicker">GROUP 66 · WI-FI SENSING</div>
          <h1>Sense the room.<br /><span>Skip the camera.</span></h1>
          <p>
            Five sensing modules, one workspace. Explore motion, activity, falls, breathing and indoor space through Wi-Fi signals.
          </p>
          <div className="btns">
            <button className="btn primary" onClick={() => document.getElementById('modules')?.scrollIntoView({ behavior: 'smooth' })}>Explore modules ↓</button>
            <Link className="btn ghost" to="/module/space_proximity_sensing">Try spatial sensing ↗</Link>
          </div>
          <div className="pills">
            <span>No cameras</span><span>No wearables</span><span>Replay & simulation</span>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="mono dim">ILLUSTRATIVE ROOM SCENARIO</span>
            <span className="state-pill" style={{ color: col, background: `${col}1a` }}>ILLUSTRATION</span>
          </div>
          <FloorPlan frames={frames} fused={fused} />
          <div className="legend">
            <span><i style={{ background: TONES.normal }} />Normal</span>
            <span><i style={{ background: TONES.suspected }} />Suspected fall</span>
            <span><i style={{ background: TONES.alert }} />Alert</span>
            <span><i style={{ background: TONES.uncertain }} />Uncertain</span>
          </div>
          <div className="ctl">
            <button className="room-toggle" aria-label={playing ? 'Pause illustration' : 'Play illustration'} onClick={() => setPlaying(p => !p)}>{playing ? 'Ⅱ' : '▶'}</button>
            <input type="range" min="0" max={Math.max(0, times.length - 1)} value={idx} aria-label="Replay time"
              onChange={(e) => { setPlaying(false); setTime(times[+e.target.value]) }} />
            <span className="mono dim">t = {time}s</span>
          </div>
        </div>
      </section>

      <section id="modules" className="mods">
        <div className="module-directory-head"><div><div className="kicker">THE WORKSPACE</div><h2>Choose a module</h2></div><span className="directory-status"><i className={services ? 'online' : ''} />{services ? 'Local services connected' : 'Offline preview'}</span></div>
        <div className="module-grid">
          {MODULES.map((m) => (
            <Link to={`/module/${m.id}`} key={m.id} className={`module-card ${m.danger ? 'danger' : ''}`}>
              <div className="module-card-top"><span className="module-letter">MODULE {DETAILS[m.id].letter}</span><span className={`mode-chip ${m.role === 'space' ? 'simulation' : ''}`}>{m.role === 'space' ? services?.spatial?.available ? 'Interactive simulation' : 'Simulation offline' : services?.modules?.[m.id]?.mode || 'Bundled results'}</span></div>
              <span className="module-icon" aria-hidden="true">{DETAILS[m.id].icon}</span>
              <h3>{m.title}</h3>
              <p>{DETAILS[m.id].description || m.blurb}</p>
              <div className="module-output">{DETAILS[m.id].output}</div>
              <div className="module-card-bottom"><span>{DETAILS[m.id].action}</span><b>↗</b></div>
            </Link>
          ))}
          <Link to="/fusion" className="module-card fusion-card"><div className="module-card-top"><span className="module-letter">SYSTEM VIEW</span><span className="mode-chip">Rule demonstration</span></div><span className="module-icon" aria-hidden="true">⌘</span><h3>Fusion & improvement</h3><p>Review how the interfaces and decision rules fit together.</p><div className="module-output">Independent inputs · No joint test score</div><div className="module-card-bottom"><span>Explore the system</span><b>↗</b></div></Link>
        </div>
      </section>
    </>
  )
}
