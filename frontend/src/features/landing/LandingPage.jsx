import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { MODULES } from '../../modules/registry.js'
import { framesByRole } from '../../api/replay.js'
import DEMO from './demo.json'
import { fuse, TONES } from '../fusion/rules.js'
import { Notice } from '../../components/ui.jsx'
import FloorPlan from './FloorPlan.jsx'

export default function LandingPage() {
  // The animation runs on its own invented data (demo.json), not on the output of the modules.
  const all = useMemo(() => Object.fromEntries(MODULES.map((m) => [m.id, DEMO[m.id] ? { frames: DEMO[m.id] } : null])), [])
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(true)
  const started = useRef(false)
  const timer = useRef(null)

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
  const loaded = MODULES.filter((m) => all[m.id]).length
  const checked = Object.keys(all).length > 0

  const chips = MODULES.map((m) => [m.title, frames[m.role] && m.summarize(frames[m.role])])

  return (
    <>
      <section className="hero">
        <div className="hero-text">
          <div className="kicker">CAMERA-FREE · WEARABLE-FREE · CSI-BASED</div>
          <h1>Sense the room.<br /><span>Skip the camera.</span></h1>
          <p>
            A WiFi CSI sensing prototype that recognises human activity and falls in a single-person room,
            then explains each decision using motion source, breathing and location evidence.
          </p>
          <div className="btns">
            <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
              {playing ? 'Pause replay' : 'Start replay'}
            </button>
            <a className="btn ghost" href="#modules">Browse modules</a>
          </div>
          <div className="pills">
            <span>Single-person room</span><span>Offline replay</span><span>CSI-Bench data</span>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="mono dim">LIVING ROOM · SIGNAL MAP</span>
            <span className="state-pill" style={{ color: col, background: `${col}1a` }}>{fused.room.toUpperCase()}</span>
          </div>
          <FloorPlan frames={frames} fused={fused} />
          <div className="legend">
            <span><i style={{ background: TONES.normal }} />Normal</span>
            <span><i style={{ background: TONES.suspected }} />Suspected fall</span>
            <span><i style={{ background: TONES.alert }} />Alert</span>
            <span><i style={{ background: TONES.uncertain }} />Uncertain</span>
          </div>
          <div className="ctl">
            <input type="range" min="0" max={Math.max(0, times.length - 1)} value={idx} aria-label="Replay time"
              onChange={(e) => { setPlaying(false); setTime(times[+e.target.value]) }} />
            <span className="mono dim">t = {time}s</span>
          </div>
        </div>
      </section>

      {checked && loaded < MODULES.length && (
        <Notice kind="warn">{MODULES.length - loaded} module(s) have no data yet. They show as "no data" and the rest keep working.</Notice>
      )}

      <section id="modules" className="mods">
        <div className="kicker dim">FIVE SENSING MODULES</div>
        <div className="cards">
          {MODULES.map((m, i) => (
            <div key={m.id} className={`mcard ${m.danger ? 'danger' : ''}`}>
              <div className="mtitle">{m.title}</div>
              <div className="mblurb">{m.blurb}</div>
              <div className={`mout ${chips[i][1] ? '' : 'missing'}`}>{chips[i][1] || 'no data'}</div>
              <Link to={`/module/${m.id}`}>Open module →</Link>
            </div>
          ))}
        </div>
      </section>
    </>
  )
}
