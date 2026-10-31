import { useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import './space.css'

export default function SpaceProximitySensing() {
  const [params, setParams] = useSearchParams()
  const mode = params.get('mode') === 'research' ? 'research' : 'lite'
  const [profiles, setProfiles] = useState(null)
  const [status, setStatus] = useState('checking')
  const [attempt, setAttempt] = useState(0)
  const [effects, setEffects] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const frame = useRef(null)
  const observer = useRef(null)
  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/integration', { signal: controller.signal })
      .then(r => { if (!r.ok) throw Error(); return r.json() })
      .then(d => {
        const available = d.spatial?.profiles || { research: d.spatial }
        setProfiles(available)
        setStatus(available[mode]?.available ? 'ready' : 'offline')
      })
      .catch(e => { if (e.name !== 'AbortError') setStatus('offline') })
    return () => controller.abort()
  }, [attempt, mode])
  useEffect(() => () => observer.current?.disconnect(), [])
  useEffect(() => {
    const target = frame.current
    if (!target) return
    const visibility = new IntersectionObserver(([entry]) => {
      target.contentWindow?.postMessage({ type: 'wifi-space-effects', enabled: effects, visible: entry.isIntersecting }, window.location.origin)
    }, { rootMargin: '120px' })
    visibility.observe(target)
    return () => visibility.disconnect()
  }, [effects, status, attempt, mode])
  const chooseMode = next => {
    if (mode === next) return
    observer.current?.disconnect()
    setStatus('checking')
    setParams({ mode: next })
  }
  const resize = () => {
    observer.current?.disconnect()
    const body = frame.current?.contentDocument?.body
    if (!body) return
    const fit = () => { if (frame.current) frame.current.style.height = `${Math.max(850, body.scrollHeight + 24)}px` }
    observer.current = new ResizeObserver(fit)
    observer.current.observe(body)
    fit()
    frame.current.contentWindow?.postMessage({ type: 'wifi-space-effects', enabled: effects, visible: true }, window.location.origin)
  }
  return <main className={`space-module ${effects ? 'space-fx-on' : ''}`}>
    <header className="mhead space-hero">
      <div className="mhead-text"><div className="kicker">MODULE E</div><h1>Spatial sensing</h1><p className="lead">Place objects, adjust transmitters and view predicted zone counts.</p></div>
      <div className="space-hero-tools">
        <svg className="space-radar" viewBox="0 0 120 90" fill="none" aria-hidden="true"><ellipse cx="60" cy="45" rx="45" ry="25" stroke="currentColor" opacity=".15"/><ellipse className="radar-orbit" cx="60" cy="45" rx="45" ry="25" stroke="currentColor" strokeDasharray="12 24 4 40"/><ellipse cx="60" cy="45" rx="29" ry="16" stroke="currentColor" opacity=".35"/><path d="M35 45h50M60 30v30" stroke="currentColor" opacity=".18"/><circle className="radar-core" cx="60" cy="45" r="4" fill="currentColor"/><circle cx="96" cy="31" r="2" fill="#bb99fa"/></svg>
        <div className="space-toggles"><span className="mode-chip simulation">Signal simulation</span><button type="button" className="space-effects-toggle" aria-pressed={effects} onClick={() => setEffects(v => !v)}><span aria-hidden="true">✦</span> Effects {effects ? 'on' : 'off'}</button></div>
      </div>
    </header>
    <section className="space-mode-grid" aria-label="Spatial run mode">
      <button className={`space-mode-card ${mode === 'lite' ? 'selected' : ''}`} aria-pressed={mode === 'lite'} onClick={() => chooseMode('lite')}>
        <span className="space-mode-title"><span>Light demo</span><span className={`space-mode-state ${profiles?.lite?.available ? 'ready' : ''}`}>{profiles ? profiles.lite?.available ? 'Ready' : 'Not installed' : 'Checking'}</span></span>
        <span className="space-mode-description">Same models. No full dataset download.</span>
        <span className="space-mode-features">Custom simulation · Signal import · 6 examples</span>
      </button>
      <button className={`space-mode-card ${mode === 'research' ? 'selected' : ''}`} aria-pressed={mode === 'research'} onClick={() => chooseMode('research')}>
        <span className="space-mode-title"><span>Full research</span><span className={`space-mode-state ${profiles?.research?.available ? 'ready' : ''}`}>{profiles ? profiles.research?.available ? 'Ready' : 'Not installed' : 'Checking'}</span></span>
        <span className="space-mode-description">Complete data and research assets.</span>
        <span className="space-mode-features">Custom simulation · Full test replay · Research tools</span>
      </button>
    </section>
    {status === 'checking' ? <div className="card">Connecting to the spatial model…</div> : status === 'ready' ?
      <iframe key={mode + attempt} ref={frame} src={mode === 'lite' ? '/spatial-lite/' : '/spatial/'} title={`Interactive spatial sensing · ${mode === 'lite' ? 'Light demo' : 'Full research'}`} className="space-frame" onLoad={resize} /> :
      <section className="card space-offline"><h2>{mode === 'lite' ? 'Light demo is unavailable' : 'Full research is unavailable'}</h2><p>{mode === 'lite' ? 'Install the light runtime package, then reconnect.' : 'Restore the full research package to enable all test scenes.'}</p><div className="btns"><button className="btn primary" onClick={() => { setStatus('checking'); setAttempt(a => a + 1) }}>Reconnect</button>{profiles?.[mode === 'lite' ? 'research' : 'lite']?.available && <button className="btn ghost" onClick={() => chooseMode(mode === 'lite' ? 'research' : 'lite')}>Switch to {mode === 'lite' ? 'full research' : 'light demo'}</button>}<Link className="btn ghost" to="/">Back to home</Link></div></section>}
  </main>
}
