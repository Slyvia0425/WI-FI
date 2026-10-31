import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

export default function SpaceProximitySensing() {
  const [status, setStatus] = useState('checking')
  const [attempt, setAttempt] = useState(0)
  const frame = useRef(null)
  const observer = useRef(null)
  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/integration', { signal: controller.signal })
      .then(r => { if (!r.ok) throw Error(); return r.json() })
      .then(d => setStatus(d.spatial?.available ? 'ready' : 'offline'))
      .catch(e => { if (e.name !== 'AbortError') setStatus('offline') })
    return () => controller.abort()
  }, [attempt])
  useEffect(() => () => observer.current?.disconnect(), [])
  const resize = () => {
    observer.current?.disconnect()
    const body = frame.current?.contentDocument?.body
    if (!body) return
    const fit = () => { if (frame.current) frame.current.style.height = `${Math.max(850, body.scrollHeight + 24)}px` }
    observer.current = new ResizeObserver(fit)
    observer.current.observe(body)
    fit()
  }
  return <main className="space-module">
    <header className="mhead">
      <div className="mhead-text"><div className="kicker">MODULE E</div><h1>Spatial sensing</h1><p className="lead">Place objects, adjust transmitters and view predicted zone counts.</p></div>
      <span className="mode-chip simulation">Signal simulation</span>
    </header>
    {status === 'checking' ? <div className="card">Connecting to the spatial model…</div> : status === 'ready' ?
      <iframe key={attempt} ref={frame} src="/spatial/" title="Interactive spatial sensing" className="space-frame" onLoad={resize} /> :
      <section className="card space-offline"><h2>Spatial service is offline</h2><p>Start the integrated server with the spatial models installed.</p><code>python scripts/serve_integrated.py</code><div className="btns"><button className="btn primary" onClick={() => { setStatus('checking'); setAttempt(a => a + 1) }}>Reconnect</button><Link className="btn ghost" to="/">Back to home</Link></div></section>}
  </main>
}
