import { useEffect, useState } from 'react'
import { Notice } from '../../components/ui.jsx'
import bundled from './data/archive_tcs.json'


function ScoreTrace({ frames, index, threshold, annotation }) {
  const width = 900, height = 230, pad = 35
  const end = frames.at(-1)?.t || 1
  const sx = (t) => pad + t / end * (width - 2 * pad)
  const sy = (s) => height - pad - s * (height - 2 * pad)
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label="Recorded Fall score, threshold, and nominal fall annotation">
      {[0, 0.5, 1].map((v) => <g key={v}>
        <line x1={pad} x2={width - pad} y1={sy(v)} y2={sy(v)} stroke="var(--border-strong)" />
        <text x={4} y={sy(v) + 4} fill="currentColor" fontSize="12">{v.toFixed(1)}</text>
      </g>)}
      <line x1={pad} x2={width - pad} y1={sy(threshold)} y2={sy(threshold)} stroke="#FFB347" strokeDasharray="6 5" />
      {annotation != null && <line x1={sx(annotation)} x2={sx(annotation)} y1={pad} y2={height - pad} stroke="#FF4D6D" strokeDasharray="3 4" />}
      <polyline fill="none" stroke="var(--accent)" strokeWidth="2.5" points={frames.slice(0, index + 1).map((f) => `${sx(f.t)},${sy(f.fall_score)}`).join(' ')} />
      <text x={pad} y={height - 8} fill="currentColor" fontSize="12">0 s</text>
      <text x={width - pad} y={height - 8} textAnchor="end" fill="currentColor" fontSize="12">{end.toFixed(1)} s</text>
    </svg>
  )
}

export default function ArchivedTCSReplay() {
  const data = bundled, error = null, loading = false
  const [selected, setSelected] = useState('')
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (data) setSelected(data.default_recording_id || data.recordings?.[0]?.id || '')
  }, [data])
  const recordings = data?.recordings || []
  const recording = recordings.find((r) => r.id === selected) || recordings[0]
  const frames = recording?.frames || data?.frames || []
  const frame = frames[Math.min(index, frames.length - 1)]
  useEffect(() => { setIndex(0); setPlaying(false) }, [selected])
  useEffect(() => {
    if (!playing || !frames.length) return undefined
    const timer = setInterval(() => setIndex((i) => (i + 1) % frames.length), 500)
    return () => clearInterval(timer)
  }, [playing, frames.length, selected])
  if (loading) return <p>Loading recorded validation results…</p>
  if (error || !frame) return <Notice kind="warn">Recorded replay unavailable: {error?.message || 'no frames'}.</Notice>
  const threshold = frame.threshold ?? data.threshold ?? 0.5
  const jumpNearFall = () => {
    setPlaying(false)
    const hit = frames.findIndex((f) => f.t >= (recording?.annotation_seconds || 0) - 8)
    setIndex(Math.max(0, hit))
  }
  return (
    <div>
      <Notice>Real TCS-Fall validation recording · subject 03 · frozen D model · recorded replay, not live sensor input. A positive score is suspected fall evidence.</Notice>
      {recordings.length > 0 && <label className="sm">Recording{' '}
        <select value={selected} onChange={(e) => setSelected(e.target.value)} aria-label="Fall recording">
          {recordings.map((r) => <option key={r.id} value={r.id}>{r.id} · background warnings: {r.background_alarm_episodes}</option>)}
        </select>
      </label>}
      <div className="big" style={{ marginTop: 16 }}>
        {frame.state === 'suspected' ? 'Suspected fall' : 'Below threshold'}{' '}
        <span className="sm">score {frame.fall_score.toFixed(3)} · threshold {threshold.toFixed(2)}</span>
      </div>
      <p className="sm">CSI time {frame.t.toFixed(3)} s · input {frame.window_start?.toFixed(3)}–{frame.window_end?.toFixed(3)} s · {index + 1}/{frames.length} windows</p>
      {frame.alarm_onset && <Notice kind="warn">New positive episode starts here. This is model evidence, not a confirmed accident.</Notice>}
      <ScoreTrace frames={frames} index={index} threshold={threshold} annotation={recording?.annotation_seconds} />
      <p className="sm">Orange dashed: threshold {threshold.toFixed(2)}. Red dotted: nominal annotation at {recording?.annotation_seconds ?? 'unknown'} s. It is not used to compute predictions.</p>
      <input type="range" min="0" max={Math.max(0, frames.length - 1)} value={index} style={{ width: '100%' }} aria-label="Fall replay window"
        onChange={(e) => { setPlaying(false); setIndex(Number(e.target.value)) }} />
      <div className="btns">
        <button className="btn primary" onClick={() => setPlaying((v) => !v)}>{playing ? 'Pause' : 'Play recorded replay (2×)'}</button>
        <button className="btn ghost" onClick={jumpNearFall}>Near annotated event</button>
        <button className="btn ghost" onClick={() => { setPlaying(false); setIndex(frames.length - 1) }}>Full trace</button>
      </div>
      <p className="sm">The default example includes a background false warning near 33 s. All 12 recordings are available; this replay does not establish joint accuracy with Module B.</p>
    </div>
  )
}

