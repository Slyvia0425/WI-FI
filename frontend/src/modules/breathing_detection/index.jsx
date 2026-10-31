import { useEffect, useState } from 'react'
import { MODULES } from '../registry.js'
import { useModuleData } from '../../api/replay.js'
import ResearchView from './ResearchView.jsx'
import CSIBenchExperience from './CSIBenchExperience.jsx'
import { localizeTree, reviewText } from './review-language.js'
import { DURATION, eventsAt, readingAt, historicalReading } from './session.js'
import './breathing.css'
import './monitor.css'
import './streaming.css'

const time = s => `00:${String(Math.floor(s)).padStart(2, '0')}`
const statusText = { collecting: 'Collecting fresh breathing evidence', breathing: 'Breathing evidence detected', none: 'Insufficient breathing evidence', not_assessable: 'Current window not assessable' }

function Signal({ sample, seconds, reference, gaps, language }) {
  const cursor = Math.min(30, seconds)
  const hidden = second => gaps.some(g => second >= g.start && second <= g.end)
  const path = wave => {
    let started = false
    return wave.map((v, i) => {
      const t = i / (wave.length - 1) * 30
      if (t > cursor || hidden(t)) { started = false; return '' }
      if (gaps.some(g => g.start > (i - 1) / (wave.length - 1) * 30 && g.start <= t)) started = false
      const cmd = started ? 'L' : 'M'; started = true
      return `${cmd}${36 + t / 30 * 728},${100 - Math.max(-4, Math.min(4, v)) * 18}`
    }).join(' ')
  }
  return <svg viewBox="0 0 800 200" className="bm-signal" role="img" aria-label={reviewText('Received standardized CSI amplitude; missing and future data hidden', language)}>
    {[-2, 0, 2].map(v => <g key={v}><line x1="36" x2="764" y1={100-v*18} y2={100-v*18} stroke="currentColor" opacity=".12" /><text x="20" y={104-v*18} textAnchor="end">{v}</text></g>)}
    {[0,10,20,30].map(v => <text key={v} x={36+v/30*728} y="188" textAnchor="middle">{v}s</text>)}
    {gaps.filter(g => g.start <= cursor).map((g,i) => <rect key={i} x={36+g.start/30*728} y="25" width={Math.max(2,(Math.min(g.end,cursor)-g.start)/30*728)} height="139" fill="#efc087" opacity=".2" />)}
    {reference && <path d={path(sample.reference_waveform)} fill="none" stroke="#b7a5fc" strokeDasharray="4 4" />}
    <path d={path(sample.csi_waveform)} fill="none" stroke="#60e9ce" strokeWidth="2" />
  </svg>
}

export default function BreathingDetection() {
  const mod = MODULES.find(m => m.id === 'breathing_detection')
  const { data } = useModuleData(mod)
  // A stale backend export must not silently replace the new streaming schema.
  const dashboard = data?.streaming?.version === 1 && data?.csi_bench?.version === 1
    && data?.csi_bench?.fan_experience?.version === 1
    && data?.rate_estimator?.id === mod.bundled.rate_estimator?.id ? data : mod.bundled
  const [view, setView] = useState('monitor')
  const [dataset, setDataset] = useState('csi')
  const language = 'en'
  const [recordIndex, setRecordIndex] = useState(0)
  const [sampleIndex, setSampleIndex] = useState(0)
  const [phase, setPhase] = useState('ready')
  const [elapsed, setElapsed] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [reference, setReference] = useState(false)
  const [mode, setMode] = useState('recorded')
  const records = dashboard?.records || []
  const record = records[recordIndex]
  const sample = record?.samples[sampleIndex]
  useEffect(() => {
    if (phase !== 'running' || view !== 'monitor' || dataset !== 'wire') return
    const start = performance.now(), previous = elapsed
    const timer = setInterval(() => {
      const next = Math.min(DURATION, previous + (performance.now()-start)/1000*speed)
      setElapsed(next)
      if (next >= DURATION) setPhase('finished')
    }, 100)
    return () => clearInterval(timer)
    // Restart the clock only on playback actions, not every tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, speed, view, dataset])
  useEffect(() => {
    const pause = () => { if (document.hidden) setPhase(p => p === 'running' ? 'paused' : p) }
    document.addEventListener('visibilitychange', pause)
    return () => document.removeEventListener('visibilitychange', pause)
  }, [])
  const reset = () => { setPhase('ready'); setElapsed(0); setReference(false) }
  const openView = v => { if (phase === 'running') setPhase('paused'); setView(v) }
  if (!sample) return <div className="bm-page">{reviewText('Loading recordings…', language)}</div>
  const r = readingAt(sample, elapsed, dashboard.threshold, mode)
  const events = phase === 'ready' ? [] : eventsAt(sample, elapsed, dashboard.threshold, mode)
  const ended = elapsed >= 30
  const disconnected = r.streamState === 'disconnected'
  const finished = phase === 'finished'
  const historical = finished ? historicalReading(sample, elapsed, mode) : null
  const displayedRate = finished ? historical?.rate_bpm : r.rate
  const displayedStatus = finished ? historical?.status || 'not_assessable' : r.status
  const message = finished ? 'Replay complete — historical results' : phase === 'ready' ? 'Ready to begin' : phase === 'paused' ? 'Playback paused — observation clock frozen'
    : disconnected ? ended ? 'Recording ended — current results expired' : 'Input disconnected — results hidden'
      : r.streamState === 'warming_up' ? r.generation ? 'Input restored — collecting a fresh window' : 'Collecting the first window'
        : ended ? 'Recording ended — checking input timeout' : 'Monitoring complete windows'
  const help = finished ? 'Review the last completed window below, or choose another recording.' : !r.available ? 'Streaming results are unavailable. Regenerate the dashboard export.'
    : disconnected ? 'No packets received for 2 seconds. Previous results are history, not current evidence.'
      : r.generation ? 'Recovery does not reuse old data. Wait for 10 fresh seconds for detection and 30 for rate.'
        : 'Detection uses 10 seconds; experimental rate uses 30 seconds. Both update every 5 seconds when ready.'
  const gaps = [...(sample.sampling_gaps || []), ...r.transportGaps]
  const rateHistory = sample.streaming?.[mode]?.events.filter(e => e.rate_bpm != null) || []
  const historicalRate = phase === 'finished' ? rateHistory.at(-1)?.rate_bpm : null
  return localizeTree(<div className="bm-page" lang={language === 'zh' ? 'zh-CN' : 'en'}>
    <header className="bm-heading"><div><div className="bm-eyebrow">WIFI SENSE / BREATHING</div><h1>Breathing workspace</h1><p>Observe the signal. Understand the result.</p></div><nav aria-label="Module views"><button className={view === 'monitor' ? 'active' : ''} onClick={() => openView('monitor')}>Monitoring</button><button className={view === 'research' ? 'active' : ''} onClick={() => openView('research')}>Models & validation ↗</button></nav></header>
    {view === 'research' ? <><div className="bm-return"><button onClick={() => openView('monitor')}>← Back to monitoring</button><span>Offline research, not live monitoring.</span></div><ResearchView language={language} /></> : <>
      <section className="bm-dataset-picker" aria-label="Experience dataset"><div><strong>Recorded-data experience</strong><p>Choose a scenario, then start the replay.</p></div><div role="group" aria-label="Experience dataset"><button type="button" aria-pressed={dataset === 'csi'} onClick={() => { reset(); setDataset('csi') }}>Breathing detection</button><button type="button" aria-pressed={dataset === 'wire'} onClick={() => { reset(); setDataset('wire') }}>Respiratory rate</button><button type="button" aria-pressed={dataset === 'fan'} onClick={() => { reset(); setDataset('fan') }}>Fan interference</button></div></section>
      <div className="bm-source-strip"><span>{dashboard === data && data?._source === 'backend' ? 'Recorded demo · Backend export' : 'Recorded demo · Bundled data'} · {dataset === 'wire' ? 'WiRe-Breath · 30s' : 'CSI-Bench · 10s'}</span><button disabled>Connect a device · Coming soon</button></div>
      {dataset !== 'wire' ? <CSIBenchExperience key={dataset} data={dataset === 'fan' ? dashboard.csi_bench.fan_experience : dashboard.csi_bench} language={language} fanMode={dataset === 'fan'} /> : <>
      <section className="bm-selection" aria-label="Recording settings"><div><label htmlFor="bm-record">Scenario</label><select id="bm-record" value={recordIndex} onChange={e => { reset(); setRecordIndex(+e.target.value); setSampleIndex(0) }}>{records.map((v,i) => <option key={v.id} value={i}>{v.title} · {v.description}</option>)}</select></div><div><label htmlFor="bm-segment">Independent segment</label><select id="bm-segment" value={sampleIndex} onChange={e => { reset(); setSampleIndex(+e.target.value) }}>{record.samples.map((s,i) => <option key={s.id} value={i}>Segment {s.segment} · 30s</option>)}</select></div><div className="bm-meta">Subject {sample.subject} · {sample.room}<br />{sample.posture} / {sample.environment}</div></section>
      <details className="bm-advanced"><summary>Advanced options</summary><section className="bm-selection"><div><label htmlFor="bm-transport">Replay condition</label><select id="bm-transport" value={mode} onChange={e => { reset(); setMode(e.target.value) }}><option value="recorded">As recorded</option><option value="interrupted">Simulate packet loss · 12–15s</option></select></div><p>{mode === 'interrupted' ? 'Deliberately discarded packets. This is an engineering simulation, not an original recording fault.' : 'Original sampling gaps are preserved. Independent segments are never joined.'}</p></section><p>Backend-computed streaming replay, not live inference. A 30s observation plus packet-delivery and timeout observation time; 34s total.</p></details>
      {mode === 'interrupted' && <p className="bm-mode-note">Simulate packet loss · 12–15s</p>}
      <div className="bm-work-grid"><section className="bm-observation"><div className="bm-card-heading"><span>This observation</span><span className={`bm-phase ${phase}`}>{phase}</span></div><h2>{message}</h2><p className="bm-session-help">{help}</p>
        <div className="bm-clock"><strong>{time(elapsed)}</strong><span>/ {time(DURATION)}</span><small>Replay time</small></div><progress value={elapsed} max={DURATION} aria-label="Replay progress" />
        <div className="bm-controls"><button className="bm-primary" disabled={!r.available} onClick={() => { if (phase === 'running') setPhase('paused'); else { if (phase === 'finished') setElapsed(0); setPhase('running') } }}>{({ready:'Start observation',running:'Pause',paused:'Resume',finished:'Replay segment'})[phase]}</button><button disabled={phase === 'ready'} onClick={reset}>Reset</button><label htmlFor="bm-speed">Playback speed<select id="bm-speed" value={speed} onChange={e => setSpeed(+e.target.value)}>{[1,2,4].map(v => <option key={v} value={v}>{v}×</option>)}</select></label></div>
        <div className="bm-milestones">{[10,15,20,25,30].map(t => <div key={t} className={elapsed >= t ? 'complete' : ''}><i />{t}s<span>{t === 30 ? 'Rate eligible*' : 'Detection update*'}</span></div>)}</div><p>*Requires a complete, quality-valid window; recovery restarts accumulation.</p>
        <div className="bm-signal-heading"><h3>CSI signal</h3><span>Not a reconstructed breathing waveform</span></div><Signal sample={sample} seconds={elapsed} reference={reference} gaps={gaps} language={language} />
        <details className="bm-details"><summary>Signal details & reference</summary><p>Standardized mean CSI amplitude, clipped to ±4 SD; not reconstructed breathing. Amber marks missing input. Reference is evaluation-only, never a prediction input.</p><label><input type="checkbox" checked={reference} onChange={e => setReference(e.target.checked)} />Show received respiratory reference</label>{elapsed >= 30 && reference && <p>Reference: {sample.reference_bpm?.toFixed(1) ?? 'Unavailable'} BPM</p>}<code>{sample.id}</code></details>
      </section><aside className="bm-results" aria-label="Current monitoring results"><div className="bm-card-heading"><span>{finished ? 'Historical session result' : 'Current results'}</span><span>{finished ? 'Not a current measurement' : '5-second update cadence'}</span></div>
        <div className={`bm-evidence ${disconnected && !finished ? 'not_assessable' : displayedStatus}`} aria-live="polite"><span>Breathing evidence</span><strong><i />{disconnected && !finished ? 'No current result — input disconnected' : statusText[displayedStatus]}</strong><p>{finished ? historical ? `Latest complete ${historical.window_start.toFixed(1)}–${historical.window_end.toFixed(1)}s window` : 'No completed window is available.' : r.windowEnd != null ? `Latest complete ${r.windowStart.toFixed(1)}–${r.windowEnd.toFixed(1)}s window` : r.stale ? 'Previous evidence expired. Fresh input is required.' : 'Waiting for 10 seconds of fresh input.'}</p></div>
        <div className="bm-bpm"><div><span>Respiratory rate</span><span className="bm-tag purple">Experimental</span></div><strong>{displayedRate != null ? displayedRate.toFixed(1) : '—'}</strong><span>breaths / min</span><p>{finished ? displayedRate != null ? 'Historical estimate · not a current measurement' : 'No rate was available in the last completed window.' : disconnected ? 'Expired — not a current measurement' : r.rate != null ? 'Experimental estimate; interference can affect accuracy.' : r.rateReason === 'timestamp_gap' ? 'Sampling gap affects the 30s window' : r.rateReason === 'insufficient_breathing_evidence' ? 'Withheld: insufficient breathing evidence' : 'Needs 30 seconds of fresh, valid input'}</p></div>
        <div className={`bm-quality ${r.gapSeen || (disconnected && !finished) ? 'warning' : ''}`}><span>Input status</span><strong>{finished ? 'Recording complete' : phase === 'ready' ? 'Waiting to start' : disconnected ? 'Disconnected / end of input' : r.streamState === 'warming_up' ? 'Receiving — warming up' : 'Receiving recorded input'}</strong><p>{r.gapSeen ? 'A recorded sampling gap has been seen. Each completed window is checked separately.' : 'Sampling quality is separate from rate reliability.'}</p></div>
        <div className="bm-guidance"><strong>What to do next</strong><p>{phase === 'ready' ? 'Choose a condition and start. No upload or Wi-Fi credentials needed.' : phase === 'paused' ? 'Resume playback; pausing is not simulated packet loss.' : disconnected ? ended ? 'Review the historical summary or replay another segment.' : 'Wait for simulated input to return; old results will not be reused.' : 'Continue observing. Changing the recording or replay condition resets the session.'}</p></div><p className="bm-limit">Temporal rejection and rate-reliability gates remain disabled. Insufficient evidence does not establish absence of breathing.</p>
      </aside></div>
      <section className="bm-bottom-grid"><div className="bm-events"><div className="bm-card-heading"><h2>Recent events</h2><span>Received events only</span></div>{events.length ? <ol>{events.map((e,i) => <li key={`${e.time}-${i}`}><time>{time(e.time)}</time><div><strong>{e.title}</strong><p>{e.detail}</p></div></li>)}</ol> : <div className="bm-empty">Start to see detection updates, timeout and recovery events.</div>}</div><div className="bm-summary"><div className="bm-card-heading"><h2>Historical session summary</h2><span>Not current evidence</span></div>{phase === 'finished' ? <><dl><div><dt>Independent recording</dt><dd>30s observation + delivery / timeout check</dd></div><div><dt>Detection updates</dt><dd>{r.completed}</dd></div><div><dt>Breathing-positive updates</dt><dd>{r.positive}</dd></div><div><dt>Last available historical rate</dt><dd>{historicalRate == null ? 'Unavailable' : `${historicalRate.toFixed(1)} BPM · experimental`}</dd></div><div><dt>Replay condition</dt><dd>{mode === 'interrupted' ? 'Simulated packet loss at 12–15s' : 'As recorded'}</dd></div></dl><p>Past results are not live measurements. One short recording cannot establish long-term stability.</p></> : <div className="bm-empty">Available after playback and the end-of-input timeout check.</div>}</div></section>
      <footer className="bm-footer">Independent recordings are never concatenated. WiRe-Breath · Ge, Wang & Ho (2025) · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Resampled and standardized.</footer>
      </>}
    </>}
  </div>, language)
}
