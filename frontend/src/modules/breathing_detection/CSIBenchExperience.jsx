import { useEffect, useState } from 'react'
import { localizeTree } from './review-language.js'
import { CSI_DURATION, csiReadingAt } from './csi-session.js'

export default function CSIBenchExperience({ data, language, fanMode = false }) {
  const [sceneIndex, setSceneIndex] = useState(() => fanMode ? Math.max(0, data?.scenes?.findIndex(s => s.fan_on && s.label === 1) ?? 0) : 0)
  const [sampleIndex, setSampleIndex] = useState(0)
  const [phase, setPhase] = useState('ready')
  const [elapsed, setElapsed] = useState(0)
  const [speed, setSpeed] = useState(1)
  useEffect(() => {
    if (phase !== 'running') return
    const start = performance.now(), previous = elapsed
    const timer = setInterval(() => {
      const next = Math.min(CSI_DURATION, previous + (performance.now() - start) / 1000 * speed)
      setElapsed(next)
      if (next >= CSI_DURATION) setPhase('finished')
    }, 100)
    return () => clearInterval(timer)
    // elapsed is the cursor captured at the playback action, not an effect dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, speed])
  useEffect(() => {
    const pause = () => { if (document.hidden) setPhase(p => p === 'running' ? 'paused' : p) }
    document.addEventListener('visibilitychange', pause)
    return () => document.removeEventListener('visibilitychange', pause)
  }, [])
  const reset = () => { setPhase('ready'); setElapsed(0) }
  if (!data?.scenes?.length) return localizeTree(<p>CSI-Bench examples are unavailable. Regenerate the dataset export.</p>, language)
  const scene = data.scenes[sceneIndex], sample = scene.samples[sampleIndex]
  const r = csiReadingAt(sample, elapsed, data.threshold)
  const path = sample.waveform.slice(0, phase === 'ready' ? 0 : r.visiblePoints).map((v,i) => `${i ? 'L' : 'M'}${36 + i / 299 * 728},${100 - Math.max(-4, Math.min(4,v)) * 18}`).join(' ')
  return localizeTree(<>
    <div className="bm-disclosure">CSI-Bench · official test examples · frozen model · independent 10-second samples. Offline replay, not a live occupancy sensor.</div>
    {fanMode && <div className="bm-fan-note"><strong>Fan interference · recorded-data experience</strong><p>Choose fan-on recordings or fan-off controls, then select a device recording. These are different records, not paired fan-on/off measurements.</p><p>LOS: line of sight. NLOS: non-line of sight. LOS fan-on empty-room recordings are unavailable; that false-positive rate is unknown.</p></div>}
    <section className="bm-selection" aria-label="Recording settings"><div><label htmlFor="csi-scene">Detection scenario</label><select id="csi-scene" value={sceneIndex} onChange={e => { reset(); setSceneIndex(+e.target.value); setSampleIndex(0) }}>{data.scenes.map((s,i) => <option key={s.id} value={i}>{s.title}</option>)}</select></div><div><label htmlFor="csi-sample">Independent sample</label><select id="csi-sample" value={sampleIndex} onChange={e => { reset(); setSampleIndex(+e.target.value) }}>{scene.samples.map((s,i) => <option value={i} key={s.id}>{fanMode ? s.metadata.device_id : 'Sample'} {fanMode ? '' : i+1} · 10s</option>)}</select></div><div className="bm-meta">CSI-Bench · test_id<br />{sample.metadata.environment_id} · {sample.metadata.original_shape[0]} subcarriers</div></section>
    <div className="bm-work-grid"><section className="bm-observation"><div className="bm-card-heading"><span>Detection experience</span><span className="bm-phase">{phase}</span></div><h2>{phase === 'ready' ? 'Ready to begin' : phase === 'paused' ? 'Playback paused — observation clock frozen' : r.complete ? 'Sample complete — historical result retained' : 'Collecting the first window'}</h2><p className="bm-session-help">300 stored samples at nominal 30 Hz. Original packet timestamps are unavailable; this view does not simulate verified packet continuity.</p>
      <div className="bm-clock"><strong>00:{String(Math.floor(elapsed)).padStart(2,'0')}</strong><span>/ 00:10</span><small>Sample time</small></div><progress max="10" value={elapsed} aria-label="Replay progress" />
      <div className="bm-controls"><button className="bm-primary" onClick={() => { if (phase === 'running') setPhase('paused'); else { if (phase === 'finished') setElapsed(0); setPhase('running') } }}>{({ready:'Start observation',running:'Pause',paused:'Resume',finished:'Replay sample'})[phase]}</button><button disabled={phase === 'ready'} onClick={reset}>Reset</button><label htmlFor="csi-speed">Playback speed<select id="csi-speed" value={speed} onChange={e => setSpeed(+e.target.value)}>{[1,2,4].map(v => <option value={v} key={v}>{v}×</option>)}</select></label></div>
      <div className="bm-signal-heading"><h3>Received signal</h3><span>Normalized mean CSI amplitude</span></div><svg className="bm-signal" viewBox="0 0 800 200" role="img" aria-label={language === 'zh' ? '已播放样本的标准化平均 CSI 幅度，并非呼吸波形' : 'Normalized mean CSI amplitude from the revealed sample; not a breathing waveform'}>{[-2,0,2].map(v => <g key={v}><line x1="36" x2="764" y1={100-v*18} y2={100-v*18} stroke="currentColor" opacity=".12" /><text x="20" y={104-v*18}>{v}</text></g>)}{[0,5,10].map(v => <text key={v} x={36+v/10*728} y="188" textAnchor="middle">{v}s</text>)}<path d={path} fill="none" stroke="#60e9ce" strokeWidth="2" /></svg>
      <p className="bm-limit">Waveforms are normalized separately. Empty-room noise can still fluctuate; waveform height alone cannot establish breathing or occupancy.</p><details className="bm-details"><summary>Sample provenance</summary><p>Selected by sample ID before inference, not by prediction correctness. These examples are not a new benchmark.</p><code>{sample.id}</code><p>CSI-Bench / test_id · 300 × {sample.metadata.original_shape[0]}</p></details>
    </section><aside className="bm-results" aria-label="Current monitoring results"><div className="bm-card-heading"><span>{r.complete ? 'Historical sample result' : 'Current results'}</span><span>One complete 10s window</span></div><div className={`bm-evidence ${r.status}`} aria-live="polite"><span>Model prediction</span><strong><i />{!r.complete ? 'Collecting fresh breathing evidence' : r.status === 'breathing' ? 'Breathing evidence detected' : 'No breathing evidence detected'}</strong><p>{r.complete ? 'Completed offline sample; not a current measurement.' : 'The model result is hidden until the full sample is read.'}</p></div>
      <div className="bm-quality"><span>Dataset ground truth</span><strong>{scene.title}</strong><p>{sample.label === 0 ? 'Original dataset label: empty. This label is not inferred by the model.' : 'Original dataset label: sleep, used as breathing-present supervision.'}</p></div>
      {r.complete && <div className="bm-quality"><span>Prediction vs. dataset label</span><strong>{(r.status === 'breathing') === (sample.label === 1) ? 'Matches the dataset label' : 'Does not match the dataset label'}</strong><p>Detection score: {r.probability.toFixed(4)} · threshold: {data.threshold.toFixed(4)}</p></div>}
      <div className="bm-bpm"><div><span>Respiratory rate</span><span className="bm-tag">Not provided</span></div><strong>—</strong><p>This independent 10s sample cannot supply the required 30s rate window. No respiratory reference is available in this view.</p></div><p className="bm-limit">No breathing evidence is not proof of an empty room. This model is not a general occupancy detector or a medical alarm.</p>
    </aside></div>
    {fanMode && data.assessment && <details className="bm-details bm-fan-assessment"><summary>Fan assessment details · not the current sample confidence</summary><p>4,156 previously observed test samples · U23 / E09 · frozen model and threshold. Not independent external validation and not BPM accuracy.</p><div className="bm-fan-table"><table><thead><tr><th>Condition</th><th>Breathing detected / labeled positive</th><th>False positives / empty room</th></tr></thead><tbody>{Object.entries(data.assessment.by_scenario).map(([name,m]) => <tr key={name}><td>{({LOS_FanOff:'Fan off · LOS',LOS_FanOn:'Fan on · LOS',NLOS_FanOff:'Fan off · NLOS',NLOS_FanOn:'Fan on · NLOS'})[name]}</td><td>{m.confusion_matrix.tp} / {m.available_classes.sleep} · {(m.breathing_recall*100).toFixed(2)}%</td><td>{m.empty_false_positive_rate == null ? 'Unavailable — no empty-room samples' : `${m.confusion_matrix.fp} / ${m.available_classes.empty} · ${(m.empty_false_positive_rate*100).toFixed(2)}%`}</td></tr>)}</tbody></table></div><p>Selected examples are not the evaluation set. Scores are not calibrated probabilities; changing the recording resets the observation.</p></details>}
  </>, language)
}
