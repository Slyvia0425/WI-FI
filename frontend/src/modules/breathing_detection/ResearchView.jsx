import { useEffect, useState } from 'react'
import { MODULES } from '../registry.js'
import { useModuleData } from '../../api/replay.js'
import { localizeTree, reviewText } from './review-language.js'
import './breathing.css'

const fmt = (v, d = 1) => Number.isFinite(v) ? v.toFixed(d) : '—'
const pct = v => `${fmt(v * 100, 2)}%`
const statusLabels = { breathing: 'Breathing evidence', none: 'No detected evidence', not_assessable: 'Not assessable' }
const tabs = [['preprocessing', 'Data & preprocessing'], ['training', 'Models'], ['experiments', 'Experiments'], ['evaluation', 'Evaluation']]

function Wave({ sample, reference, language }) {
  const values = [...sample.csi_waveform, ...(reference ? sample.reference_waveform : [])]
  const limit = Math.max(2.5, ...values.map(Math.abs))
  const y = v => 108 - v / limit * 73
  const path = wave => wave.map((v, i) => `${i ? 'L' : 'M'}${38 + i / (wave.length - 1) * 746},${y(v)}`).join(' ')
  return <svg className="br-wave" viewBox="0 0 800 230" role="img" aria-label={reviewText(`Thirty-second standardized CSI amplitude summary${reference ? ' and respiratory sensor reference' : ''}; not reconstructed respiration`, language)}>
    <defs><linearGradient id="br-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#60e9ce" stopOpacity=".18" /><stop offset="1" stopColor="#60e9ce" stopOpacity="0" /></linearGradient></defs>
    {[-2, -1, 0, 1, 2].map(v => <g key={v}><line x1="38" x2="784" y1={y(v)} y2={y(v)} stroke="currentColor" opacity=".1" /><text x="22" y={y(v) + 4} textAnchor="end">{v}</text></g>)}
    {[0, 5, 10, 15, 20, 25, 30].map(t => <g key={t}><line x1={38 + t / 30 * 746} x2={38 + t / 30 * 746} y1="23" y2="188" stroke="currentColor" opacity=".07" /><text x={38 + t / 30 * 746} y="213" textAnchor="middle">{t}s</text></g>)}
    <path d={`${path(sample.csi_waveform)} L784,188 L38,188 Z`} fill="url(#br-fill)" />
    {reference && <path d={path(sample.reference_waveform)} fill="none" stroke="#b7a5fc" strokeWidth="1.6" strokeDasharray="5 4" />}
    <path d={path(sample.csi_waveform)} fill="none" stroke="#60e9ce" strokeWidth="2" strokeLinejoin="round" />
  </svg>
}

function Stat({ label, value, unit, detail, purple }) {
  return <article className={`br-stat ${purple ? 'purple' : ''}`}><span>{label}</span><div>{value}<small>{unit}</small></div><p>{detail}</p></article>
}

export default function ResearchView({ language = 'en' }) {
  const mod = MODULES.find(m => m.id === 'breathing_detection')
  const { data, error, loading } = useModuleData(mod)
  const dashboard = data?.records?.length && data?.rate_estimator?.id === mod.bundled.rate_estimator?.id ? data : mod.bundled
  const [recordIndex, setRecordIndex] = useState(0)
  const [sampleIndex, setSampleIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [reference, setReference] = useState(false)
  const [tab, setTab] = useState('preprocessing')
  const records = dashboard?.records || []
  const record = records[Math.min(recordIndex, records.length - 1)]
  const samples = record?.samples || []
  const sample = samples[Math.min(sampleIndex, samples.length - 1)]
  useEffect(() => {
    if (!playing || !samples.length) return
    const timer = setInterval(() => setSampleIndex(i => {
      if (i >= samples.length - 1) { setPlaying(false); return i }
      return i + 1
    }), 2400)
    return () => clearInterval(timer)
  }, [playing, samples.length])
  if (!dashboard?.benchmarks || !sample) return localizeTree(<div className="br-page"><h1>Breathing detection</h1><p>{loading ? 'Loading recorded results…' : 'Recorded research results are not available.'}</p></div>, language)
  const b = dashboard.benchmarks
  const visibleRate = sample.rate_assessable && sample.status === 'breathing'
  const corrected = sample.candidate_bpm != null && Math.abs(sample.candidate_bpm - sample.baseline_bpm) > .01
  const reason = !sample.rate_assessable ? 'A sampling gap exceeds 100 ms in this 30-second window.' : sample.status !== 'breathing' ? 'BPM is withheld because the last 10-second window does not support a breathing output.' : 'Development estimate. Reliability screening is disabled; plausible rates may still be inaccurate.'
  const chooseRecord = index => { setPlaying(false); setRecordIndex(index); setSampleIndex(0) }
  const step = index => { setPlaying(false); setSampleIndex(index) }
  const switchTab = (event, key) => {
    if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const i = tabs.findIndex(t => t[0] === key)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? 3 : (i + (event.key === 'ArrowRight' ? 1 : 3)) % 4
    setTab(tabs[next][0]); document.getElementById(`br-tab-${tabs[next][0]}`)?.focus()
  }
  return localizeTree(<div className="br-page">
    <header className="br-heading"><div><div className="br-eyebrow"><i />SIGNAL INTELLIGENCE / MODULE 04</div><h1>A closer look at<br /><em>every breath.</em></h1><p>Explore breathing evidence, experimental respiratory rates, and the moments a Wi-Fi signal needs a second look.</p></div><div className="br-identity" aria-hidden="true"><div className="br-orbit"><svg viewBox="0 0 160 80"><path d="M0 40 H15 Q25 40 30 20 T45 40 T60 40 Q70 40 75 10 T90 40 T105 40 Q115 40 120 22 T135 40 H160" /></svg></div><span>CSI → EVIDENCE → INSIGHT</span></div></header>
    <div className="br-mode"><span className="br-chip mint"><i />OFFLINE RESEARCH</span><span>Real recorded data · no live device connection</span><span className="br-source">{data?.records ? data._source : 'bundled'} dataset</span></div>
    {error && <p className="br-inline-note">Backend unavailable. Showing the bundled offline dataset.</p>}
    {data && !data.records && <p className="br-inline-note">The backend uses an older format. Showing the bundled research dashboard.</p>}
    <section className="br-stats" aria-label="Evaluation overview">
      <Stat label="DETECTION · CSI-BENCH TEST" value={fmt(b.csi_bench_test.macro_f1 * 100, 2)} unit="% F1" detail="Frozen LSTM · binary Macro-F1" />
      <Stat label="EXTERNAL · WIRE-BREATH" value={fmt(b.external_recall * 100, 2)} unit="%" detail="Breathing recall · assessable windows" />
      <Stat label="RATE · EXPERIMENTAL" value={fmt(b.rate.mae_bpm, 2)} unit="BPM" detail="Retrospective MAE · unified fixed estimator" purple />
      <Stat label="RATE COVERAGE" value={fmt(b.rate_coverage * 100, 2)} unit="%" detail="394 / 467 segments · 100 ms gap rule" />
    </section>
    <section aria-labelledby="br-explorer-title"><div className="br-section-head"><div><div className="br-eyebrow">01 / RECORDING EXPLORER</div><h2 id="br-explorer-title">The signal behind the result</h2></div><span className="br-chip">30s recorded segments</span></div>
      <div className="br-scene-select"><label htmlFor="br-record">Recording scenario</label><select id="br-record" value={recordIndex} onChange={e => chooseRecord(Number(e.target.value))}>{records.map((r, i) => <option key={r.id} value={i}>{r.title} — {r.description}</option>)}</select></div>
      <div className="br-explorer-grid"><div className="br-signal-panel">
        <div className="br-signal-head"><div><span className="br-mini">CSI AMPLITUDE SUMMARY</span><h3>{record.title}</h3></div><span className={`br-chip ${sample.environment === 'Quiet' ? 'mint' : 'amber'}`}>{sample.environment}</span></div>
        <div className="br-wave-legend"><span><i />CSI summary</span>{reference && <span className="reference"><i />Respiratory sensor</span>}<span className="br-units">Standardized amplitude</span></div>
        <Wave sample={sample} reference={reference} language={language} />
        <label className="br-reference-toggle"><input type="checkbox" checked={reference} onChange={e => setReference(e.target.checked)} />Show sensor reference <span>evaluation only</span></label>
        <p className="br-chart-note">Mean CSI amplitude, not reconstructed respiration. Signals are independently standardized; phase agreement is not implied.{!sample.rate_assessable && ' This resampled segment contains a sampling gap.'}</p>
        <div className="br-evidence"><div className="br-section-caption"><span>10-second evidence windows</span><span>threshold {fmt(dashboard.threshold, 3)}</span></div><div className="br-window-grid">{sample.evidence.map(w => <div className="br-window" key={w.start}><div><span>{w.start}–{w.start + 10}s</span><b>{w.assessable ? pct(w.probability) : 'Gap'}</b></div><div className="br-track"><i style={{ width: `${w.assessable ? w.probability * 100 : 0}%` }} /><em style={{ left: `${dashboard.threshold * 100}%` }} /></div></div>)}</div></div>
        <div className="br-playback"><button className="br-play" aria-label={playing ? 'Pause segment browsing' : 'Play recorded segments'} onClick={() => { if (!playing && sampleIndex === samples.length - 1) setSampleIndex(0); setPlaying(!playing) }}>{playing ? 'Ⅱ' : '▶'}</button><button aria-label="Previous segment" disabled={sampleIndex === 0} onClick={() => step(sampleIndex - 1)}>←</button><input type="range" min="0" max={samples.length - 1} value={sampleIndex} onChange={e => step(Number(e.target.value))} aria-label="Recorded segment" /><button aria-label="Next segment" disabled={sampleIndex >= samples.length - 1} onClick={() => step(sampleIndex + 1)}>→</button><span>{sampleIndex + 1} / {samples.length}</span></div>
        <div className="br-playback-note">Independent segments · sample browsing, not a continuous monitoring timeline.</div>
      </div><aside className="br-reading-panel">
        <div className="br-mini">CURRENT READING <span>last 10 seconds</span></div><div className={`br-status ${sample.status}`}><i />{statusLabels[sample.status] || sample.status}</div>
        <div className="br-rate"><span>{visibleRate ? fmt(sample.candidate_bpm) : '—'}</span><small>breaths / min</small></div><span className="br-chip purple">EXPERIMENTAL RATE</span><p className="br-rate-reason">{reason}</p>
        <div className="br-reading-details"><div><span>Original spectral estimate</span><b>{sample.rate_assessable ? `${fmt(sample.baseline_bpm)} BPM` : 'Unavailable'}</b></div><div><span>Boundary correction</span><b>{!sample.rate_assessable ? 'Unavailable' : corrected ? 'Applied · fixed boundary0 rule' : 'Unchanged'}</b></div><div><span>Rate reliability screen</span><b className="br-amber-text">Disabled</b></div>{reference && <div><span>Sensor-derived reference</span><b>{fmt(sample.reference_bpm)} BPM</b></div>}</div>
        <div className={`br-quality ${sample.rate_assessable ? '' : 'warning'}`}><span>{sample.rate_assessable ? '✓' : '!'}</span><div><strong>{sample.rate_assessable ? 'Timestamp quality passed' : 'Sampling gap detected'}</strong><p>{sample.rate_assessable ? 'Quality alone does not establish rate reliability.' : 'BPM unavailable; breathing evidence is evaluated separately.'}</p></div></div>
        <p className="br-score-note">Evidence bars show model scores, not calibrated probabilities. Displayed rates use the unified fixed estimator; rate metrics are retrospective development results.</p>
      </aside></div>
      <div className="br-record-foot"><span>SUBJECT {sample.subject}</span><span>{sample.room}</span><span>{sample.posture}</span><span>SEGMENT {String(sample.segment).padStart(3, '0')}</span><code>{sample.id.split('@')[0]}</code></div><p className="br-curation">Five curated challenge/reference recordings. Aggregate scores use the full evaluation sets, not this selection.</p><p className="br-curation">Data: Ge, Wang & Ho, <a href="https://doi.org/10.1109/JBHI.2025.3619664" target="_blank" rel="noreferrer">WiRe-Breath (2025)</a> · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a>. Resampled and standardized for this preview.</p>
    </section>
    <section className="br-research" aria-labelledby="br-research-title"><div className="br-section-head"><div><div className="br-eyebrow">02 / RESEARCH NOTEBOOK</div><h2 id="br-research-title">Built on evidence. Clear about limits.</h2></div><span className="br-chip">Development snapshot</span></div>
      <div className="br-tabs" role="tablist" aria-label="Research stages">{tabs.map(([key, title], i) => <button key={key} id={`br-tab-${key}`} role="tab" aria-selected={tab === key} aria-controls="br-tab-panel" tabIndex={tab === key ? 0 : -1} onClick={() => setTab(key)} onKeyDown={e => switchTab(e, key)}><span>0{i + 1}</span>{title}</button>)}</div>
      <div id="br-tab-panel" className="br-tab-panel" role="tabpanel" aria-labelledby={`br-tab-${tab}`}>
        {tab === 'preprocessing' && <div className="br-two-columns"><div><span className="br-chip mint">COMPLETE</span><h3>Two sources. One ordered input.</h3><p>CSI-Bench provides breathing / empty-room supervision. WiRe-Breath adds aligned respiratory waveforms and interference scenarios.</p><div className="br-pipeline"><span>MAT + timestamps</span><b>→</b><span>30 Hz windows</span><b>→</b><span>Ordered CSI</span></div><p className="br-small">Frequency order is preserved. Rates use 234 original subcarriers; the LSTM receives 232 interpolated positions.</p></div><div className="br-facts"><div><b>467</b><span>WiRe-Breath segments</span></div><div><b>85</b><span>original recordings</span></div><div><b>5</b><span>participants · 2 rooms</span></div><div><b>5 folds</b><span>person holdout + recording validation</span></div></div></div>}
        {tab === 'training' && <div className="br-two-columns"><div><span className="br-chip mint">FROZEN REFERENCE</span><h3>Breathing evidence, first.</h3><p>A lightweight bidirectional LSTM consumes 10-second ordered sequences. Three full-data seeds were checked before seed 123 was frozen.</p><dl className="br-model-spec"><div><dt>Input</dt><dd>300 × 232</dd></div><div><dt>Projection / hidden</dt><dd>32 / 64</dd></div><div><dt>Decision threshold</dt><dd>{fmt(dashboard.threshold, 6)}</dd></div></dl></div><div className="br-model-card"><span className="br-mini">SEPARATE EXPERIMENTAL ESTIMATOR</span><h3>30-second respiratory rate</h3><p>Detrended, tapered subcarrier spectra → mean normalized spectrum → bounded peak. The fixed boundary0 rule reconsiders only exact boundary outputs.</p><span className="br-chip purple">No new neural rate model</span><p className="br-small">Reliability screening remains disabled. Reference rates are derived from a respiratory sensor waveform.</p></div></div>}
        {tab === 'experiments' && <div className="br-table-scroll"><table><thead><tr><th>Experiment</th><th>Observation</th><th>Decision</th></tr></thead><tbody>{dashboard.experiments.map(ex => <tr key={ex.name}><td><strong>{ex.name}</strong><small>{ex.note}</small></td><td>{ex.result}</td><td><span className={`br-chip ${ex.decision === 'Reference' ? 'mint' : ex.decision === 'Experimental' ? 'purple' : 'amber'}`}>{ex.decision}</span></td></tr>)}</tbody></table></div>}
        {tab === 'evaluation' && <div className="br-two-columns"><div><span className="br-chip">CSI-BENCH · FROZEN TEST</span><h3>Binary detection performance</h3><div className="br-matrix"><span>Actual ↓ / Predicted →</span><span>Empty</span><span>Breathing</span><span>Empty</span><b>{b.csi_bench_test.tn.toLocaleString()}</b><b className="error">{b.csi_bench_test.fp}</b><span>Breathing</span><b className="error">{b.csi_bench_test.fn}</b><b>{b.csi_bench_test.tp.toLocaleString()}</b></div><p className="br-small">14,999 assessable test windows. Empty-room FPR {pct(b.csi_bench_test.fpr)}. The observed test is not reused for model selection.</p></div><div><span className="br-chip purple">WIRE-BREATH · DEVELOPMENT</span><h3>Coverage belongs beside accuracy.</h3><p>Fixed-estimator person-averaged MAE: <strong>{fmt(b.rate_person_mae, 2)} BPM</strong>. Pooled MAE: <strong>{fmt(b.rate.mae_bpm, 2)} BPM</strong>. Participant sizes explain the difference.</p><div className="br-callout">Both Quiet and Interference contain breathing participants. WiRe-Breath cannot measure empty-room specificity.</div><p className="br-small">Current rate metrics use a fixed rule on previously observed development data. Historical grouped experiments are listed separately; new recordings are needed for independent evaluation.</p></div></div>}
      </div>
    </section>
    <section className="br-next"><span className="br-eyebrow">WHAT REMAINS</span><h2>Useful estimates need honest uncertainty.</h2><p>Validate rate reliability across new people and difficult interference. Add matched empty-room negatives. Keep breathing evidence separate from whether BPM can be trusted.</p><div className="br-next-pills"><span>Rate reliability · in development</span><span>New external recordings · needed</span><span>Live sensing · not connected</span></div></section>
  </div>, language)
}
