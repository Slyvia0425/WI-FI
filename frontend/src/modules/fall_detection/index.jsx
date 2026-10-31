import { useState } from 'react'
import ModulePage from '../../components/ModulePage.jsx'
import { Notice } from '../../components/ui.jsx'
import { useModuleData } from '../../api/replay.js'
import { MetricTable } from '../../components/charts.jsx'
import bundled from './data/replay.json'
import archive from './data/archive_tcs.json'
import ArchivedTCSReplay from './ArchivedTCSReplay.jsx'

const source = { id: 'fall_detection', title: 'Fall detection', bundled }

function ResNetActivityReference() {
  const { data, loading, error } = useModuleData(source)
  const [selected, setSelected] = useState('')
  const [filter, setFilter] = useState('all')
  if (loading) return <p>Loading ResNet18 validation results…</p>
  if (error || !data?.frames?.length) return <Notice kind="warn">ResNet18 replay unavailable: {error?.message || 'no data'}.</Notice>
  if (!['independent_segment_validation', 'b_low_confidence_gate_validation'].includes(data.mode)) return <Notice kind="warn">Run the ResNet18 integration export before using this page.</Notice>
  const frames = data.frames.filter((f) => filter === 'all' || (filter === 'misses' && f.ground_truth === 'Fall' && f.prediction === 'Nonfall') || (filter === 'false_positives' && f.ground_truth === 'Nonfall' && f.prediction === 'Fall') || (filter === 'skipped' && f.C_ran === false) || (filter === 'skipped_falls' && f.C_ran === false && f.ground_truth === 'Fall'))
  const frame = frames.find((f) => f.sample_id === selected) || frames[0]
  const reference = frame?.activity_reference
  return (
    <div>
      <Notice>CSI-Bench official ResNet18 · HP validation only. These are independent complete segments, not a continuous recording. The displayed sample position is never used to align Module B.</Notice>
      <Notice kind="warn">Active rule: run C only when paired B confidence is below 0.6. At 0.6 or higher, C is not evaluated. On saved validation results this skips {data.gate_summary?.Fall_skipped ?? '—'} actual Fall segments. These are post-hoc routing results, not a new held-out evaluation.</Notice>
      <div className="btns" style={{ marginTop: 16 }}>
        <label className="sm">Show{' '}<select value={filter} onChange={(e) => { setFilter(e.target.value); setSelected('') }} aria-label="ResNet sample filter">
          <option value="all">All HP validation segments</option><option value="misses">Missed falls</option><option value="false_positives">Nonfall false positives</option><option value="skipped">C not evaluated</option><option value="skipped_falls">Actual falls skipped by B</option>
        </select></label>
        <label className="sm">Segment{' '}<select value={frame?.sample_id || ''} onChange={(e) => setSelected(e.target.value)} aria-label="CSI-Bench segment" style={{ maxWidth: '100%' }}>
          {frames.map((f) => <option key={f.sample_id} value={f.sample_id}>{f.sample_id}</option>)}
        </select></label>
      </div>
      {!frame ? <p>No matching segments.</p> : <>
        <div className="stage-body" style={{ marginTop: 20 }}>
          <div className="card">
            <h3>MODULE C · RESNET18</h3>
            <div className="big">{frame.C_ran === false ? 'C not evaluated' : frame.state === 'suspected' ? 'Suspected fall' : 'Below threshold'}</div>
            {frame.C_ran === false ? <p>{frame.gate?.reason}. No C score or binary decision is produced.</p> : <p>Fall score <strong>{frame.fall_score.toFixed(4)}</strong> · threshold <strong>{frame.threshold.toFixed(6)}</strong></p>}
            <p className="sm">Validation label: {frame.ground_truth} · {frame.C_ran === false ? 'Skipped; no C classification' : frame.correct ? 'Correct segment prediction' : 'Segment misclassification'}</p>
            <p className="sm">B confidence controls whether C runs. When selected, C keeps its frozen threshold and model. This replay selects previously saved C scores; it does not execute the model on page changes.</p>
          </div>
          <div className="card">
            <h3>MODULE B · ACTIVITY REFERENCE</h3>
            {reference?.status === 'aligned' ? <>
              <div className="big">{reference.activity}</div>
              <p>Confidence {(reference.confidence * 100).toFixed(1)}% · {reference.stable ? 'passes B confidence cutoff' : 'weak B evidence'}</p>
              <p className="sm">Same input verified · {reference.alignment} · quality {reference.quality}</p>
              <p className="sm">B chooses among five trained activities; Fall is not one of them. High B confidence now skips C under the requested rule; it does not establish absence of a fall.</p>
              <p className="sm">B confidence is not a fall probability or evidence of temporal stability.</p>
            </> : <>
              <div className="big">Reference unavailable</div>
              <p className="sm">{reference?.reason || 'No paired B result'}</p>
              <p className="sm">Valid same-input B evidence is required by this routing policy. Missing B is unavailable, not a negative fall decision.</p>
            </>}
          </div>
        </div>
        <details style={{ marginTop: 16 }}><summary>Shared-input identity for connecting a real B prediction</summary>
          <p className="sm">B must report this identity for the same raw amplitude segment, before either model applies its own preprocessing.</p>
          <code style={{ overflowWrap: 'anywhere' }}>{frame.input_id}</code>
        </details>
      </>}
      <p className="sm">{frames.length} segments in this filter. Saved validation scores with the new gate applied. Skipped C cases remain unknown. No real notification is sent.</p>
      <details style={{ marginTop: 24 }}><summary>Archived TCS-Fall experiment · separate model and dataset</summary>
        <Notice>The main results above are CSI-Bench ResNet18. This archived block is the independently trained five-layer TCS D model; it is not ResNet18 fine-tuning.</Notice>
        <div className="tblwrap"><MetricTable rows={archive.metrics.table} /></div>
        <ArchivedTCSReplay />
      </details>
    </div>
  )
}

export default function FallDetection() {
  return <ModulePage id="fall_detection" summary="B first: only confidence below 0.6 triggers ResNet18 C. Model weights and C threshold unchanged; skipped inputs have no C decision."
    stages={{ evaluation: <ResNetActivityReference /> }} />
}
