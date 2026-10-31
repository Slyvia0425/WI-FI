import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const data = JSON.parse(readFileSync(new URL('./data/replay.json', import.meta.url), 'utf8'))
const id = 'mean_spectrum_boundary0_v1'
test('offline samples and streaming replay declare the same fixed estimator', () => {
  assert.equal(data.rate_estimator.id, id)
  assert.equal(data.rate_estimator.minimum_prominence, 0)
  assert.equal(data.rate_estimator.reliability_gate_enabled, false)
  assert.equal(data.streaming.rate_method, id)
  for (const r of data.records) for (const s of r.samples) {
    assert.equal(s.rate_estimator_id, id)
    if (s.rate_assessable) {
      assert.equal(s.candidate_bpm, s.rate_details.bpm)
      assert.equal(s.rate_details.estimator_id, id)
    } else assert.equal(s.candidate_bpm, null)
    for (const mode of ['recorded','interrupted']) for (const e of s.streaming[mode].events.filter(e => e.status)) {
      assert.equal(e.rate_estimator_id, id)
      if (e.rate_bpm != null) {
        assert.equal(e.rate_bpm, e.rate_details.bpm)
        assert.equal(e.rate_details.reliability_gate_enabled, false)
      }
    }
  }
})
test('every available recorded-stream rate matches the research sample rate exactly', () => {
  let pairs = 0
  for (const r of data.records) for (const s of r.samples) {
    for (const e of s.streaming.recorded.events.filter(e => e.rate_bpm != null)) {
      assert.equal(e.rate_bpm, s.candidate_bpm, s.id)
      pairs++
    }
  }
  assert.ok(pairs > 0)
})
test('new metrics are labelled retrospective; historical experiment and CSI examples survive', () => {
  assert.equal(data.benchmarks.rate_evaluation, 'retrospective_development_not_independent_test')
  assert.ok(Math.abs(data.benchmarks.rate.mae_bpm - .6826023159898477) < 1e-12)
  assert.equal(data.benchmarks.rate.large_errors, 19)
  assert.equal(data.benchmarks.rate.good_spoiled, 0)
  assert.ok(data.experiments.some(e => e.name === 'Historical fold-selected rate correction'))
  assert.equal(data.csi_bench.scenes.length, 2)
})
