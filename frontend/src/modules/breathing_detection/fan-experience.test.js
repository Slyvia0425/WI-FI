import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { csiReadingAt } from './csi-session.js'
import { reviewText } from './review-language.js'

const bundle = JSON.parse(readFileSync(new URL('./data/replay.json', import.meta.url), 'utf8'))
const fan = bundle.csi_bench.fan_experience

test('fan extension contains seven real scenario/class cells, four devices each', () => {
  assert.equal(fan.version, 1)
  assert.equal(fan.scenes.length, 7)
  assert.equal(new Set(fan.scenes.flatMap(s => s.samples.map(x => x.id))).size, 28)
  assert.ok(!fan.scenes.some(s => s.los && s.fan_on && s.label === 0))
  for (const scene of fan.scenes) {
    assert.equal(scene.samples.length, 4)
    assert.equal(new Set(scene.samples.map(s => s.metadata.device_id)).size, 4)
    for (const sample of scene.samples) {
      assert.equal(sample.label, scene.label)
      assert.equal(sample.metadata.fan_on, scene.fan_on)
      assert.equal(sample.metadata.los, scene.los)
      assert.equal(sample.split, 'test_id')
      assert.equal(sample.waveform.length, 300)
      assert.ok(sample.waveform.every(Number.isFinite))
      assert.equal(sample.candidate_bpm, undefined)
      assert.equal(csiReadingAt(sample, 9.99, fan.threshold).probability, null)
      assert.equal(csiReadingAt(sample, 10, fan.threshold).status, sample.status)
      assert.equal(csiReadingAt(sample, 0, fan.threshold).status, 'collecting')
    }
  }
})

test('assessment counts remain separate from selected examples and missing class stays unknown', () => {
  assert.equal(fan.assessment.sample_count, 4156)
  assert.equal(fan.threshold, bundle.csi_bench.threshold)
  assert.equal(fan.checkpoint_sha256, bundle.csi_bench.checkpoint_sha256)
  const metrics = fan.assessment.by_scenario
  assert.equal(metrics.LOS_FanOn.confusion_matrix.tp, 588)
  assert.equal(metrics.NLOS_FanOn.confusion_matrix.tp, 593)
  assert.equal(metrics.NLOS_FanOn.confusion_matrix.fp, 1)
  assert.equal(metrics.LOS_FanOn.empty_false_positive_rate, null)
  assert.match(fan.selection, /not prediction correctness/)
  assert.equal(bundle.csi_bench.scenes.length, 2)
})

test('fan selection and limits have Chinese review translations', () => {
  for (const scene of fan.scenes) assert.notEqual(reviewText(scene.title, 'zh'), scene.title)
  assert.equal(reviewText('CSI-Bench · Fan interference', 'zh'), 'CSI-Bench · 风扇干扰')
  assert.match(reviewText('Unavailable — no empty-room samples', 'zh'), /没有空房间/)
  assert.match(reviewText('Fan assessment details · not the current sample confidence', 'zh'), /不是当前样本/)
})
