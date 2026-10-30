import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { csiReadingAt } from './csi-session.js'
import { reviewText } from './review-language.js'

const bundle = JSON.parse(readFileSync(new URL('./data/replay.json', import.meta.url), 'utf8'))
const data = bundle.csi_bench
test('two real CSI-Bench scenes preserve labels, IDs and independent 300-point samples', () => {
  assert.deepEqual(data.scenes.map(s => s.id), ['breathing','empty'])
  assert.match(data.selection, /selection before inference/)
  assert.equal(data.checkpoint_sha256, 'fff3cdb28c7577489400be7cc82506f074703ea22425404be1cc3224923df956')
  for (const scene of data.scenes) {
    assert.equal(scene.samples.length, 3)
    for (const s of scene.samples) {
      assert.equal(s.label, scene.id === 'empty' ? 0 : 1)
      assert.equal(s.split, 'test_id')
      assert.equal(s.duration_seconds, 10)
      assert.equal(s.waveform.length, 300)
      assert.ok(s.waveform.every(Number.isFinite))
      assert.ok(s.probability >= 0 && s.probability <= 1)
      assert.equal(s.status, s.probability >= data.threshold ? 'breathing' : 'none')
    }
  }
})
test('no result before 10 seconds and resetting the cursor hides prior evidence', () => {
  for (const scene of data.scenes) for (const s of scene.samples) {
    for (const t of [0,5,9.999]) {
      assert.equal(csiReadingAt(s,t,data.threshold).status, 'collecting')
      assert.equal(csiReadingAt(s,t,data.threshold).probability, null)
    }
    assert.equal(csiReadingAt(s,10,data.threshold).status, s.status)
    assert.equal(csiReadingAt(s,10,data.threshold).visiblePoints, 300)
    assert.equal(csiReadingAt(s,100,data.threshold).elapsed, 10)
  }
})
test('prediction disagreement is preserved rather than replaced with ground truth', () => {
  const empty = data.scenes.find(s => s.id === 'empty').samples
  assert.ok(empty.some(s => s.status === 'none'))
  const falsePositive = empty.find(s => s.status === 'breathing')
  assert.ok(falsePositive, 'Deterministically selected false positive must remain visible')
  assert.equal(csiReadingAt(falsePositive,10,data.threshold).status, 'breathing')
  assert.equal(falsePositive.label, 0)
})
test('Chinese copy covers dataset distinction and does not claim occupancy certainty', () => {
  assert.equal(reviewText('Empty room','zh'), '空房间')
  assert.equal(reviewText('Breathing present','zh'), '有人呼吸')
  assert.match(reviewText('No breathing evidence is not proof of an empty room. This model is not a general occupancy detector or a medical alarm.', 'zh'), /不代表确认空房间/)
})
