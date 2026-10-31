import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { readingAt, eventsAt, DURATION, historicalReading } from './session.js'

const data = JSON.parse(readFileSync(new URL('./data/replay.json', import.meta.url), 'utf8'))
const samples = data.records.flatMap(r => r.samples)
test('historical replay preserves the last window without reviving live results', () => {
  for (const sample of samples) for (const mode of ['recorded', 'interrupted']) {
    assert.equal(historicalReading(sample, 0, mode), null)
    const history = historicalReading(sample, DURATION, mode)
    const last = sample.streaming[mode].events.filter(e => e.status).at(-1)
    assert.deepEqual(history, last || null)
    assert.equal(readingAt(sample, DURATION, data.threshold, mode).rate, null)
    if (mode === 'interrupted') assert.equal(history?.rate_bpm ?? null, null)
  }
})
test('all backend timelines hide future evidence, rate and events', () => {
  for (const s of samples) for (const mode of ['recorded','interrupted']) {
    assert.equal(readingAt(s, 9.99, data.threshold, mode).status, 'collecting')
    assert.equal(readingAt(s, 29.99, data.threshold, mode).rate, null)
    for (const t of [0,9,12,14,15,25,30,34]) assert.ok(eventsAt(s,t,data.threshold,mode).every(e => e.time <= t))
    assert.equal(readingAt(s, 0, data.threshold, mode).completed, 0)
  }
})
test('normal replay has five overlapping windows, not three fixed results', () => {
  const s = samples[0]
  const events = s.streaming.recorded.events.filter(e => e.status)
  assert.deepEqual(events.map(e => e.window_end), [10,15,20,25,30])
  for (const e of events) {
    assert.ok(e.time >= e.window_end)
    assert.equal(readingAt(s,e.time,data.threshold).windowEnd,e.window_end)
  }
  assert.ok(events.at(-1).rate_bpm > 0)
})
test('simulated loss expires results, recovery uses only fresh data', () => {
  for (const s of samples) {
    const at = t => readingAt(s,t,data.threshold,'interrupted')
    assert.equal(at(14).streamState,'disconnected')
    assert.equal(at(14).rate,null)
    assert.equal(at(14).stale,true)
    assert.equal(at(16).streamState,'warming_up')
    assert.equal(at(20).status,'collecting')
    assert.equal(at(20).rate,null)
    assert.equal(at(30).rate,null)
    assert.ok(s.streaming.interrupted.events.filter(e => e.status && e.generation > 0).every(e => e.window_start >= 12))
    assert.ok(eventsAt(s,20,data.threshold,'interrupted').some(e => e.title.includes('Input resumed')))
  }
})
test('end-of-input watchdog hides even previously valid rates', () => {
  for (const s of samples) for (const mode of ['recorded','interrupted']) {
    const r = readingAt(s,DURATION,data.threshold,mode)
    assert.equal(r.streamState,'disconnected')
    assert.equal(r.rate,null)
    assert.equal(r.windowEnd,null)
    assert.equal(readingAt(s,100,data.threshold,mode).elapsed,DURATION)
  }
})
test('missing streaming schema cannot masquerade as completed evidence', () => {
  const r = readingAt({},30,.05)
  assert.equal(r.available,false)
  assert.equal(r.rate,null)
  assert.equal(r.completed,0)
})
