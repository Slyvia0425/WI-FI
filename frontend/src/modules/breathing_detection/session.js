export const DURATION = 34
export const SIGNAL_DURATION = 30

export function readingAt(sample, seconds, threshold, mode = 'recorded') {
  const elapsed = Math.max(0, Math.min(DURATION, seconds))
  const timeline = sample.streaming?.[mode]
  const snapshot = timeline?.snapshots.filter(s => s.time <= elapsed).at(-1)
  const updates = (timeline?.events || []).filter(e => e.status && e.time <= elapsed)
  const current = snapshot && !snapshot.last_result_stale && snapshot.window_end != null
  return { elapsed, completed: updates.length, positive: updates.filter(e => e.status === 'breathing').length,
    status: current ? snapshot.status : 'collecting', streamState: snapshot?.stream_state || 'waiting',
    stale: snapshot?.last_result_stale || false, generation: snapshot?.generation || 0,
    windowStart: snapshot?.window_start, windowEnd: snapshot?.window_end,
    gapSeen: (sample.sampling_gaps || []).some(g => g.start <= elapsed),
    rateReady: current && elapsed >= 30, rate: current ? snapshot.rate_bpm : null,
    rateReason: !current ? 'accumulating' : updates.at(-1)?.rate_reason,
    available: !!timeline, transportGaps: timeline?.transport_gaps || [] }
}

export function eventsAt(sample, seconds, threshold, mode = 'recorded') {
  const timeline = sample.streaming?.[mode]
  if (!timeline) return []
  const events = [{ time: 0, title: 'Recorded stream started', detail: '10s detection · 30s rate · updates every 5s. Independent segment.' }]
  for (const e of timeline.events.filter(e => e.time <= seconds)) events.push({ time: e.time,
    title: e.event === 'reset' ? 'Stream reset' : e.status === 'breathing' ? 'Breathing evidence detected' : e.status === 'none' ? 'Insufficient breathing evidence' : 'Window not assessable',
    detail: e.event === 'reset' ? e.reason : `${e.window_start.toFixed(1)}–${e.window_end.toFixed(1)}s window · ${e.reason.replaceAll('_', ' ')}${e.rate_bpm != null ? ` · ${e.rate_bpm.toFixed(1)} BPM (experimental)` : ''}` })
  let previous = null
  for (const s of timeline.snapshots.filter(s => s.time <= seconds)) {
    if (s.stream_state !== previous && s.stream_state === 'disconnected') events.push({ time: s.time, title: 'Input timed out — old results hidden', detail: 'No packets for 2s. Recovery requires fresh windows.' })
    if (previous === 'disconnected' && s.stream_state === 'warming_up') events.push({ time: s.time, title: 'Input resumed — collecting fresh data', detail: 'Previous evidence is not reused: detection needs 10s, rate needs 30s.' })
    previous = s.stream_state
  }
  for (const gap of sample.sampling_gaps || []) if (gap.start <= seconds) events.push({ time: gap.start, title: 'Recorded sampling gap', detail: 'Affected windows are assessed separately.' })
  if (mode === 'interrupted' && seconds >= 12) events.push({ time: 12, title: 'Simulated packet loss started', detail: 'Packets from 12–15s are discarded. This is not an original recording fault.' })
  return events.sort((a, b) => b.time - a.time)
}
