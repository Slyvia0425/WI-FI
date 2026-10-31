export const CSI_DURATION = 10

export function csiReadingAt(sample, seconds, threshold) {
  const elapsed = Math.min(CSI_DURATION, Math.max(0, seconds))
  const complete = elapsed >= CSI_DURATION
  return { elapsed, complete, status: !complete ? 'collecting' : sample.probability >= threshold ? 'breathing' : 'none',
    probability: complete ? sample.probability : null,
    // Stored samples span 0..299 at nominal 30 Hz; no invented packet timestamps.
    visiblePoints: Math.min(sample.waveform.length, Math.floor(elapsed * sample.nominal_rate_hz) + 1) }
}
