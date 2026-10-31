// Transparent rule-based fusion, as in the proposal (section 4.8).
// Input: the current frame of each module keyed by role (or null when missing):
//   { motionSource, activity, fall, breathing }
// Output: room state text, tone (normal | suspected | alert | uncertain) and rule notes.
// This is an expected-effect demo, not a joint ground truth.

export const HUMAN_MIN = 0.6

export const TONES = {
  normal: '#22E4FF',
  suspected: '#FFB347',
  alert: '#FF4D6D',
  uncertain: '#8B8FA8',
}

export function fuse({ motionSource: A, activity: B, fall: C, breathing: D }) {
  const notes = []
  if (!A && !B && !C && !D) {
    return { room: 'Uncertain', tone: 'uncertain', notes: ['No module evidence available'], humanGate: false }
  }

  const humanGate = !!A && A.label === 'Human' && A.confidence >= HUMAN_MIN
  const fan = !!A && A.label === 'Fan'
  if (humanGate) notes.push('Motion source: Human, so activity and fall are enabled')
  if (fan) notes.push('Motion source: Fan, so breathing trust is lowered')
  if (A && A.label === 'unknown') notes.push('Motion source: unknown, no strong conclusion')

  if (humanGate && C && C.state === 'alert') {
    notes.push('Fall detection: sustained high fall score')
    return { room: 'Fall alert', tone: 'alert', notes, humanGate }
  }
  if (humanGate && C && C.state === 'suspected') {
    notes.push('Fall detection: high score, waiting for next windows')
    return { room: 'Possible fall', tone: 'suspected', notes, humanGate }
  }
  if (humanGate && C?.state === 'not_evaluated') return { room: 'Occupied; fall not evaluated', tone: 'uncertain', notes: [...notes, 'B confidence gate did not run C; no negative fall decision'], humanGate }
  if (humanGate) {
    const act = B && B.activity && B.activity !== 'none' ? B.activity.toLowerCase() : 'activity unclear'
    const rest = D && D.status === 'breathing' ? ', breathing evidence' : ''
    return { room: `Occupied, ${act}${rest}`, tone: 'normal', notes, humanGate }
  }
  if (fan) return { room: 'No person evidence (fan)', tone: 'uncertain', notes, humanGate }
  return { room: 'Uncertain', tone: 'uncertain', notes: notes.length ? notes : ['Evidence too weak'], humanGate }
}
