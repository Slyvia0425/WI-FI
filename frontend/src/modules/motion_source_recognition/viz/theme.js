// Colours of the motion source charts. The four class colours are slots 1 to 4 of the validated categorical palette (dark surface), in order;
// they pass the adjacent-pair checks on the card surface #0C1424. Because any two of the four can touch in the timeline, every cell also carries its letter.
export const CLASSES = ['Human', 'Pet', 'IRobot', 'Fan']
export const CLASS_COLOR = { Human: '#3987e5', Pet: '#d95926', IRobot: '#199e70', Fan: '#c98500' }
export const CLASS_LETTER = { Human: 'H', Pet: 'P', IRobot: 'R', Fan: 'F' }
// Text on a class-coloured fill: white or ink, by the luminance of the fill.
export const CLASS_TEXT = { Human: '#ffffff', Pet: '#ffffff', IRobot: '#04101C', Fan: '#04101C' }
export const shorten = (name) => ({ 'Reference: most frequent class of the room': 'Room lookup (reference)', 'Reference: always the most frequent class': 'Most frequent class (reference)' }[name]
  || name.replace('Random Forest', 'RF').replace('Logistic Regression', 'LR').replace(' on ', ' · '))
export const UNKNOWN_COLOR = '#898781'
// Series that are not classes (lines of the trade-off chart) take the next slots, 5 to 7.
export const SERIES = ['#d55181', '#008300', '#9085e9']
export const STATUS = { good: '#0ca30c', warn: '#fab219', critical: '#d03b3b', info: '#898781', neutral: '#898781' }
export const SURFACE = '#0C1424'
export const HAIR = 'rgba(159,179,204,0.2)'
export const AXIS = 'rgba(159,179,204,0.35)'
export const INK = { primary: '#E6F1FF', secondary: '#9FB3CC', muted: '#6F86A3' }

// One-hue sequential ramp for magnitude on the dark surface (more is lighter), steps 700 to 200 of the blue ramp.
const RAMP = ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4']
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16))
export function ramp(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1)
  const i = Math.min(RAMP.length - 2, Math.floor(x))
  const a = hex(RAMP[i]), b = hex(RAMP[i + 1]), f = x - i
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(',')})`
}
export const rampText = (t) => (t > 0.55 ? '#04101C' : INK.primary)
export const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`
export const num = (v, d = 2) => Number(v).toFixed(d)
