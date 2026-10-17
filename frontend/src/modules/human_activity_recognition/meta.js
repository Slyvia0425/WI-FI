// Identity of this module. Stable module id comes from the folder name.
export default {
  order: 2,
  navLabel: "Human activity",
  title: "Human activity recognition",
  role: "activity",
  blurb: "Four stable states with probabilities, confidence, stability and transition handling.",
  owner: 'Member B',
  summarize: (f) => {
    const name = String(f.activity || 'unknown').replace(/^./, (c) => c.toUpperCase())
    if (f.stable) return `${name} ${Math.round((f.confidence || 0) * 100)}%`
    if (f.transition) return 'Transition'
    return name === 'Unknown' ? 'Unknown' : `${name} (unstable)`
  },
}
