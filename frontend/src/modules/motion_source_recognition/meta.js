// Identity of this module. `navLabel` is the short name in the top menu; `title` is used everywhere else. Shown in the navigation, on the landing cards and in the page header.
// `summarize(frame)` is the short text shown for one frame on the landing page and in the fusion table.
// `order` sets the position in the navigation and on the landing cards.
// `role` tells the shared fusion code which input this module provides.
export default {
  order: 1,
  navLabel: "Motion source",
  title: "Motion source recognition",
  role: "motionSource",
  blurb: "Human, Pet, IRobot or Fan: who is disturbing the signal.",
  owner: '',
  summarize: (f) => `${f.label} ${Math.round(f.confidence * 100)}%`,
}
