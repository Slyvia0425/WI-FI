// Identity of this module. `navLabel` is the short name in the top menu; `title` is used everywhere else. Shown in the navigation, on the landing cards and in the page header.
// `summarize(frame)` is the short text shown for one frame on the landing page and in the fusion table.
// `order` sets the position in the navigation and on the landing cards.
// `role` tells the shared fusion code which input this module provides.
export default {
  order: 3,
  navLabel: "Fall detection",
  title: "Fall detection",
  role: "fall",
  blurb: "B confidence below0.6 triggers the frozen ResNet18 C.",
  owner: '',
  summarize: (f) => f.state === 'not_evaluated' ? 'C not evaluated' : `score ${f.fall_score.toFixed(2)}`,
  danger: true,
}
