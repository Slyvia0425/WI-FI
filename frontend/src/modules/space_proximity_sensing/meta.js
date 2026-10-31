// Identity of this module. `navLabel` is the short name in the top menu; `title` is used everywhere else. Shown in the navigation, on the landing cards and in the page header.
// `summarize(frame)` is the short text shown for one frame on the landing page and in the fusion table.
// `order` sets the position in the navigation and on the landing cards.
// `role` tells the shared fusion code which input this module provides.
export default {
  order: 5,
  navLabel: "Spatial sensing",
  title: "Spatial sensing",
  role: "space",
  blurb: "Place one to three objects and estimate occupied zones from simulated Wi-Fi signals.",
  owner: 'Chen Siyi',
  summarize: (f) => `${f.zone}${f.proximity ? `, ${f.proximity}` : ''}`,
  zones: {
    doorway: {
      x: 305,
      y: 372,
      label: "Doorway"
    },
    sofa: {
      x: 135,
      y: 325,
      label: "Sofa"
    },
    bed: {
      x: 505,
      y: 115,
      label: "Bed"
    },
    desk: {
      x: 510,
      y: 340,
      label: "Desk"
    },
    center: {
      x: 330,
      y: 200,
      label: "Room centre"
    }
  },
}
