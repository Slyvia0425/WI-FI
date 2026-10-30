# Temporary Chinese acceptance-review copy

English remains the default on each page load. The button above the breathing
workspace switches this module's monitor and research view to Chinese. The
platform-wide navigation and other modules remain English. No browser translator,
external translation service, dependencies, backend changes, data rewrites or
language persistence are used. Switching language preserves playback and model
state; translation only transforms React presentation elements.

Source: `review-language.js`; tests: `review-language.test.js`. Original English
copy is retained in the components. Technical identifiers in `<code>`, option
values, IDs, CSS classes, event handlers, numeric results and raw data are not
translated. Proper names and technical abbreviations may remain English.

To remove after acceptance:

1. Remove the temporary language state/button and localization wrappers/imports
   from `index.jsx`; return its original JSX directly. Remove the localization
   import/wrapper and language-dependent SVG label from `CSIBenchExperience.jsx`.
2. Remove the `language` prop and localization wrappers/imports from
   `ResearchView.jsx` and the `Signal`/`Wave` accessibility labels; keep the
   original English strings.
3. Remove this file, `review-language.js`, `review-language.test.js`, and the
   `.bm-review-language` / `[lang="zh-CN"]` rules in `streaming.css`.

Verify with `node --test src/modules/breathing_detection/*.test.js` and
`npm run build` from `frontend/`.
