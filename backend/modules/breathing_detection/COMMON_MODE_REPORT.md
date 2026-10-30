# Common versus relative-carrier representation pilot

## Decision

The fixed [common-mode protocol](COMMON_MODE_PROTOCOL.md) is complete. **Do not
replace the current rate estimator or automatically remove the common trace.**
The relative representation helps some errors, unlike the previous carrier-vote
rule, but its benefit does not generalize safely across participants/recordings.
Breathing-frequency information can be useful in either component.

All 394 existing quality-passing windows were compared. Raw files, respiratory
references, original unified estimator, frozen LSTM, 100 ms quality gate,
replay/frontend and prior experiment artifacts are unchanged. The 73 excluded
windows remain excluded. No GitHub commit/push or production integration.

## What was actually decomposed

Each 30-second [1,234,900] amplitude window is linearly detrended per carrier,
then each nonconstant carrier is divided by its own standard deviation, matching
the initial standardization in the current spectral estimator. Let Z(k,t) be
these standardized traces. No reference-based carrier sign, weighting or
selection is used:

- Common C(t) = equal-weight mean of Z across carriers.
- Relative R(k,t) = Z(k,t) − C(t).
- Z = C + R; the carrier mean of R is zero.

Both components are retained for inspection. The unchanged unified spectral
estimator is run on common-only and relative-only inputs as fixed contrasts.
There is no spike cleaning, PCA, component mixture, new bounds or threshold
search. Components without a nonconstant trace would be explicitly unavailable,
not silently assigned the original BPM; all components are available in this
394-window evaluation.

Energy fractions are mean-square ratios in **standardized input coordinates**,
not physical received power or respiratory-band energy. Common plus relative
fractions sum to one (max numerical error 4.44e-16), and reconstruction max error
is 3.55e-15. Spectral estimation separately restandardizes the remaining traces:
small residual noise can therefore acquire a large normalized peak. It must not
be interpreted as newly recovered physical signal energy.

This equal-mean decomposition is one limited representation. It is not a
physical separation of respiration and interference. Different carrier signs
or phase relationships can cancel respiration in the common trace, and shared
breathing effects can be removed from relative traces.

## Fixed representation results

Good means original error <=2 BPM (365 windows); large means >5 BPM. A good
output made inaccurate or unavailable counts as harm. Results are retrospective
development evidence, not independent final performance.

| Measure | Original | Common only | Relative only |
|---|---:|---:|---:|
| Retained quality-passing windows | 394 | 394 | 394 |
| MAE, BPM | 0.683 | 4.192 | 0.625 |
| RMSE, BPM | 2.466 | 7.940 | 2.427 |
| Recording-macro MAE, BPM | 0.702 | 4.667 | 1.072 |
| >5 BPM errors | 19 | 107 | 17 |
| Originally accurate results harmed | 0 | 113 | 5 |
| Original large errors corrected to <=2 BPM | 0 | 6 | 6 |
| New large errors | 0 | 94 | 6 |
| Changed estimates | 0 | 229 | 54 |

Relative-only improves pooled MAE by about 8.5%, but reduces large errors only
10.5%, below the prespecified 25% requirement. Eight original large errors move
to <=5 BPM (six to <=2), while six new large errors appear: 19−8+6=17. Five
originally accurate estimates become inaccurate (1.37%). Recording-macro MAE
**worsens**, meaning the sample-weighted average benefit is not representative
of equal-weighted recording outcomes. Overlapping segments within recordings
also mean that segment counts are not independent successes.

### Grouped diagnostic selection

Reuse the frozen whole-record leave-person-out protocol. Inner validation may
choose original/common/relative only with full coverage and zero accurate harm,
then lower recording-macro MAE; ties prefer fewer changes, then original. No
outer participant tunes a configuration, and no post-result mixture is added.

| Held-out subject | Inner choice | Original MAE | Selected MAE | Selected good harm | Large errors, original→selected |
|---|---|---:|---:|---:|---|
| 01 | Relative | 0.491 | 0.257 | 0 | 3→1 |
| 02 | Original | 0.418 | 0.418 | 0 | 4→4 |
| 03 | Relative | 0.727 | 0.369 | 0 | 4→3 |
| 04 | Relative | 0.879 | 1.441 | 2 | 1→3 |
| 05 | Relative | 1.519 | 1.742 | 3 | 7→6 |

Pooled selected MAE is 0.627, recording-macro MAE 1.072, large errors 17, and
accurate harm 5/365. Five original large errors are corrected to <=2 BPM, but
seven move to <=5 and five new large errors appear. Outer safety, no-person-
worsening and >=25% benefit all fail: `development_support_pass=false` and
`application_enabled=false`. Fold-dependent choices are only a diagnostic and
are not deployed participant-specific methods. Total quality coverage remains
394/467 = 84.37%; no rate is hidden to improve the metric.

## Signal evidence: useful, but not a universal split

In the 19 original large-error windows, a reference-near genuine mean-spectrum
local peak exists in 18 originals, 17 common traces and 18 relative spectra.
It becomes the highest local peak near the reference in six common and six
relative cases, versus zero originals. Across all 394 windows, reference-near
local peaks occur in 388 originals, 363 common and 391 relative representations.
These are reference-only diagnostics, **not** an oracle component selector.
Tiny sidelobes may count as genuine local maxima; mere peak presence does not
establish useful breathing evidence or physiological correctness.

Common standardized energy has a median 13.6% in original large errors and 9.0%
in originally accurate windows. Their ranges overlap extensively (4.4–73.0%
versus 0.8–60.5%). This is descriptive, not a validated rejection threshold.
Large common energy cannot by itself identify movement, gain changes or a fan.

### Example where relative representation helps

Subject03 Freefaller Interference segment001 has reference 16.70 BPM. Original
and common-only both select 41.31; relative-only selects 16.70. The standardized
common energy is 73.0%, with substantial synchronous excursions. Suppressing
that common trace exposes a stronger reference-near residual component, but
the data do not prove the physical source of the synchronous excursions.
See [paired signals/spectra](outputs/common_mode_v1/case_08.png).

Subject01 Semi Interference segment001 similarly moves from 6.15 to 20.65 BPM
versus reference 19.78. Subject04 Soldier Interference segment001 moves from
6.15 to 20.65 versus reference 21.09. These are real representation improvements,
not justification for universally removing shared dynamics.

### Examples where removing the common trace hurts

All five newly inaccurate baseline-good results are inspected after assessment
without changing any candidate. In four, common-only remains within 2 BPM of
the reference, showing why the common trace cannot simply be called noise.

| Subject / record | Reference | Original | Common | Relative | Figure |
|---|---:|---:|---:|---:|---|
| 04 Freefaller Interference 001 | 21.97 | 21.53 | 22.41 | 6.59 | [H1](outputs/common_mode_v1/relative_harm_01.png) |
| 04 Freefaller Interference 002 | 23.73 | 23.73 | 12.30 | 10.99 | [H2](outputs/common_mode_v1/relative_harm_02.png) |
| 05 Log Interference 001 | 11.43 | 12.30 | 12.30 | 19.78 | [H3](outputs/common_mode_v1/relative_harm_03.png) |
| 05 Log Interference 002 | 12.30 | 12.30 | 12.30 | 16.70 | [H4](outputs/common_mode_v1/relative_harm_04.png) |
| 05 Log Quiet 001 | 10.11 | 10.11 | 8.35 | 41.75 | [H5](outputs/common_mode_v1/relative_harm_05.png) |

H1 illustrates that a modest common energy fraction (9.8%) can still contain a
useful respiratory-frequency component. The residual selects a lower competing
peak although reference-near residual energy also remains. H2 illustrates that
both separated representations can be worse than their original combination.
No physical source label or “respiration removed” causal assertion follows from
these spectra alone.

The earlier spike-cleaning-harmed Subject05 Yearner Interference segment003
remains accurate here: original 11.43, relative 11.87, reference 11.43. Its
common-only estimate is 34.28. This is another reason to retain both components
for diagnostics, not declare a single universally correct component.

## Outcome and next boundary

This is a modest, mixed improvement in representation, **not** an adopted
algorithm improvement. Preserve the original estimator and do not retrospectively
choose whichever component agrees with the reference. We have not validated a
CSI-only rule that can safely choose between components; a common-energy cutoff
cannot be inferred from overlapping distributions or a few successful cases.

A future experiment would need a separately frozen, reference-free component
selection or multichannel representation, tested on independent recordings where
possible. It must preserve useful common breathing structure as well as relative
structure, not assume either is interference. No such new rule, PCA method,
phase pipeline or parameter sweep is implemented in this pilot.

## Artifacts and verification

Implementation: `preprocessing/common_mode.py`; fixed runner:
`evaluation/evaluate_common_mode.py`; post-assessment inspection:
`evaluation/analyze_common_mode.py`. No production inference imports are added.

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/wifi_motion_mpl_20261029 .venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_common_mode
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/wifi_motion_mpl_20261029 .venv/bin/python -m backend.modules.breathing_detection.evaluation.analyze_common_mode
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s backend/modules/breathing_detection/tests -q
```

Protected local ignored outputs: `outputs/common_mode_v1/config.json`,
`window_predictions.json`, `metrics.json`, 20 fixed target plots (19 original
large errors plus previous cleaning-harmed accurate case), `harm_analysis.json`
and five additional post-assessment harm plots. Completed runs refuse overwrites.
Figures for the successful Subject03 Freefaller case and harmed Subject04
Freefaller case were also visually inspected.

Protocol/code/source hashes are saved before extraction. All 467 raw file hashes
and protected original unified predictions/metrics, checkpoint, replay and
existing frontend lock hashes are verified. Whole-record split disjointness and
394 unique once-only outer outputs are checked. The full suite passes **139
tests**, including 10 new tests for reconstruction, energy accounting, identical
and opposite traces, flat carriers, invalid inputs, positive-scale/detrending
invariance, carrier permutation, input immutability, tiny-residual spectral
amplification, evaluation-only reference diagnostics and missing-component harm.
