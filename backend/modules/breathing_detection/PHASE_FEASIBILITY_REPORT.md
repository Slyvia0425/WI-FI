# Complex-CSI phase feasibility audit

## Decision

Completed the [frozen phase feasibility protocol](PHASE_FEASIBILITY_PROTOCOL.md).
The release contains genuine complex measurements, but **none of the three tested
simple phase representations is suitable for replacing the current amplitude
rate estimator**. All five grouped inner validations choose the unchanged
amplitude baseline. No phase model, amplitude/phase fusion, LSTM retraining or
production/UI integration is performed.

This is a negative result for these specific diagnostic transforms under the
stated validity policy, not proof that every possible phase method is unusable,
that phase was fabricated, or that a particular hardware fault caused errors.
Physical phase calibration cannot be verified from the available metadata.

## Source audit and constraints

All **467** files have finite `complex64` CSI with shape [1500,234]. Nonzero
imaginary-value fractions range 91.83–95.80% (median 93.53%); these are not merely
real amplitudes relabeled as complex. Earlier complex-magnitude checks already
matched the supplied amplitude within float32 precision.

Every file has some exact zero coefficients. None has a carrier that is zero
throughout the entire file, but 98–138 carriers per full source file contain at
least one undefined/invalid phase coefficient. Zero coefficients constitute
0.209–1.026% of cells (median 0.632%). This is **not** evidence that half the
measurements are zero: rare zeros are spread across many carrier trajectories.

For this conservative pilot, a carrier is excluded from phase processing if any
coefficient in the native interval bracketing the current 30s grid is nonfinite
or magnitude <=1e-12. No angle(0) is treated as a measurement, no undefined phase
is interpolated, and no reference chooses carriers. Under this complete-window
policy, valid assessed windows retain **96–136 of 234 carriers, median 116**,
and 77–126 originally adjacent valid pairs. Removing a column never makes its
two nonadjacent neighbors an artificial adjacent pair. This policy can discard
usable portions of intermittently invalid traces; segmented phase processing
was not tested and the score should not be generalized to every validity policy.

The available metadata fields are file/room/subject/posture/environment/session/
trial/segment identifiers, sampling rate, duration, point count and carrier count.
They do not supply a physical subcarrier index/frequency vector, antenna mapping,
phase-reference calibration, AGC or packet-status traces. A [time,234] array does
not establish independent antenna pairs. Therefore no inter-antenna CSI ratio is
invented, and column-index detrending is not described as verified physical
frequency-domain phase sanitization.

## Phase continuity diagnostics

Among valid trajectories, per-window median absolute wrapped temporal increment
ranges 1.501–1.609 radians, with median **1.557 rad (89.2 degrees)**. The fraction
of increments exceeding pi/2 has median **49.46%** (range 47.74–51.13%). Across-
carrier temporal-increment circular coherence has a per-window median with
dataset median **0.072** (range 0.058–0.199). The corresponding median adjacent-
column phase-difference circular coherence is **0.504** (range 0.432–0.714).
These statistics are over all 467 source windows, not just the 394 scored ones.

This shows substantial temporal phase variation and does not support assuming
one stable, shared temporal offset explains everything. It does not identify
the hardware mechanism, motion or the quality of any individual respiratory
component. Wrapped increments cannot reveal unique true increments beyond pi;
ordinary unwrap imposes a branch-continuity assumption rather than recovering
otherwise unobservable turns. A small wrapped jump alone would not certify
calibration either.

## Fixed transforms, no parameter search

All native phase operations use complex128 arithmetic after inventorying the
original complex64 source; no extra source precision is implied.

1. **Raw phase:** unwrap angle(H) over native time for fully valid carriers.
2. **Adjacent phase:** unwrap angle(H[k+1] conj(H[k])) over native time, only for
   valid originally neighboring columns. This algebraically cancels a common
   time-varying rotation, but not all frequency-dependent effects. It can also
   cancel useful respiration that is shared between the two columns.
3. **Index-affine residual:** unwrap across valid ordered columns per timestamp,
   least-squares subtract an intercept and original-column-index linear slope,
   wrap the residual, then unwrap over native time. Original indices are retained
   when columns are dropped. This is a diagnostic nuisance projection with
   potentially wrong unwrap/affine assumptions, **not certified calibration**;
   useful physical changes can be removed as well.

Real phase traces are interpolated to the unchanged 900-point/30Hz grid without
extrapolation, extra future tail or cross-file concatenation. The unchanged
unified spectral estimator reads out a BPM from each representation. It
standardizes/detrends phase traces just as it does amplitude; no alternative
phase-specific estimator or learned phase weights are introduced.

All **394 existing timestamp-quality-passing** windows have nonflat readouts for
all three transforms. The original 73 exclusions stay excluded. Flat or missing
representations would be recorded unavailable and counted in good-output harm,
not silently filled with the amplitude answer. Baseline/reference recomputation
matches the existing stored outputs exactly.

## Results

Good means originally <=2 BPM error (365 windows), and large means >5 BPM.

| Readout | Amplitude baseline | Raw phase | Adjacent phase | Index-affine residual |
|---|---:|---:|---:|---:|
| Available valid windows | 394 | 394 | 394 | 394 |
| MAE, BPM | 0.683 | 4.812 | 6.957 | 7.339 |
| RMSE, BPM | 2.466 | 6.086 | 9.821 | 9.224 |
| Recording-macro MAE, BPM | 0.702 | 4.841 | 6.432 | 7.903 |
| >5 BPM errors | 19 | 159 | 188 | 225 |
| Originally accurate outputs made inaccurate | 0 | 262 | 245 | 302 |
| Original large errors corrected to <=2 BPM | 0 | 6 | 2 | 3 |
| New large errors | 0 | 147 | 176 | 213 |
| Original 19 errors with a reference-near local peak | 18 | 8 | 3 | 4 |

No rate is withheld in the actual 394-window run. All three alternatives are
worse on MAE for **every** participant:

| Subject | Amplitude MAE | Raw phase | Adjacent phase | Index-affine residual |
|---|---:|---:|---:|---:|
| 01 | 0.491 | 4.357 | 8.462 | 5.390 |
| 02 | 0.418 | 4.975 | 6.475 | 8.316 |
| 03 | 0.727 | 4.359 | 5.514 | 8.889 |
| 04 | 0.879 | 6.275 | 7.330 | 3.797 |
| 05 | 1.519 | 5.104 | 7.417 | 7.617 |

There are isolated correct phase estimates, but extensive new errors and harm
do not support promoting them. Reference-near peaks and individual matches
are descriptive; chance matches or unwrap-induced low-frequency structures have
not been ruled out. They do not establish recovered physiological information.

### Grouped diagnostic

The frozen whole-record leave-person-out protocol selects among amplitude and
the three phase readouts using only inner-validation recording-macro MAE, full
coverage and zero originally accurate harm. Record/person disjointness and
once-only coverage are checked. **All five folds select amplitude.**

The pooled selected outcome is therefore the unchanged MAE 0.683, 19 large
errors, zero newly harmed accurate outputs, 394/467 total quality coverage.
`outer_safety_pass=true` is only the unchanged fallback;
`development_support_pass=false` because there is no benefit. No fold-dependent
algorithm is deployed. This assessment uses already observed, overlapping
development segments, not a new independent test; grouping keeps confirmed
within-record overlaps together.

The readouts compare complete pipelines, not pure phase-versus-amplitude
information capacity: phase uses its complete-window valid-carrier policy while
the existing amplitude baseline retains its existing carrier policy. No new
matched-mask amplitude control is added after inspecting results, and this
limitation precludes attributing the entire deficit solely to phase modality.

## Fixed example figures

The first three original large errors sorted by severity and the previously
known contradictory accurate case were fixed before extraction:

| Case | Reference | Amplitude | Raw phase | Adjacent phase | Index residual | Figure |
|---|---:|---:|---:|---:|---:|---|
| 03 Freefaller Interference 001 | 16.70 | 41.31 | 17.14 | 20.21 | 34.28 | [01](outputs/phase_feasibility_v1/case_01.png) |
| 04 Freefaller Interference 002 | 23.73 | 23.73 | 16.26 | 18.02 | 20.65 | [02](outputs/phase_feasibility_v1/case_02.png) |
| 04 Soldier Interference 001 | 21.09 | 6.15 | 12.30 | 30.76 | 21.09 | [03](outputs/phase_feasibility_v1/case_03.png) |
| 05 Yearner Interference 002 | 10.99 | 28.12 | 18.02 | 13.62 | 18.46 | [04](outputs/phase_feasibility_v1/case_04.png) |

Each figure includes the first valid raw-phase trace, circular increments,
resampled diagnostic traces, normalized mean spectra, respiratory reference and
wrapped phase across the first 32 valid columns. Illustrative carriers are never
selected using the reference. Case01 was visually inspected. Dense phase changes
and separately normalized spectra must not be interpreted as causal source
identification or physical energy improvements.

## Next boundary

Freeze this result and retain the original amplitude baseline. Do not concatenate
phase with the LSTM input, add phase/amplitude fusion or sweep phase-unwrapping/
index-projection thresholds on these failures. Having complex64 values is not
equivalent to having a validated useful phase representation.

Further phase work would require additional acquisition knowledge/calibration
evidence or independently recorded calibrated complex CSI with respiratory
reference. The current audit cannot promise that those additions would solve
interference, and it does not justify labeling the entire dataset unreliable.
No new external data collection/download is undertaken here.

## Reproduction, provenance and tests

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/wifi_motion_mpl_20261029 .venv/bin/python -m backend.modules.breathing_detection.evaluation.audit_phase_feasibility
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s backend/modules/breathing_detection/tests -q
```

Implementation is `preprocessing/phase_diagnostics.py`; runner is
`evaluation/audit_phase_feasibility.py`. No production inference imports added.
Protected local ignored output `outputs/phase_feasibility_v1/` contains
`config.json`, `source_audit.json`, `window_predictions.json`, `metrics.json`
and four PNGs. Completed output cannot be overwritten by the runner.

Protocol/source/code and protected original unified results/checkpoint/replay/
frontend lock hashes are recorded before extraction. All 467 raw file hashes
are checked before and after. The full suite passes **158 tests**, including
12 new tests for zero/invalid phase, true adjacent-pair preservation, common-
rotation cancellation, regular-case index-affine invariance, removal of useful
common respiration, synthetic frequency recovery, read-only extraction, shape/
timestamps/no extrapolation, extra-future-tail isolation and unwrapping ambiguity
above pi. These synthetic checks establish software behavior, not physical
calibration or physiological validity. Existing unrelated changes are preserved.
