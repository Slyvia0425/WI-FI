# Motion-related signal-pollution screening: development result

## Decision

Completed the prespecified feature analysis and grouped validation in
[`MOTION_POLLUTION_PROTOCOL.md`](MOTION_POLLUTION_PROTOCOL.md).
**Do not enable this screening gate.** On held-out people, the inner-selected
rules withhold seven rates: six accurate outputs and one intermediate error,
with **zero of the 19 large errors intercepted**. Both the safety and benefit
requirements fail. The application continues to use the unchanged unified fixed
estimator; model, rates, frontend bundle/layout and all deployed gates remain
untouched. This experiment does not improve the current application.

## What was tested

All 394 quality-passing windows from 467 independent WiRe-Breath segments use the
same `mean_spectrum_boundary0_v1` rate estimator (365 accurate at <=2 BPM, ten
intermediate, 19 large at >5 BPM). Four CSI-only features measure normalized
transients, 1s energy bursts, 0.7–5 Hz broadband power relative to 0.1–5 Hz, and
absolute adjacent-subcarrier correlation loss. Feature extraction succeeds for
all 394. No reference waveform/rate, participant or environment label enters the
feature computation or gate decision.

Eight fixed rule candidates combine the four features at record-balanced 90th
or 95th percentile thresholds. Thresholds are calibrated on inner **training**
records only, with equal total weight per recording. Inner validation selects
rules under <=1% accurate withholding and <=15% total withholding, pooled and
per person. Outer assessment holds out a person and all their records. Source
metadata/split fingerprints, source/code hashes, fold IDs, thresholds and all
candidate audits are retained. No outer failure was used to modify a feature,
threshold quantile, budget or selection rule.

The unified estimator was chosen after retrospective development comparison;
the data have already been observed. Therefore this is further exploratory
development, not new unbiased final evaluation. Interference is a scenario
label, **not** reliable window-level movement ground truth. This analysis cannot
report physical-motion accuracy, diagnose a motion type or establish causality.

## Selected-rule assessment

| Measure | Unified estimator, no gate | Hypothetical selected gate |
|---|---:|---:|
| Retained / quality-passing windows | 394 / 394 | 387 / 394 |
| Coverage over all 467 segments | 84.37% | 82.87% |
| Accurate rates withheld / 365 | 0 | 6 (1.64%) |
| Large errors intercepted / 19 | 0 | 0 |
| Large errors retained | 19 | 19 |
| Pooled MAE, BPM | 0.683 | 0.680 |
| Pooled RMSE, BPM | 2.466 | 2.479 |
| Equal-person MAE, BPM | 0.807 | 0.810 |
| Recording-macro MAE, BPM | 0.702 | 0.720 |

The tiny pooled MAE decline is **not a useful improvement**: no large error is
removed, accurate rates are lost, coverage falls, and RMSE/person-/record-macro
errors worsen. The hypothetical outputs were saved for analysis, not applied.

| Outer person | Inner-selected rule | Accurate withheld / accurate | Large intercepted / large |
|---|---|---:|---:|
| 01 | transient, training Q95 | 0 / 91 | 0 / 3 |
| 02 | transient, training Q95 | 4 / 134 (2.99%) | 0 / 4 |
| 03 | no gate | 0 / 70 | 0 / 4 |
| 04 | transient, training Q95 | 2 / 23 (8.70%) | 0 / 1 |
| 05 | no gate | 0 / 47 | 0 / 7 |

Transient Q95 means both jump and energy-burst features exceed their own
train-calibrated Q95 thresholds, not a universal physical-motion cutoff. It
intercepts validation errors without accurate-result harm in three inner folds,
but that benefit does not transfer to their held-out people. Subjects02 and04
also fail the per-person accurate-withholding safety constraint.

### Fixed-rule controls: descriptive, not replacements

All controls use their fold's training-calibrated thresholds. They were fixed
before extraction but were not selected using outer assessment:

| Fixed family / quantile | Total withheld | Accurate withheld | Large intercepted | Outer safety |
|---|---:|---:|---:|---|
| transient / Q90 | 20 | 15 | 3 | fail |
| broadband / Q90 | 6 | 3 | 1 | fail per-person |
| either pair / Q90 | 26 | 18 | 4 | fail |
| three of four / Q90 | 5 | 3 | 2 | fail per-person |
| transient / Q95 | 10 | 6 | 3 | fail |
| broadband / Q95 | 0 | 0 | 0 | no useful action |
| either pair / Q95 | 10 | 6 | 3 | fail |
| three of four / Q95 | 1 | 0 | 1 | pass safety, fail benefit |

The last control intercepts only 1/19 (5.26%) large errors, below the prespecified
25% benefit requirement, and does not justify posthoc replacement of the nested
result. Even the most error-intercepting fixed control catches 4/19 (21.05%)
while discarding 18 accurate rates. No control is promoted.

## Matched signal-level error analysis

All 19 large-error targets have accurate controls **within the same original
recording**: same participant, posture, room and environment. No cross-scenario
fallback was needed. Up to three controls per target were selected by sample ID,
not feature similarity or the smallest error. These cases span 11 recordings and
reuse 26 distinct controls, so 19 targets are not 19 independent experiments.

| CSI feature | Accurate-window median (365) | Large-error median (19) | Targets above own control median |
|---|---:|---:|---:|
| Jump ratio | 167.83 | 176.69 | 10 / 19 |
| Energy-burst ratio | 19.91 | 20.60 | 7 / 19 |
| Broadband fraction | 0.716 | 0.785 | 11 / 19 |
| Adjacent incoherence | 0.386 | 0.225 | 9 / 19 |

Matched median differences are approximately +0.036 in jump ratio, -1.996 in
burst ratio, +0.0027 in broadband fraction and -0.0019 in incoherence. Thus the
apparent global broadband difference is much less distinctive within a record.
Incorrect outputs do not uniformly exhibit larger spikes, greater energy bursts,
or lower subcarrier coherence. Incoherence even has the opposite global median
direction from the hypothesized screening proxy. Do **not** simply reverse its
threshold after seeing these outer results.

Raw-file inspection of the three plotted target/control pairs confirms that
large amplitude excursions are already present in source CSI, not solely drawn
by the preview or introduced by resampling. Both accurate and inaccurate samples
can show excursions and substantial broadband power. Their cause could involve
the signal/measurement process or actual activity; available labels cannot
distinguish these possibilities. A large normalized ratio is not a movement label.

Plots of the three worst matched cases compare normalized CSI summary, median
relative 1s energy and spectrum. Reference markers are evaluation-only. These
are illustrative severity-selected examples, not a representative sample:

- [Subject03 Freefaller, error 24.61 BPM](outputs/motion_pollution_v1/matched_case_01_review.png)
- [Subject05 Yearner, error 17.14 BPM](outputs/motion_pollution_v1/matched_case_02_review.png)
- [Subject04 Soldier, error 14.94 BPM](outputs/motion_pollution_v1/matched_case_03_review.png)

The `_review` copies only improve plot layout; the original plots and numeric
diagnostics remain preserved. Each trace is independently normalized and is not
a reconstructed respiratory waveform. The comparison cannot establish a physical
interference source or validate reference-waveform-derived rates clinically.

## Next useful boundary

Follow-up: [`SPIKE_PREPROCESSING_REPORT.md`](SPIKE_PREPROCESSING_REPORT.md) now
records direct raw/complex and resampling checks, a fixed cleaning comparison,
and exact source overlap in 346/382 adjacent within-record file pairs. Same-record
controls may share original samples as well as environment/person characteristics.
The descriptive comparisons here are not independent-pair tests; the existing
whole-record split keeps confirmed overlap within partitions. No gate is promoted.

Freeze this negative result rather than adding more thresholds to the same 19
known failures. Before a new gate experiment, investigate the raw CSI excursions
and whether any are acquisition/representation artifacts rather than motion;
if justified, declare a small robust-preprocessing ablation against the unchanged
unified estimator. Do not assume despiking helps: suppressing a true respiratory
component could also damage accurate rates. Any such experiment must preserve
the grouped protocol and report damage to previously accurate outputs.

Physical-motion identification ultimately needs trustworthy motion annotations
or synchronized external evidence, especially to separate transient movement,
stable periodic interference and capture artifacts. New independent reference-
equipped recordings are needed for a stronger reliability claim. No existing
motion/periodic/temporal gate should be enabled based on these proxies.

## Artifacts and verification

```bash
.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_motion_pollution
MPLCONFIGDIR=/private/tmp/wifi_motion_mpl_20261029 \
  .venv/bin/python -m backend.modules.breathing_detection.evaluation.analyze_motion_pollution
```

Both runners protect completed artifacts. Outputs are ignored under
`outputs/motion_pollution_v1/`: `config.json`, `features.json`, `metrics.json`,
`window_results.json`, `diagnostics.json`, and matched-case PNGs. Review images
use the same plot function/data with layout-only adjustments; no rules are rerun
or modified to produce them. The protocol, code and tests are separate local
source files. No frontend export command is run.

The complete backend suite passes **103 tests**, including 13 new tests for
read-only/scale-invariant features, transients/noise proxies, anti-correlation,
physical adjacency, invalid input, equal-record quantiles, train-only calibration,
conjunctive/strict thresholds, per-person harm, immutable BPM, no-gate fallback
and matched-control priority. Frozen hashes verify that the checkpoint, unified
rate predictions/metrics, backend/frontend replay JSON and unrelated frontend
lockfile remain unchanged. Fold IDs are disjoint and cover all 394 assessed rows
once per fold; the 19 matched cases and disabled deployment state are verified.
`git diff --check` passes. All work stays local; no GitHub commit/push is made.
