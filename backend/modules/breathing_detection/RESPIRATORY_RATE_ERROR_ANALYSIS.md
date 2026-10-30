# Large respiratory-rate errors and reliability features

## Scope and definitions

This follow-up analyzes the selected 30-second mean-normalized-spectrum
baseline. Of 467 source segments, 394 pass the existing timestamp quality
rule. Large errors are absolute errors greater than 5 BPM; good estimates
have absolute error at most 2 BPM. Rates remain measured against the dominant
frequency of the supplied respiratory sensor waveform, not independently
annotated breathing counts.

No new estimator was selected from outer assessment errors. Feature directions,
large-error definitions and confidence-selection rules are explicit in
`evaluation/analyze_rate_errors.py`. This remains already-observed development
cross-validation.

## Error distribution and frequency ratios

There are 35 large errors among the 394 quality-passing windows (8.88%), from
19 original recordings. Thirty are Interference and five Quiet: large-error
rates are 30/188 (15.96%) and 5/206 (2.43%) respectively. The association with
recorded motion interference is strong descriptively, but conditions and
participants are not independently randomized.

Large errors by participant are 6, 7, 6, 4, 12 for Subjects01--05. Frequent
affected records include Subject01 Semi Interference (5), Subject02 Yearner
Interference (4), and Subject03 Yearner, Subject04 Soldier and Subject05
Soldier Interference (3 each).

| Frequency relation | Large-error windows |
|---|---:|
| Estimate within 2 BPM of half the reference rate | 15 |
| Estimate within 2 BPM of twice the reference rate | 2 |
| Other large errors | 18 |

Near-half errors dominate the simple integer-ratio cases. This is numerical
proximity, not proof of a true subharmonic: competing motion peaks, low-rate
drift and a reference waveform's own harmonics can produce similar ratios.
Blindly doubling low estimates would damage correct low-rate estimates and
cannot address the other 18 cases. The per-case JSON retains both rates for
waveform review.

## CSI-only reliability features

Features use CSI alone; respiratory reference is used only for error labels
and inner-validation threshold selection. All values below are medians.

| Feature | Good estimates (349) | Large errors (35) |
|---|---:|---:|
| 0.7--5 Hz / 0.1--5 Hz energy ratio | 0.851 | 0.846 |
| Energy concentrated within ±1/30 Hz of the selected breathing peak | 0.422 | 0.135 |
| Selected peak / strongest separated competing peak | 4.015 | 1.260 |
| Subcarriers agreeing within ±1/30 Hz of the selected rate | 0.326 | 0.226 |
| Standard deviation of three 10s rate estimates, BPM | 0.829 | 3.996 |

The high-frequency energy ratio overlaps substantially (good P10--P90:
0.802--0.865; large-error: 0.801--0.867). It is a coarse proxy rather than an
independent motion label; it does not support a standalone motion gate here.

Low peak concentration and weak separation from a competing peak are more
promising. Subcarrier agreement and within-window temporal consistency add
supporting evidence, but their distributions also overlap. A true change in
breathing rate can itself increase temporal spread, so this feature should
not automatically imply motion or absence of breathing.

## Grouped confidence-screening experiment

For each fixed outer fold, select one feature and one threshold using inner
validation only. Reject high-risk values to maximize large errors removed,
subject to retaining at least 80% of validation windows and rejecting at most
1% of good validation estimates. No rejection is a valid fallback.

All five folds selected peak concentration, with concentration cutoffs ranging
from 0.123 to 0.218. Each threshold is local to its fold and is not a single
production cutoff. Applied to the corresponding held-out people:

| Measure | Timestamp rule only | Plus selected confidence screen |
|---|---:|---:|
| Retained windows / all 467 | 394 (84.37%) | 340 (72.81%) |
| MAE among retained, BPM | 1.162 | 0.452 |
| Within 2 BPM among retained | 88.58% | 96.18% |
| Remaining large errors | 35 | 10 |

The screen rejects 54 additional windows, including 25/35 large errors
(71.43%), but also 22/349 good estimates (6.30%). Its 1% good-rejection budget
does not generalize to the outer people. Subject05 retains only 38/57
quality-passing windows; Subject04 still has 3 large errors among 24 retained
windows. Pooled error improvement therefore does not justify deployment.

## Decisions

Keep the existing baseline and timestamp rule. Treat peak concentration,
competing-peak ratio and temporal stability as candidates for a separate
respiratory-rate confidence mechanism. Do not promote the thresholds from
this experiment, and do not auto-correct half/double frequency cases.

Unreliable rate estimation is distinct from missing breathing evidence: a
future API may retain a breathing-present decision while withholding BPM.
This analysis does not change the existing breathing/not-assessable decision
layer.

Next, review the worst cases' CSI and respiratory-reference spectra/waveforms
to distinguish competing motion peaks from reference harmonic ambiguity.
Then test a conservative combination of concentration and competing-peak
evidence within inner validation, reporting per-person coverage and good
rejections. Any new rule remains a development experiment; new recordings are
needed for independent final evaluation.

## Artifacts and reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.analyze_rate_errors
```

The ignored output
`outputs/respiratory_rate_baseline_v1/large_error_analysis.json` includes every
large-error sample ID, estimates/reference, raw confidence features, error
category, recording distribution, per-fold selected thresholds, and rejection
tradeoffs. The analysis reads no CSI-Bench official test data or model weights.
