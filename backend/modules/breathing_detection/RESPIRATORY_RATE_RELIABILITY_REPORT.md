# Conservative rate reliability experiment

The subsequent fixed within-window temporal/harmonic experiment is documented in
[`RESPIRATORY_RATE_RELIABILITY_STAGE2_REPORT.md`](RESPIRATORY_RATE_RELIABILITY_STAGE2_REPORT.md).
It also remains disabled: no nested fold selected a safe beneficial gate.

## Scope and fixed selection

This experiment builds on the boundary-only rate correction and evaluates
withholding an unreliable BPM without changing any breathing-presence decision.
It does not connect a rule to the application or assert a calibrated confidence
probability. The dataset is positive-only and historically observed development
data.

For each fixed outer fold, use that fold's previously inner-selected boundary
estimator for BOTH its validation records and its held-out person's assessment.
Do not use other folds' selected estimators when constructing validation
features: those selections may have seen the held-out person's reference.
Only CSI-only peak features are shared across folds.

Three features are evaluated at the actual corrected rate: concentration within
±1/30 Hz, subcarrier peak agreement within ±1/30 Hz, and power relative to the
strongest separated competing peak. A candidate withholds when at least two
or all three features are weak. Grids are concentration {0.1,0.15,0.2,0.25},
agreement {0.1,0.2,0.3}, and peak ratio {1.1,1.3,1.5}. No-gate is a fallback.

Inner validation maximizes rejected >5 BPM errors while rejecting <=15% of
quality-passing windows and <=1% of currently accurate (<=2 BPM) estimates.
Ties prefer fewer good and total rejections. Outer safety also requires <=1%
good rejection for each individual person, not just pooled data.

## Remaining error review

Before this gate, the corrected estimator has 22 large errors: 20 Interference
and two Quiet; nine are near-half, two near-double, and eleven other errors.
The newly introduced Subject05 Yearner Interference segment002 error remains
unrejected: selected rate 28.125 BPM versus reference 10.986 BPM, concentration
0.1415, agreement 0.2009 and peak ratio 1.1007. The selected Subject05 rule
requires all three weaknesses at concentration <0.15, agreement <0.3 and
peak ratio <1.1, so its ratio narrowly clears the cutoff. This known outer
case must not be used to move the cutoff to catch it retrospectively.

## Outer assessment result

| Measure | Corrected estimator | With reliability withholding |
|---|---:|---:|
| Rates retained / 394 quality-passing windows | 394 | 386 |
| Coverage of all 467 segments | 84.37% | 82.66% |
| Pooled MAE, BPM | 0.823 | 0.749 |
| RMSE, BPM | 2.917 | 2.837 |
| Within 2 BPM among retained | 91.88% | 93.26% |
| Large errors among retained | 22 | 19 |

Eight windows are withheld: three large errors, three intermediate errors,
and two currently accurate estimates. The rule catches 3/22 (13.64%) large
errors. Among the corrected estimator's 362 accurate windows, 2/362 (0.55%)
are unnecessarily withheld. The older uncorrected baseline had 349 accurate
windows; distinguish these denominators when reading inherited diagnostic
fields in the JSON.

| Outer person | Currently accurate | Accurate withheld | Retained / quality-passing | Large errors before -> after |
|---|---:|---:|---:|---:|
| 01 | 91 | 0 | 94 / 94 | 3 -> 3 |
| 02 | 134 | 1 | 138 / 143 | 4 -> 2 |
| 03 | 70 | 0 | 75 / 75 | 4 -> 4 |
| 04 | 20 | 0 | 25 / 25 | 4 -> 4 |
| 05 | 47 | 1 | 54 / 57 | 7 -> 6 |

Subject05 accurate-window rejection is 1/47 (2.13%), exceeding the 1% safety
limit despite acceptable pooled rejection. Subjects03 and04 select no gate
from inner validation. Thus the candidate fails outer safety and has modest
error interception. It is not promoted.

## Interpretation and decision

Weak spectral evidence can identify some unreliable outputs, but these simple
threshold combinations are not sufficiently stable across people. Incorrect
peaks can be locally plausible, while correct rates may have weak agreement.
Do not keep adding thresholds based on the remaining observed outer failures.

Freeze the present experiments and retain both the baseline and boundary-only
candidate as offline reference methods. The reliability layer remains
disabled. Before production integration, use a separately defined experiment
for richer temporal evidence or new recordings that include trustworthy
reference and hard interference; continue reporting coverage and per-person
good rejection. When an eventual validated rule withholds BPM, it must leave
independently supported breathing evidence available.

No local replay is exported as an enabled reliable-rate system because the
stated safety condition was not met. The current analysis does not change the
API, model weights, reference labels, binary gates or 100 ms timestamp rule.

## Reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.evaluate_rate_reliability
```

Outputs are ignored under `outputs/respiratory_rate_reliability_v1/`:
`config.json` (saved rules/budgets), `metrics.json` (folds and explicit
`outer_safety_pass=false`), and `window_results.json` (rate features, withheld
flag and reason). The runner refuses an existing configured experiment.
