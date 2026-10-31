# Stage-2 temporal / harmonic respiratory-rate reliability result

## Decision

Completed the fixed experiment in
[`RESPIRATORY_RATE_RELIABILITY_STAGE2_PROTOCOL.md`](RESPIRATORY_RATE_RELIABILITY_STAGE2_PROTOCOL.md).
**Do not enable the candidate.** All five inner validations select the no-gate
fallback. Temporal disagreement and half/double competition provide descriptive
signals, but the tested combinations do not safely distinguish accurate rates
from large errors. No frontend, model, rate output, timestamp threshold or
application gate has been changed.

## Scope and reproducibility

All 467 independent WiRe-Breath segments are considered; 394 pass the existing
30-second timestamp-quality rule and have reference/candidate rates. Feature
extraction succeeds for all 394. Each segment is subdivided into overlapping
0–20, 5–25 and 10–30 second windows. There is no inter-file concatenation.
Disagreement tolerance is fixed at 3 BPM, based on 20 seconds, not FFT padding.
Full-spectrum half/double local-peak tolerance is 2 BPM. Only CSI and the current
candidate rate enter feature extraction; reference rates enter selection and
evaluation only. These diagnostics do not identify a physical source of motion.

The rate under test is the previous boundary-correction experimental estimator,
not the original monitor baseline. In each outer fold its inner-selected method
is used consistently on both inner-validation and held-out-person assessment:
subjects01–03 use `boundary_prom0.0`, subject04 uses the baseline fallback, and
subject05 uses `boundary_prom0.1`. This is not a single deployment-ready method.
The estimator and reliability rule reuse inner validation, and the whole dataset
has previously been observed: results are exploratory development evidence,
not prospective independent final performance.

The runner writes config before accessing real stage-2 features/outer scoring.
Config contains fixed rules, budgets, split fingerprint and source/code hashes.
Existing output directories cannot be overwritten. Diagnostics are generated
separately after the run and are not used to revise or select any rule.

```bash
.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_rate_reliability_stage2
.venv/bin/python -m backend.modules.breathing_detection.evaluation.analyze_rate_reliability_stage2
```

Artifacts are ignored under `outputs/respiratory_rate_reliability_stage2_v1/`:
`config.json`, `features.json`, `metrics.json`, `window_results.json`, and
`diagnostics.json`. They retain per-fold validation/assessment IDs, candidate
audits, features, reasons and the 22 individual large-error cases. No replay JSON
is exported. Re-running a completed experiment intentionally raises an error.

## Main grouped outcome

Selection is restricted to inner validation, with <=1% good-rate withholding
and <=15% total withholding, both pooled and per person. Good means error <=2 BPM;
large means error >5 BPM, relative to the **current corrected estimator**.
The nine candidates are one temporal-only rule and eight temporal/harmonic
conjunctions. All folds choose no gate: active rules either fail the safety
budget or do not intercept a validation large error.

| Measure | Current corrected estimate | Nested stage-2 withholding |
|---|---:|---:|
| Retained quality-passing windows | 394 / 394 | 394 / 394 |
| Coverage of all segments | 84.37% | 84.37% |
| Pooled MAE, BPM | 0.823 | 0.823 |
| Pooled RMSE, BPM | 2.917 | 2.917 |
| Equal-person MAE, BPM | 1.190 | 1.190 |
| Accurate rates unnecessarily withheld | 0 / 362 | 0 / 362 |
| Large errors intercepted | 0 / 22 | 0 / 22 |
| Large errors retained | 22 | 22 |

`outer_safety_pass=true` only because nothing was withheld. This is **not** a
successful reliability method: `benefit_pass=false` and
`development_support_pass=false`. The prespecified benefit requirement is at
least 25% large-error interception without increasing retained MAE. No current
prediction is modified or promoted.

## Why lower retained MAE alone would be misleading

These are prespecified fixed-rule outer diagnostics, **not alternative selected
results**:

| Fixed rule | Total withheld | Accurate withheld / 362 | Large intercepted / 22 | Retained MAE |
|---|---:|---:|---:|---:|
| All 3 subwindows disagree | 17 | 12 (3.31%) | 3 (13.64%) | 0.698 |
| >=2 disagree + ratio >=0.5 + >=1 harmonic vote | 11 | 7 (1.93%) | 4 (18.18%) | 0.744 |
| >=2 disagree + ratio >=0.5 + >=2 harmonic votes | 5 | 5 (1.38%) | 0 | 0.830 |
| >=3 disagree + ratio >=0.5 + >=1 harmonic vote | 6 | 5 (1.38%) | 1 | 0.804 |
| >=3 disagree + ratio >=0.5 + >=2 harmonic votes | 3 | 3 (0.83%) | 0 | 0.828 |

All four ratio>=1 fixed controls withhold zero assessment windows. A ratio of
one is intrinsically difficult when the main estimator already selected the
strongest allowed peak; it is not a guarantee of weak interference. The lower-MAE
controls above fail pooled safety, the predefined benefit criterion, or both.
None should replace the nested no-gate result retrospectively. No extra grid,
retuned threshold, subject-specific exception or observed-case correction was
added after inspecting these results.

## What the features show

| Error group | Windows | All 3 subwindows within 3 BPM of candidate | >=2 subwindows disagree | All 3 disagree |
|---|---:|---:|---:|---:|
| Accurate, <=2 BPM | 362 | 304 | 26 | 12 |
| Intermediate, >2 to <=5 BPM | 10 | 1 | 3 | 2 |
| Large, >5 BPM | 22 | 7 | 9 | 3 |

Seven large errors have all three subwindow peaks close to the wrong candidate.
This supports a limited conclusion: agreement within a short window does not
establish physiological correctness. It does not prove that the wrong peak is
motion or identify its physical source. Conversely, 12 accurate full-window
estimates disagree with all three shorter-window peaks, so temporal inconsistency
is not sufficient for rejection. A longer window may estimate more accurately
than its short subdivisions, especially at low rates or under interference.

Of the 22 large errors, nine are near-half, two near-double, and eleven other.
Four of the nine near-half errors have zero temporal disagreements; only one
has at least two disagreements. A conjunction requiring temporal disagreement
therefore misses most near-half cases. Eleven large-error windows have a
half/double competitor ratio >=0.5, but so do 51 accurate windows. Competition
alone is likewise not a trustworthy correctness indicator.

The previously known Subject05 Yearner Interference segment002 case remains
unresolved: candidate 28.125 BPM versus reference 10.986 BPM. Its three shorter
windows all choose 6.152 BPM, rather than the reference. The temporal-only fixed
control flags it, but the nested rule cannot be promoted because that control
also rejects accurate rates. Replacing the full estimate with the stable short
window output would not solve this case. No exception was added for it.

## Interpretation and next boundary

This experiment rules out enabling this **specific small family** of rejection
rules under the stated budgets. It does not prove all temporal methods are
ineffective. Three overlapping subdivisions provide no long-duration continuity
or independent evidence, and the dataset remains breathing-positive with only
five people. Synthetic low-rate tests are software checks, not new clinical data.

Freeze the stage-2 result, retain explicit experimental status, and do not add
more thresholds on the same outer failures. Before choosing the final frontend
algorithm, define a single fixed respiratory-rate estimator and evaluate it as a
separate engineering/development step. A genuinely stronger reliability claim
needs independent continuous CSI with respiratory reference and difficult
interference; possible richer CSI features should be tested under a separately
declared protocol, not inferred from these 22 cases. Withholding BPM must never
overwrite independently supported breathing evidence.

## Verification

Eight added tests cover stable/low-rate signals, a synthetic half-candidate,
conjunctive rejection, missing features, nonfinite/flat/wrong-length input,
read-only feature extraction, pooled-versus-person safety, no-gate fallback,
current-estimator denominators and the stable-wrong-peak limitation. The full
module suite passes 83 tests. Source/checkpoint/config provenance is preserved;
there is no model retraining, GitHub submission or UI mutation.
