# Original/common/relative consistency audit

## Decision: stop before inventing another fusion rule

Completed the [frozen three-route audit](THREE_ROUTE_PROTOCOL.md) on all 394
quality-passing windows. **No defensible safe distinction was found in the
examined consistency/peak cues. No new fusion rule is trained, evaluated or
deployed.** This follows the declared stop condition; it is not a claim that
every possible multichannel representation or model is ineffective.

The current original unified estimator remains unchanged (retrospective MAE
0.683 BPM, 19 >5 BPM errors). These are the existing baseline results, not new
fusion performance. No rate is withheld, no quality threshold/reference/data is
changed, and no frontend, LSTM or GitHub submission is made.

## Scope and method

Reuse the frozen `common_mode_v1/window_predictions.json` rather than changing
decomposition, rate estimation or preprocessing. All 394 windows are retained as
controls, including six original >5 BPM errors corrected to <=2 BPM by relative-
only and five original <=2 BPM outputs made >2 BPM by relative-only. The 73
original quality exclusions stay excluded. References define these groups only;
they do not construct CSI features or choose candidates.

`preprocessing/three_route.py` accepts only each component's availability, CSI
rate and CSI local-peak evidence. It records all three pairwise rate differences
and agreement within 2 BPM, plus local peaks in each representation matching
the original/proposed rate within 2 BPM. Matching prefers greater height, then
closer frequency, then lower BPM. Save matched height, prominence and rank;
missing evidence is explicit. Features do not accept sample identity or use the
previous reference-near peak diagnostics. No classifier, threshold grid or
reference-driven mixture is fit.

Relative-only changed 54 exact FFT-grid outputs in the previous experiment, but
only **17** changes exceed 2 BPM. All six corrected-large and five harmed-good
cases are among these 17. The other six are not hidden or treated as successes.
The 2 BPM agreement tolerance is fixed from the existing accuracy definition;
it does not make two transformed streams independent sensors.

## What does and does not separate the cases

### Original-spectrum support is necessary-looking, but not discriminative here

Every one of the six successful proposals has an original local peak near its
proposed rate. So does **every one of the five harmful proposals**. Across all
17 substantial proposals, 15 have that support.

| Original local peak near proposed rate | Six corrected large errors | Five harmed accurate results |
|---|---:|---:|
| Matched peak exists | 6/6 | 5/5 |
| Height range, allowed-band maximum normalized | 0.761–0.999 | 0.699–0.986 |
| Prominence range, same normalization | 0.117–0.797 | 0.338–0.619 |

Ranges overlap considerably. Requiring a more prominent candidate would reject
some successes while retaining failures. These observations do not prove all
possible feature combinations inseparable, but do not justify adding an observed-
case-specific cutoff to claim safe selection.

### Agreement does not establish which rate is correct

These are descriptive partitions of the 17 proposals, **not evaluated fusion
rules or estimates of independent validation accuracy**:

| Agreement pattern (within 2 BPM) | All proposals | Corrected large | Harmed good | Other |
|---|---:|---:|---:|---:|
| Original and common agree | 11 | 3 | 4 | 4 |
| Common and relative agree | 3 | 1 | 1 | 1 |
| Neither of those pairs agrees | 3 | 2 | 0 | 1 |

Original/common agreement would warn against four harms, but also veto three
genuine corrections. Common/relative agreement includes a correction, a harm
and a third worsening intermediate case (Subject02 Log Interference segment007:
original 13.62, common 23.29, relative 22.85, reference 11.43 BPM).

The three “neither agrees” windows are not a justified exception rule: all routes
disagree, so there is no reference-free basis here to trust relative by default.
Choosing that small post-hoc subgroup because two match the reference would be
case-fitting on already observed data. No such rule is added.

## A particularly important contradictory pair

| Evidence | Successful correction: Subject02 Yearner Interference 006 | Harm: Subject04 Freefaller Interference 002 |
|---|---:|---:|
| Original BPM | 23.291 | 23.730 |
| Common BPM | 11.865 | 12.305 |
| Relative BPM | 12.305 | 10.986 |
| Reference BPM (evaluation only) | 11.865 | 23.730 |
| Original local peak near proposal: rank | 2 | 2 |
| Original proposal-peak height | 0.980 | 0.986 |
| Original proposal-peak prominence | 0.292 | 0.571 |
| Common local peak near proposal: rank | 1 | 1 |
| Common proposal-peak height | 1.000 | 1.000 |
| Common proposal-peak prominence | 0.995 | 0.998 |

Both share the intuitive pattern “original about 24, common/relative about 12,
and a strong original secondary peak at the proposal.” In the harm case, several
apparently reassuring peak strengths are actually **greater**. The same cue can
mean correcting a double-rate estimate or creating a half-rate estimate. This
is not resolved by blind halving, two-route agreement or simply increasing a
minimum height/prominence. Relative and common are correlated transformations
of the same measured input, so their agreement is not independent corroboration.

Existing waveform/spectrum figures: [successful pair](outputs/common_mode_v1/case_07.png)
and [harm pair](outputs/common_mode_v1/relative_harm_02.png). The harm plot was
visually re-inspected in this audit. No reference label is changed.

## Why there is no new grouped fusion score

The declared workflow makes fusion conditional on a defensible CSI-only
distinction. This audit did not establish one, so it stops before freezing a
new rule. There is no new nested selection, no new outer assessment, no implied
zero-harm fusion result, and no adaptive exception based on the six successes
or five harms. The previous `common_mode_v1` grouped result remains a failed
representation-only candidate and must not be relabeled as a fusion evaluation.

These data and error cases have repeatedly been inspected. Even if a formula
were made to separate eleven selected cases, that would not establish general
reliability. Whole-record/person grouping must still be used if a future
separately justified candidate is tested; observed development data cannot
become an untouched independent test by renaming the experiment.

## Next boundary

Keep original/common/relative traces as offline diagnostics, not confidence
labels or an automatic chooser. Do not continue a height/prominence/agreement
threshold sweep on these failures. A new algorithmic attempt should bring
different, justified evidence (for example, a verified usable complex-CSI
representation) or independent controlled continuous recordings with respiratory
reference and interference labels. Neither is implemented or claimed validated
here. This audit does not prove that more data alone will solve the problem.

## Artifacts and checks

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m backend.modules.breathing_detection.evaluation.audit_three_route
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s backend/modules/breathing_detection/tests -q
```

Protected local ignored directory `outputs/three_route_audit_v1/` contains
`config.json`, `features.json` (all 394 IDs, CSI-only features and separately
stored evaluation group labels), and `summary.json`. The runner refuses
overwriting a completed run and records protocol/code/source/protected-artifact
hashes before extracting features. It verifies its cached source and protected
original results/checkpoint/replay/lock hashes after extraction. Raw datasets
are not opened or modified by this cached-evidence audit.

The full module suite passes **146 tests**, seven added for exact agreement
boundaries, matching/tie behavior, missing evidence, invalid input, read-only
feature extraction, ignoring unrelated reference fields, and the limitation
that matching agreement patterns are not correctness labels. Existing user
changes, including `frontend/package-lock.json`, are preserved.
