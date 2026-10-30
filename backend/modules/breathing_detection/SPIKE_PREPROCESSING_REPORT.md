# Raw spike-source audit and robust preprocessing comparison

## Decision

Completed the source audit and fixed comparison in
[`SPIKE_PREPROCESSING_PROTOCOL.md`](SPIKE_PREPROCESSING_PROTOCOL.md).
**There is an average-error improvement, but the candidate is not adopted.**
Nested development selection reduces MAE from 0.683 to 0.581 BPM, but large
errors decrease only from 19 to 18, and one of 365 originally accurate estimates
becomes a large error. It fails both the prespecified zero-harm condition and
the >=25% large-error reduction requirement. The current application therefore
remains on the unchanged unified estimator (0.683 BPM / 19 large errors on the
previously observed development set). No frontend, LSTM, reference, raw data,
timestamp gate or deployed inference path is changed.

## Source findings

### Original measurements, not a preview or interpolation artifact

All 467 MAT files contain `csi_complex` as well as `csi_abs`, each with 1,500
samples and 234 subcarriers. Their magnitudes agree within single-precision
tolerance; the maximum absolute difference is 0.00390625 on large raw values.
This is not evidence of a broken amplitude conversion. Complex CSI is available
for future investigation, but the stored 2D shape does not by itself provide
independent antenna pairs or establish phase calibration.

The six top-level fields are complex CSI, amplitude CSI, CSI time, respiratory
reference, reference time and metadata. Metadata inventory across all 467 files
contains file/room/person/posture/environment/session/trial/segment identifiers,
sampling rate, duration and point/subcarrier counts. It does not provide AGC,
RSSI or packet-status traces needed to attribute a spike to a specific hardware
mechanism. Physical movement, acquisition effects and release processing cannot
be distinguished from amplitude excursions alone.

Independent timestamp interpolation reproduces the current loader **exactly for
all 467 files**. All aligned starts equal the first CSI timestamp; no unexplained
alignment offset is found. Original timestamps are strictly increasing. Linear
interpolation is bounded by its bracketing values and cannot create overshoots
beyond source amplitude ranges, though it can alter peak shape and spectra.
This audit does not prove the resampling policy is optimal or rule out aliasing;
it finds no discrepancy in its implementation.

Using the predeclared descriptive excursion criterion (>6 robust scales on at
least half the carriers), 86 source windows and 66 resampled windows contain a
flagged common excursion. This is not the number of physically moving windows;
other windows can still contain isolated carrier spikes. Median intervals before
flagged excursions and before other frames are approximately 20.52 and 20.48 ms.
These medians do not establish a general packet-gap explanation, nor rule out
individual timestamp-related events. The 73 quality-excluded windows stay excluded.

### Repeated peaks: source windows overlap

Repeated exact extreme-vector signatures were found only within the same
recording groups. A follow-up descriptive audit checked **exact complex-valued
suffix/prefix sequences**, not just similar amplitude or equal magnitudes:

| Consecutive files within a recording | Pair count |
|---|---:|
| Exact overlap of 1,200 / 1,500 points | 195 |
| Exact overlap of 900 points | 70 |
| Exact overlap of 600 points | 55 |
| Exact overlap of 300 points | 26 |
| No exact suffix/prefix overlap found | 36 |
| Total pairs checked | 382 |

All 346 overlapping pairs also have matching relative timestamps (tolerance
0.1 ms). This directly confirms reused source samples across files, consistent
with overlapping segmentation. It explains why one original extreme vector can
reappear in multiple files; repetition is not proof of clipping, AGC failure or
fabrication. There are no adjacent exact duplicate frames within individual files.
The follow-up did not change any filter, score, split or selected configuration.

These files must **not** be treated as 467 statistically independent experiments.
Earlier whole-recording grouping keeps all confirmed overlaps together, so the
identified overlap does not cross those train/validation/assessment partitions.
This check covers adjacent files within metadata recording groups; it is not an
exhaustive proof against every possible cross-record duplication. Operational
replay still resets at file boundaries; it does not concatenate the files.

## Fixed preprocessing comparison

Four candidates apply the same time-aware local median/MAD isolated-spike filter
before interpolation or after 30Hz interpolation, at thresholds six or eight.
The half-window is 0.1s, minimum five observations, with a fixed global-MAD floor.
Only runs <=2 points are linearly replaced between valid original brackets, with
bracket span <=0.15s and no original interval >0.1s. Boundaries/longer runs remain
untouched. No future file tail or independent segment enters a window's filter.
No reference waveform is filtered or used to decide which points to replace.

All four candidates produce rates for the same 394 valid windows. The 365 good
denominator below refers to the current **unified** estimator, not the older
349-good raw mean-spectrum baseline or 362-good fold-selected estimator.

### Primary grouped result

Rules were fixed before batch extraction. Inner validation selects lowest
recording-macro MAE among baseline/candidates with full coverage and zero harm
to accurate estimates. Outer assessment holds out a person and all their records.
No learning uses the outer person's reference rates.

| Measure | Unchanged unified estimator | Nested preprocessing selection |
|---|---:|---:|
| Retained quality-passing windows | 394 / 394 | 394 / 394 |
| Coverage of all segments | 84.37% | 84.37% |
| Pooled MAE, BPM | 0.683 | 0.581 |
| Pooled RMSE, BPM | 2.466 | 2.153 |
| Recording-macro MAE, BPM | 0.702 | 0.604 |
| Large errors >5 BPM | 19 | 18 |
| Error within 2 BPM | 92.64% | 94.42% |
| Previously accurate results spoiled | 0 / 365 | 1 / 365 |
| Originally large errors made accurate | 0 | 3 |

The MAE decrease is approximately 14.9%, but net large-error reduction is only
5.26%. No person's MAE worsens, yet average improvement does not satisfy safety.

| Outer person | Inner choice | Baseline MAE | Candidate MAE | Good estimates spoiled |
|---|---|---:|---:|---:|
| 01 | no preprocessing | 0.491 | 0.491 | 0 |
| 02 | raw threshold 8 | 0.418 | 0.356 | 0 |
| 03 | raw threshold 8 | 0.727 | 0.480 | 0 |
| 04 | no preprocessing | 0.879 | 0.879 | 0 |
| 05 | raw threshold 8 | 1.519 | 1.295 | 1 |

These are fold-selected experimental preprocessing choices, not a new runtime
method varying by person. The experiment uses historically observed development
data and the estimator was itself chosen retrospectively; results are not
prospective independent performance.

### All fixed candidates: descriptive controls

| Candidate | Pooled MAE | Large errors | Accurate estimates spoiled | Newly large errors |
|---|---:|---:|---:|---:|
| No preprocessing | 0.683 | 19 | 0 | 0 |
| Raw, threshold 6 | 0.593 | 19 | 2 | 4 |
| Raw, threshold 8 | 0.584 | 19 | 2 | 4 |
| After resampling, threshold 6 | 0.550 | 19 | 2 | 4 |
| After resampling, threshold 8 | 0.581 | 20 | 3 | 5 |

The lowest pooled MAE (post-resampling threshold six) does not reduce the total
large-error count and damages two accurate outputs. It is not a safe replacement
and must not be selected posthoc instead of the primary grouped outcome. Cleaning
removes some bad peaks but introduces others. Small per-cell modifications can
still affect many time frames: median raw-threshold-eight replacement is 0.72%
of cells, but 620 frames have at least one replaced carrier. It is not just the
removal of a handful of whole packets.

## Concrete improvements and failures

Plots compare raw/native and 30Hz signals, before/after candidates, spectra and
peak zooms. Reference markers are evaluation-only:

- [Subject03 Freefaller](outputs/spike_preprocessing_v1/preprocessing_case_01.png):
  baseline 41.31 BPM versus reference 16.70; all four candidates return about
  16.26 BPM. This is a clear example of useful spike suppression.
- [Subject05 Yearner segment002](outputs/spike_preprocessing_v1/preprocessing_case_02.png):
  28.13 becomes 18.90 BPM, but reference is 10.99; error remains >5 BPM.
- [Subject04 Soldier](outputs/spike_preprocessing_v1/preprocessing_case_03.png):
  prominent amplitude spikes disappear, but rate changes only from 6.15 to
  6.59 BPM while the reference is 21.09. A cleaner-looking trace does not solve
  the competing low-frequency peak.

The harmed accurate case is Subject05 Yearner Interference segment003: reference
and baseline are both 11.426 BPM; every candidate returns 17.139 BPM, error
5.713 BPM. In the selected raw-threshold-eight method, only 0.889% of amplitude
cells change, yet the selected spectral peak changes substantially. This is
evidence against automatic indiscriminate cleaning, not a reason to hand-write
an exception for that observed segment. The one harmed rate is 1/47 (2.13%) of
subject05's accurate estimates, also above the older per-person 1% harm budget.
There are three newly large errors in the nested outcome, offset by improvements
to previous large errors. No output is rejected to improve these metrics.

## Interpretation and next boundary

The audit narrows the origin: excursions exist in the released raw complex CSI,
not merely the frontend or a mistaken timestamp interpolation. Source overlap
explains repeated occurrences, not the physical cause of each excursion. Metadata
does not support a confident AGC/saturation/motion attribution. There is genuine
average-error improvement from isolated-spike suppression, but it is not yet a
safe or complete solution to the remaining large errors.

Freeze this result without raising thresholds, changing run-length limits or
adding observed-case exceptions. Keep the current unified estimator unchanged.
Before more amplitude-only gate tuning, investigate whether source complex CSI
provides independent phase/common-amplitude evidence or whether a separately
declared anti-alias/resampling control is justified. Neither has been validated
by this experiment, and available data do not guarantee a usable antenna-ratio
method. Any follow-up should preserve record grouping, report previously accurate
harm and avoid claiming independent validation on repeatedly observed files.

## Reproduction, artifacts and verification

```bash
.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_spike_preprocessing
MPLCONFIGDIR=/private/tmp/wifi_motion_mpl_20261029 \
  .venv/bin/python -m backend.modules.breathing_detection.evaluation.analyze_spike_preprocessing
.venv/bin/python -m backend.modules.breathing_detection.evaluation.audit_spike_overlap
```

Completed output directories/files are protected against overwrite. Ignored
`outputs/spike_preprocessing_v1/` contains config, raw audit with per-file hashes,
window predictions/edits, fold selections, metrics, descriptive analysis, exact
overlap ledger and three plotted examples. Config precedes real batch extraction
and records fixed rules, source/code hashes and protected artifact hashes.

The complete backend suite passes **116 tests**, including 13 additions covering
isolated versus sustained events, source boundaries/gaps, true low-rate signals,
read-only filtering, no extrapolation, no extra-tail dependence, placement shape,
complex/amplitude consistency, zero-harm selection, unavailable-output harm and
exact complex-sequence overlap versus merely equal magnitude. All 467 raw MAT
hashes and checkpoint/unified-rate/replay/lock hashes are verified unchanged.
Grouped IDs remain disjoint. `git diff --check` passes. No frontend export,
algorithm deployment, Git commit or GitHub push is performed.
