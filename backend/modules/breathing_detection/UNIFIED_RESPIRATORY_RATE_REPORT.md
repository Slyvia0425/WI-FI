# Unified respiratory-rate development reference v1

## Fixed implementation and integration

`preprocessing/unified_rate.py::estimate_unified_rate` is the single shared
estimator, identified as `mean_spectrum_boundary0_v1`. Its input is exactly one
finite `[1,K,900]` amplitude window at 30 Hz (30 seconds). It accepts no subject,
recording, fold, reference waveform or learned parameter input.

The existing detrended, Hann-tapered, normalized mean spectrum selects a baseline
peak within 0.1–0.7 Hz. If the result is interior, preserve it exactly. Only at
an exact FFT search endpoint (6.15234375 or 41.748046875 BPM for this window/grid)
inspect genuine local peaks using the existing wider 0.05–1 Hz spectral context.
Choose the highest allowed local peak, with minimum prominence fixed to zero.
No local peak means preserve the original baseline. A fallback is not a
reliability certificate. No automatic halving/doubling or new rejection occurs.

The 100 ms quality rule and independent breathing-evidence requirement remain
in the caller. `BreathingStream`, curated monitoring replay and current research
sample values now use this implementation. Streaming results carry estimator ID,
original baseline, correction flag and reason. The fixed rate requires 30s/30Hz;
unsupported stream rate configurations now fail explicitly instead of silently
changing the estimator. The LSTM, threshold, 10s detector and 5s cadence are
unchanged. Temporal and rate-reliability gates remain disabled.

Page layout is unchanged; only rate values, algorithm descriptions and metric
provenance are synchronized in English and temporary Chinese copy. Current
research-rate metrics are explicitly retrospective. Historical fold-selected
0.823 BPM / 22-error experiments remain separately labelled, not rewritten.
Per-sample historical rates remain in `historical_fold_selected_bpm`. The CSI-Bench
10s detection-only scenes and their honest false-positive example are preserved.
Stale backend exports with a different estimator ID fall back to the bundled
matching schema rather than mixing new labels with old rates.

## Recomputed full-data result

The choice was made after looking at previously observed development comparisons.
This run is **retrospective engineering/development verification**, not a fresh
holdout, nested-CV outcome, or evidence of prospective safety. Do not replace
historical prespecified grouped results with these better retrospective numbers.

All 467 independent WiRe-Breath segments were read again. The existing quality
check excludes 73; the shared estimator runs on all 394 remaining segments.
Every output exactly matches the previously preserved `boundary_prom0.0` fixed
control, verified against the declared source (tolerance 1e-9 BPM).

| Measure | Original mean-spectrum baseline | Unified fixed boundary0 |
|---|---:|---:|
| Retained quality-passing windows | 394 / 394 | 394 / 394 |
| Coverage over all 467 segments | 84.37% | 84.37% |
| Pooled MAE, BPM | 1.162 | 0.683 |
| Pooled RMSE, BPM | 3.663 | 2.466 |
| Equal-person MAE, BPM | 1.511 | 0.807 |
| Recording-macro MAE, BPM | 1.431 | 0.702 |
| Error within 2 BPM | 88.58% | 92.64% |
| Large errors >5 BPM | 35 | 19 |
| Originally accurate estimates spoiled / withheld | 0 / 0 | 0 / 0 |

| Participant | Baseline MAE | Unified MAE | Large errors before → after |
|---|---:|---:|---:|
| 01 | 0.851 | 0.491 | 6 → 3 |
| 02 | 0.648 | 0.418 | 7 → 4 |
| 03 | 1.225 | 0.727 | 6 → 4 |
| 04 | 2.566 | 0.879 | 4 → 1 |
| 05 | 2.267 | 1.519 | 12 → 7 |

Exactly 20 rates change. All 349 originally accurate (<=2 BPM) baseline estimates
remain accurate, and 16 originally large errors become accurate. **There is still
one newly large error from a previously intermediate error**: Subject05 Yearner
Interference segment002, approximately 28.125 BPM versus reference 10.986 BPM.
The improvement is not universal and 19 large errors remain. Interference retains
18 large errors (MAE 1.269 BPM); Quiet retains one (MAE 0.147 BPM). The gate
experiments failed on the older fold-selected estimator and are not evidence
that this new fixed estimator has validated confidence or rejection thresholds.

## End-to-end streaming verification

The full 467-file replay was rerun into a **new**, protected
`outputs/streaming_validation_unified_v1/` directory. Earlier baseline replay
artifacts in `streaming_validation_v2/` were retained. Audit confirms:

- all 2,342 detection events retain exactly the same window positions, quality
  decisions, probabilities, statuses and diagnostic temporal scores;
- all 384 previously available rates remain available, with exactly 20 BPM changes;
- all 467 terminal timeout checks hide the old current result;
- all 23 available normal-replay rate pairs in the curated 42-segment collection
  exactly equal the corresponding current research sample value;
- two CSI-Bench detection scenes remain present.

The 394-window estimator evaluation above is before breathing-evidence
withholding. The streaming count of 384 additionally requires the final 10s
classifier window to support breathing; these are different denominators and
must not be conflated. Replay transport remains simulated and files remain
independent, not a long continuous recording.

## Artifacts, regeneration and checks

Fixed-estimator output is protected under `outputs/respiratory_rate_unified_v1/`:
`config.json`, `metrics.json`, `window_predictions.json`, `excluded_windows.json`,
and `integration_audit.json`. Config records fixed specification, source hash,
implementation hash and retrospective limitations. Existing historical experiment
directories and checkpoint are not overwritten.

```bash
# Evaluate once; fails if this configured output already exists.
.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_unified_rate

# Refresh current bundled sample values/metric provenance, preserving other data.
.venv/bin/python -m backend.modules.breathing_detection.evaluation.export_unified_rate_dashboard
.venv/bin/python -m backend.modules.breathing_detection.evaluation.export_streaming_dashboard

# Full streaming run and audit once; existing configured runs are protected.
.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_streaming
.venv/bin/python -m backend.modules.breathing_detection.evaluation.audit_unified_rate
```

For a full dashboard rebuild, the unified evaluation must exist first, then run
`evaluation.export_dashboard` (which applies unified values before writing),
`evaluation.export_streaming_dashboard`, and `evaluation.export_csi_dashboard`.
`export_unified_rate_dashboard` is the lightweight refresh for an existing bundle;
it does not recompute/retrain the classifier or remove CSI/streaming extensions.

Verification: 90 backend tests, 17 frontend tests, frontend production build and
`git diff --check` pass. Tests include endpoint-only replacement, no-peak fallback,
height selection without prominence tuning, low-rate preservation, invalid/short
inputs, shared stream invocation, no early rate, unchanged detection behavior and
cross-view numeric/version consistency. Build retains the existing >500 kB chunk
warning; no new dependencies were installed. Work remains local, without GitHub
commit/push or layout redesign. Independent reference-equipped recordings are
still needed before stronger claims or enabling a reliability gate.
