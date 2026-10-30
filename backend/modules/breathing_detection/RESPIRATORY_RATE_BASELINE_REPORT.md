# WiRe-Breath interpretable respiratory-rate baseline

## Method and selection

This standalone spectral baseline uses timestamp-resampled CSI amplitudes
and the fixed nested protocol `wire_breath_development_v1.json`. It does not
depend on the binary LSTM's predictions. All participants are known to be
breathing; output is excluded when the existing 100 ms gap rule fails.

Three predefined methods were compared at 10 and 30 seconds:

1. Mean normalized subcarrier spectrum.
2. Median normalized subcarrier spectrum.
3. Median individual-subcarrier spectral peak frequency.

Each stream is linearly detrended, standardized and Hann tapered. Power is
normalized by its 0.1--5 Hz energy, and rate is read from the dominant
0.1--0.7 Hz component (6--42 BPM). Zero padding refines numerical peak search;
it does not improve the actual frequency resolution of the observation.
Native resolution is approximately 6 BPM at 10 seconds and 2 BPM at 30 seconds.

The reference is the dominant frequency extracted from the aligned respiratory
sensor waveform using the same detrending/tapering. It is a waveform-derived
reference rate, not an independently annotated count. In particular, agreement
in dominant frequency does not establish waveform reconstruction, absence of
harmonic error, or clinical accuracy.

There are no fitted model parameters. Each fold selects estimator/window
length by minimum equal-record-weighted MAE on its inner validation records.
The outer subject is used only for assessment. All five folds selected the
30-second mean normalized spectrum; at fixed 10 seconds all five selected
the mean spectrum as well. This dataset has already informed development,
so these remain development cross-validation results.

## Assessment results

| Held-out person | 10s valid / total | 10s MAE, BPM | 30s valid / total | 30s MAE, BPM |
|---|---:|---:|---:|---:|
| 01 | 381 / 474 | 0.91 | 94 / 158 | 0.85 |
| 02 | 439 / 453 | 1.65 | 143 / 151 | 0.65 |
| 03 | 227 / 228 | 1.78 | 75 / 76 | 1.22 |
| 04 | 75 / 75 | 3.83 | 25 / 25 | 2.57 |
| 05 | 171 / 171 | 4.13 | 57 / 57 | 2.27 |
| Equal-person average | — | **2.46** | — | **1.51** |

The worst-person MAE improves from 4.13 to 2.57 BPM. Longer observations help,
but the quality coverage changes and must be reported alongside error.

| Pooled measure | 10 seconds | 30 seconds |
|---|---:|---:|
| Available windows | 1,401 | 467 |
| Quality-passing windows | 1,293 | 394 |
| Coverage | 92.29% | 84.37% |
| MAE, BPM | 1.91 | 1.16 |
| RMSE, BPM | 4.42 | 3.66 |
| Bias, BPM | -0.72 | -0.58 |
| Within 2 BPM | 80.67% | 88.58% |
| Quiet MAE, BPM | 0.95 | 0.35 |
| Interference MAE, BPM | 2.94 | 2.05 |

Equal-person and pooled-window averages differ because participants contribute
different numbers of windows. The former is the preferred person-generalization
summary; the latter describes the observed window mix.

## Same-segment comparison and limitations

Longer windows encounter more sampling holes. To check whether their apparent
advantage only reflects exclusion of difficult segments, compare the same 394
quality-passing original segments. They supply 1,182 non-overlapping 10-second
subwindows and 394 full 30-second windows. MAE is 2.02 BPM for the short
subwindows and 1.16 BPM for the long windows. The advantage remains, although
their reference rates cover different durations: this is a paired-source
comparison, not identical instantaneous reference targets.

Subject01 has only 59.49% coverage at 30 seconds, reflecting Room1 sampling
holes. A lower MAE at reduced coverage must not be interpreted as recovery
of those missing intervals. Keep rejected windows visible.

RMSE is considerably larger than MAE, indicating substantial outliers. Errors
are higher under motion interference and for Subjects04/05. The current
estimator has no validated confidence or motion rejection layer; a timestamp
quality pass alone does not guarantee a reliable rate. Synthetic tests verify
known periodicity and drift handling, not real-world accuracy.

## Decision and next analysis

Keep 30-second mean normalized spectrum as the initial rate-estimation
baseline. Retain 10-second results for latency/coverage comparisons. A future
30-second rolling implementation could update more often than every 30
seconds, but overlapping rolling updates were not evaluated here.

Before exposing rate estimates in the application, inspect large-error cases
and evaluate CSI-only peak concentration, subcarrier agreement and temporal
stability as potential confidence features using inner validation. Check
reference waveform quality and harmonic/subharmonic errors. Report rejection
coverage and errors together; do not tune confidence on outer assessment
errors. The binary breathing classifier and its checkpoint remain unchanged.

## Reproduction and outputs

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.evaluate_respiratory_rate \
  --output-dir backend/modules/breathing_detection/outputs/NEW_RATE_EXPERIMENT
```

The completed run is in `outputs/respiratory_rate_baseline_v1/`, ignored by Git:
`config.json` stores the settings/protocol fingerprint, `window_estimates.json`
stores per-window references and three candidate predictions, and
`metrics.json` stores inner selection, outer folds, environments and matched
segment summaries. This step reads no CSI-Bench official test data and changes
no learned model weights.
