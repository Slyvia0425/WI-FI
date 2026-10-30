# Conservative boundary-only rate correction

## Predefined rule and selection

Only exact original FFT search endpoints (6.15234375 and 41.748046875 BPM
for 900 points at 30 Hz) are reconsidered. Interior rates are preserved
exactly. Select the highest genuine local peak from the wider spectral context,
subject to a candidate minimum relative prominence. If no valid peak exists,
preserve the original rate. This fallback maintains coverage, but does not
make that unresolved boundary estimate reliable.

Candidate minimum prominences are 0, 0.05, 0.1 and 0.2; the baseline is always
eligible. Each fixed fold selects minimum inner-validation record-macro MAE,
with zero harm to originally accurate (<=2 BPM) validation estimates. Harm
includes both rejection and making a good estimate inaccurate.

The run reuses CSI-only local-peak features from the preceding experiment,
ignores its selected methods, and checks the protocol/cache fingerprint.
No new raw recordings, reference labels, LSTM parameters or API behavior
are introduced. All results are already-observed development cross-validation.

## Results

| Measure | Original baseline | Boundary-only, inner-selected |
|---|---:|---:|
| Retained / timestamp-quality-passing | 394 / 394 | 394 / 394 |
| Coverage over all 467 segments | 84.37% | 84.37% |
| Pooled MAE, BPM | 1.162 | 0.823 |
| Pooled RMSE, BPM | 3.663 | 2.917 |
| Equal-person MAE, BPM | 1.511 | 1.190 |
| Large errors >5 BPM | 35 | 22 |
| Within 2 BPM | 88.58% | 91.88% |
| Originally good estimates spoiled/rejected | 0 / 0 | 0 / 0 |

Only 18 predictions change, and all are original boundary maxima. Thirteen
original large errors become accurate within 2 BPM. One previously moderate
error becomes large, so the net reduction in large errors is 13 rather than
the total count of improved original errors. No interior prediction changes.

| Assessment person | Inner-selected minimum prominence | Baseline MAE | Corrected MAE | Large errors before -> after |
|---|---|---:|---:|---:|
| 01 | 0 | 0.851 | 0.491 | 6 -> 3 |
| 02 | 0 | 0.648 | 0.418 | 7 -> 4 |
| 03 | 0 | 1.225 | 0.727 | 6 -> 4 |
| 04 | Baseline fallback | 2.566 | 2.566 | 4 -> 4 |
| 05 | 0.1 | 2.267 | 1.750 | 12 -> 7 |

No person's MAE worsens, though Subject04 does not improve. Quiet pooled MAE
changes from 0.348 to 0.209 BPM and Interference from 2.055 to 1.496 BPM.

## Remaining risk and low-rate behavior

Subject05 Yearner Interference segment002 is the newly large error. Its
reference is 10.99 BPM. The boundary rate was 6.15 BPM (error 4.83), and the
selected prominence requirement instead returns 28.125 BPM (error 17.14).
Boundary restriction avoids the earlier modification of accurate interior
peaks, but cannot ensure the chosen alternative peak is physiological.

The dataset has only three quality-passing references below 10 BPM. All three
remain unchanged and accurate, with MAE 0.146 BPM. Synthetic 6, 6.5, 7.5 and
9 BPM tests pass within 1 BPM. This is limited low-rate evidence, not proof of
robustness across all low-rate scenarios.

The fixed zero-prominence boundary control reaches pooled MAE 0.683 BPM and
19 large errors without good-estimate harm. Its predictions equal the fixed
highest-local-peak control from the previous run: a bounded global maximum
that is already interior is itself a local peak, so zero-prominence height
selection ordinarily only changes endpoints. The new protection is most
relevant when prominence thresholds or other ranking rules would otherwise
override valid interior peaks. Do not replace the prespecified nested outcome
with the better outer control retrospectively.

## Decision

Keep this as a stronger experimental candidate than the previous mixed
height/prominence rule: it improves average error at unchanged coverage and
does not spoil any baseline-good example in the observed outer assessment.
It is not enabled in production. Twenty-two large errors remain, and retaining
an unresolved boundary result must not imply high confidence.

Next, freeze a simple candidate rather than expanding the parameter search.
Review whether competing-peak agreement can justify a conservative uncertainty
state for unresolved cases, and evaluate with fixed grouped validation and new
recordings when available. Do not automatically double rates or raise the
lower respiratory bound to suppress the observed failures.

## Reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.evaluate_boundary_rate_correction
```

The runner preserves configured results. Ignored artifacts are in
`outputs/respiratory_rate_boundary_only_v1/`: `config.json`, `metrics.json`,
and per-window predictions. Full module tests include interior invariance,
boundary/no-peak behavior, prominence fallback and genuine low-rate signals.
