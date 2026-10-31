# Frequency profiles and frozen-model representations

## Design

The frozen seed-123 model was analyzed at two stages: the temporal mean of
its nonlinear 32-dimensional input projection, and the temporal mean of the
128-dimensional bidirectional LSTM output immediately before the classifier.
The latter is the actual feature vector consumed by the classifier.

Reference samples were drawn exclusively from the CSI-Bench training cache:
1,200 empty and 1,200 breathing windows, using seed 20261028. Within each class,
800 windows fit a mean and shrinkage covariance and 400 independently selected
windows calibrate the distance distribution. Distance is Mahalanobis distance
with 10% isotropic covariance shrinkage. These are diagnostic reference ranges,
not calibrated production rejection thresholds. No validation or test sample
was used to fit these references, and no classifier parameter was changed.

Raw frequency profiles retain all 234 original positions. Model representations
use the established 232-position interpolation. Normalized positions cannot
be assumed to represent identical absolute RF frequencies across devices.

## Frequency-axis observations

Per-position median mean and variance profiles are saved in the JSON artifact.
Correlations between log1p profiles of the target's 28 failed windows and the
comparison groups are:

| Comparison | Mean profile correlation | Variance profile correlation |
|---|---:|---:|
| Two successful windows in the same record | 0.999 | 0.951 |
| Subject02 Sitting Interference, 21 windows | 0.997 | 0.779 |
| Other successful Sitting, 138 windows | 0.988 | 0.803 |
| All other quality-passing successes, 1,255 windows | 0.985 | 0.552 |

The static amplitude pattern is very similar across groups. Variance patterns
differ more, but the target's two successful windows also share its variance
profile. Consequently, a static mean or variance profile alone does not
explain the failure. The model standardizes each original subcarrier over time
before frequency interpolation, so raw mean offsets and absolute variances
are largely removed before inference; they should not be mistaken for direct
classifier inputs.

Median adjacent-subcarrier correlations are:

| Group | Raw amplitude | Model input |
|---|---:|---:|
| Target failed | 0.847 | 0.899 |
| Same-subject Sitting Interference | 0.892 | 0.916 |
| Other successful Sitting | 0.664 | 0.814 |
| CSI-Bench training breathing reference | — | 0.941 |
| CSI-Bench training empty reference | — | 0.860 |

The failed group's model-input correlation overlaps both the training breathing
distribution (10th--90th percentile 0.717--0.993) and the successful same-subject
comparison (0.720--0.969). High adjacent correlation is not a sufficient
explanation or a usable rejection rule.

## Model representation observations

Distances below are to the CSI-Bench breathing reference, using separate
covariances for each representation stage. Distances from different stages
must not be compared directly as if they had the same scale.

| Group | Projection distance median | Above projection reference P95 | LSTM distance median | Above LSTM reference P95 |
|---|---:|---:|---:|---:|
| Target failed, 28 | 8.07 | 60.7% | 24.59 | 100% |
| Target successful, 2 | 5.80 | 0% | 12.19 | 50% |
| Same-subject Sitting Interference, 21 | 5.74 | 4.8% | 9.70 | 0% |
| Other successful Sitting, 138 | 5.34 | 1.4% | 9.40 | 2.2% |
| Other quality-passing successes, 1,255 | 4.85 | 5.7% | 9.37 | 7.5% |

The independent training calibration medians are 3.82 at the projection stage
and 7.28 at the LSTM stage. Thus a distinctive deviation is already visible
before the LSTM, and every failed target window is outside the breathing
reference's final-stage P95.

The failed target is also outside the empty reference for most windows: 85.7%
exceed its LSTM P95, with median distance 11.80. Therefore the model's low
breathing score should not be interpreted as evidence that these windows
closely resemble ordinary training empty rooms. Class-specific distances use
different covariances; their numerical magnitudes do not define nearest-class
classification.

## Interpretation and next experiment

The evidence supports a record-specific shift in the frozen model's learned
representation. It does not establish a literal subcarrier-axis misalignment:
static profiles are highly correlated and adjacent relationships overlap
successful controls. Time-varying cross-subcarrier structure and its learned
projection remain plausible causes. Final-layer distances are partly linked
to the binary decision by construction, so their association with errors
does not establish an independent cause. There is only one failed target
recording; its 28 windows are correlated observations, not 28 independent
replications.

Keep the frozen classifier, its threshold, and the 100 ms gap rule unchanged.
The next useful experiment is a diagnostic frequency-axis perturbation study
with fixed shifts/reversal applied equally to failed and successful controls.
It can test model sensitivity to ordering; it cannot select a deployable
correction by maximizing recall on this single observed failure. Any eventual
learned adaptation needs grouped validation and matching negative examples.

## Reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.analyze_wire_domain_shift \
  --output backend/modules/breathing_detection/outputs/\
wire_breath_frozen_seed123/domain_shift.json
```

The local JSON contains all 234-position profiles, distance quantiles,
calibration summaries, checkpoint hash, and group definitions in the script.
