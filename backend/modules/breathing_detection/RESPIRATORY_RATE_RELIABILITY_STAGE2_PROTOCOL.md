# Stage 2: frozen within-window temporal / harmonic reliability experiment

This specification is written before extracting real stage-2 features and
examining its outer results. No frontend, application estimator, LSTM weights,
threshold, timestamp rule, original labels or split assignment is changed.

## Hypothesis and fixed features

Can disagreement inside a 30s window, combined with half/double-rate competition,
identify >5 BPM errors without withholding valid rates? Use only CSI and the
candidate BPM. A reference waveform/rate is never a feature input.

For each quality-passing independent WiRe-Breath 30s file, compute mean normalized
spectra for 20s subdivisions at 0–20, 5–25 and 10–30s. These are correlated
diagnostics, not independent recordings or a longitudinal physiological trend.
Use the existing detrending/windowing/spectral normalization. Subwindow peaks
remain within 6–42 BPM; tolerance is fixed at 3 BPM (60/20), not the zero-padded
FFT bin spacing. Real low-rate evidence remains limited; 20s provides only two
cycles at 6 BPM. No bound raising or automatic rate doubling/halving is allowed.

Compute the number of subwindow peaks disagreeing with the current candidate by
more than 3 BPM. On the full spectrum, locate genuine local peaks within 2 BPM
of half/double the candidate, inside the allowed rate band and separated from
the candidate by more than 4 BPM. Record strongest competing-peak / candidate
power and the number of subwindow peaks supporting a half/double alternative
(within 3 BPM and not also within 3 BPM of the candidate). These are competing
spectral hypotheses, not proof of physical motion or a true harmonic.

Nine fixed rules: one temporal-only control (all three subwindows disagree),
plus eight conjunctions: disagreement count >={2,3}, competitor ratio >={0.5,1.0},
and harmonic subwindow votes >={1,2}. No-rule fallback is always available.
Missing features are explicitly recorded and do not reject; this is not evidence
of reliability. No motion classifier is fitted and no BPM value is changed.

## Fixed selection and assessment

Reuse `wire_breath_development_v1` with fingerprint validation and outer
leave-person-out / inner whole-recording validation. For each fold, use its
previously inner-selected boundary estimator on BOTH validation and assessment,
not other folds' selected predictions. This is a continuation of stage 1, not
the final unified deployment estimator; estimator and rule reuse the inner
validation set, so generalization remains exploratory.

Select on inner validation only: maximize >5 BPM errors intercepted, tie-break
by fewer accurate (<=2 BPM) outputs withheld, then fewer total outputs withheld.
Require <=1% accurate withholding and <=15% total withholding both pooled and
for every inner-validation person. No rule must beat the no-gate fallback unless
it intercepts an error. No outer case is used to change the nine rules or bounds.

Outer development support requires the same pooled/per-person safety limits,
at least 25% interception of existing large errors, and no worsening of retained
MAE. Passing does not activate the rule or prove prospective reliability.
Report coverage over quality-passing windows and all 467 segments, current-good
and current-large denominators, per-person and recording-macro errors, error
types and unavailable features. Fixed-rule outer controls are diagnostics only,
not a basis for retrospective selection. No independent final-test claim is made
for this historically observed dataset.

Run: `.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_rate_reliability_stage2`.
The runner preserves existing output directories and writes config (with source
and code hashes) before real feature extraction. No outputs are exported to UI.
