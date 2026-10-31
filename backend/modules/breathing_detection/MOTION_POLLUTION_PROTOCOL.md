# Fixed signal-pollution screening experiment after unified rate v1

Written before real pollution-feature extraction and rule assessment. The fixed
rate estimator remains `mean_spectrum_boundary0_v1`. No frontend, application
gate, LSTM threshold, timestamp-quality rule or original data is changed.

## Scope

Use all 394 quality-passing 30s WiRe-Breath windows, not only the 19 known large
errors. Eighteen large errors occur in Interference, but this is a scenario
label, not per-window motion ground truth. Neither the scenario nor subject,
posture, reference waveform, reference rate or candidate error enters feature
extraction or gate decisions. The aim is to explore *motion-related signal
pollution proxies* for unreliable BPM; no physical-motion classifier/accuracy
or causality is claimed. This is already-observed development data, and the
fixed estimator itself was chosen after retrospective comparison.

## Fixed features and candidate rules

Detrend each original amplitude subcarrier and standardize its temporal scale.
Compute four fixed higher-is-more-anomalous features:

1. Jump ratio: per-subcarrier maximum absolute first difference divided by the
   median absolute difference (floor 1e-6), then 90th percentile across channels.
2. Energy-burst ratio: maximum / median RMS of 30 disjoint 1s blocks (floor 1e-6),
   then 90th percentile across channels.
3. Broadband fraction: average per-channel spectral power above 0.7 through 5 Hz
   divided by power from 0.1 through 5 Hz. Use existing Hann/detrended normalized
   spectra; the band relates to the estimator's upper respiratory search bound,
   not a physiological guarantee that all higher frequencies are motion.
4. Adjacent incoherence: one minus median absolute correlation of physically
   adjacent, nonconstant, standardized subcarrier pairs. Anti-correlation counts
   as coherent; flat channels are not bridged into supposedly adjacent pairs.

Record insufficient/nonfinite features explicitly; no-rule fallback does not
reject them, and missing features do not constitute evidence of reliability.

Thresholds are per-feature record-balanced 90th or 95th percentiles of **training
CSI features only**. Each original recording has total weight one irrespective
of how many valid windows it contains. Use an inverse weighted empirical CDF.
Eight rules = two quantiles times four fixed families:

- both jump and energy high;
- both broadband and incoherence high;
- either of the above pairs;
- at least three of the four features high.

High means strictly above the calibrated value; no equality-triggered mass
rejection. No new spectral/temporal correctness feature or extra threshold grid
will be added after seeing this experiment's outer failures.

## Calibration, selection and assessment

Reuse immutable `wire_breath_development_v1`, outer leave-person-out and inner
whole-recording train/validation. Verify sample fingerprint and recording/person
disjointness. Calibrate each candidate's thresholds on the inner training set
only. Select the rule using inner validation errors of the same unified fixed
estimator for everyone. Require <=1% accurate-result withholding (error <=2 BPM)
and <=15% total withholding pooled AND for each validation person. Maximize
intercepted >5 BPM errors, tie-break by fewer accurate and total rejections.
No gate is always eligible and wins if no safe positive benefit exists.

Evaluate each selected rule only on its held-out person. Development support
requires the same pooled/per-person safety budgets, >=25% large-error
interception and no worsening of retained MAE. Even a pass will not automatically
enable an application gate. Report quality-passing coverage and all-467 coverage,
record-macro and person-macro error, per-person harm and error interception.
Fixed-rule outer controls use their own train-calibrated fold thresholds; they
are diagnostics only, not a basis for choosing a better rule retrospectively.

## Descriptive error analysis after fixed validation

For each of the 19 errors, match up to three accurate (<=2 BPM) controls, sorted
by sample ID and using the first available tier: same recording; same person,
posture, room and environment; same person, posture and room. Mark fallback
environment mismatches and missing matches. Controls are for descriptive review
only and never calibrate/select a rule. Overlapping windows, shared controls and
same-record observations are correlated; no independent-pair significance claim.

Report four-feature distributions for accurate, intermediate and large errors,
individual matching IDs/differences and matched-tier counts. Plot the first
three matched large-error cases sorted by descending error with their first
matched controls: normalized CSI summary, 1s energy and spectrum. Reference
markers are evaluation-only and are never used by the features. Such plots
cannot prove movement or a particular interference source.

Freeze config, rules and source/code hashes before real extraction. Use new
protected artifacts in `outputs/motion_pollution_v1/`; preserve all earlier
experiments, replay JSON and model files. No thresholds are deployed, no BPM
value is corrected and independently supported breathing evidence is untouched.
