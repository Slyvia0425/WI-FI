# Raw CSI spike audit and isolated-spike preprocessing comparison

Freeze this protocol before batch audit and candidate-rate calculation. Keep
`mean_spectrum_boundary0_v1`, LSTM, quality rules, reference waveforms, application
gates and frontend unchanged. The experiment is on previously observed development
data, not independent final evaluation. No physical-motion or AGC cause is assumed.

## Source audit (all 467 files, including 73 quality exclusions)

Inspect MAT fields, complex/amplitude consistency (`abs(csi_complex)` versus
`csi_abs`), finite values, time ordering, repeated exact frames and extreme-frame
signatures across files. Record raw amplitude/frame-median quantiles and original
versus resampled maxima. For descriptive common-excursion flags only, calculate
per-subcarrier global median/MAD standardized deviation (>6) and flag a frame
when at least half its carriers exceed it. Compare flagged-frame timestamp
intervals with ordinary intervals; do not infer causality from association.

Verify timestamp interpolation independently matches the current loader, its
target points are inside original coverage, and ground-truth alignment does not
introduce an unexplained start offset. Inventory fields relevant to source
interpretation; the existence of complex CSI does not imply usable independent
antennas or a known phase calibration. A repeated extreme frame can suggest
representation/capture effects but cannot prove saturation, gain changes or
fabricated data. Hash raw MAT files so later checks can verify no data mutation.

## Fixed preprocessing (no reference input)

Use a time-aware local median/MAD filter, half-window 0.10s, at least five
observations, scale 1.4826 * max(local MAD, 0.05 * per-carrier global MAD, 1e-8).
Flag deviations greater than 6 or 8 times this scale. These two conservative
thresholds are fixed in advance, not selected from the 19 known errors.

Only replace flagged runs of at most two points per subcarrier, bracketed by
unflagged points in the same sample. Require total bracket span <=0.15s and
each original timestamp interval <=0.10s. Longer runs and unbracketed boundaries
remain untouched. Replacement is linear interpolation between the two brackets;
no entire-window smoothing, reference cleaning, rate doubling or bounds change.
Compute all flags from the original input, then replace; no iterative cleaning.

Four candidates: raw-before-resampling thresholds {6,8} and uniform-30Hz-after-
resampling thresholds {6,8}. Raw processing uses only samples bracketing the
current 30s target grid; no extra future tail/independent file is used. Run the
unchanged fixed rate estimator after each candidate. Keep the original quality
exclusions; interpolation or cleaning never upgrades a bad timestamp window.

## Evaluation

Evaluate the same 394 quality-passing windows. Compare each fixed candidate with
the unchanged baseline and report error, >5 BPM errors, <=2 BPM accurate-result
harm, per-person MAE, coverage and modified cells/frames. Feature extraction and
filtering never use a reference or sample identity; references score outcomes only.

Primary grouped selection reuses the fixed leave-person-out / whole-recording
inner validation protocol. Choose minimum inner-validation recording-macro MAE
among baseline and candidates with **zero harm** to currently accurate outputs
and full output coverage. Harm means rejection OR changing a <=2 BPM estimate
to >2 BPM. Ties prefer fewer changed rate outputs, then baseline/config order.
No outer person/error case changes a rule. The unified estimator is constant;
fold-selected preprocessing is experimental, not a runtime per-person method.

Development support requires zero accurate-result harm, full output coverage,
at least 25% reduction of large errors and no person's MAE worsening. A pass is
not automatic adoption: the final preprocessing choice still needs a fixed
configuration and independent assessment. Fixed-candidate pooled comparisons
are descriptive and cannot retroactively replace nested results.

Plots after assessment compare raw, current resampled and cleaned signals for
the first three baseline large errors sorted by severity, alongside spectral
estimates/reference markers. Reference markers are evaluation-only. This is
not physiological validation or physical-source identification. No plot review
or observed error is used to expand the parameter grid.

Use protected `outputs/spike_preprocessing_v1/` artifacts, saving configuration,
source/code hashes and protected model/unified-result/replay/lock hashes before
real extraction. Preserve all previous experiments and raw files. A preprocessing
failure must be recorded as unavailable, not silently assigned the baseline rate.
