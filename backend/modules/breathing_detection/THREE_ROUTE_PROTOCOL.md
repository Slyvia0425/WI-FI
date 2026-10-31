# Three-route consistency audit

Write before extracting new cross-route features. Use only the frozen 394-window
`common_mode_v1/window_predictions.json`; no recomputation with changed
decomposition, cleaning, rate estimator, references, quality gates or folds.
This is already-observed development data, not independent final validation.

## CSI-only evidence

The relative estimate proposes an alternative; original remains the default.
Save pairwise rate differences, <=2 BPM agreement of original/common/relative,
and original/common/relative local-peak evidence around both original and
proposed rates. Matching means within the existing 2 BPM accuracy tolerance;
when multiple local peaks match, select maximum height, then closer frequency,
then lower BPM. Save matched height, prominence and original peak rank.
Missing components or matched peaks remain explicit, not invented evidence.
Do not use reference-near peak diagnostics, identity or reference to construct
these features. Relative/common spectra are correlated transforms of one input,
not independent sensor votes or physiological confidence probabilities.

## Descriptive comparison

Include all 394 windows, especially six original >5 BPM errors corrected to
<=2 by relative-only and five originally <=2 BPM outputs made >2 by relative-only.
These groups are defined by reference for analysis only, not input to prediction.
Report their feature ranges/overlap, exact matched peaks and rate disagreements,
plus all relative changes >2 BPM (not only the eleven highlighted cases).
No thresholds, classifier or reference-driven component mix are fit in this stage.

After audit, either stop if there is no defensible CSI-only distinction, or
freeze exactly one conservative rule and its rationale in a separate protocol
before computing its predictions. No parameter sweep. Use original as fallback,
whole-record leave-person-out inner-validation selection, full coverage and
zero original-good harm. Report fixed and selected results; support requires
zero accurate harm, >=25% fewer >5 BPM errors and no person's MAE worsening.
No support can be inferred merely from baseline fallback. Missing evidence must
preserve original, never reject to hide errors. No automatic deployment.

Use protected new `outputs/three_route_audit_v1/`; record source/protocol/code
and protected artifact hashes before extraction. Preserve all previous results,
data, checkpoint, replay and frontend lock changes. No production imports,
model retraining, frontend changes or GitHub submission.
