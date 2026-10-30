# Frozen breathing-presence evaluation under fan interference

Freeze before inference. Evaluate only existing CSI-Bench `test_id` samples
whose exact deployment is LOS_FanOff, LOS_FanOn, NLOS_FanOff or NLOS_FanOn.
Metadata contains 4,156 such official-test samples; use every one, no probability-
based filtering. Training/validation metadata are inspected for context only;
no training/validation samples enter the scored denominator. This official test
has already been observed: a development subgroup audit, not new independent
external validation or a causal fan experiment.

## Locked inference

Use `lstm_interpolated_full_seed123/best_model.pt`, SHA256
`fff3cdb28c7577489400be7cc82506f074703ea22425404be1cc3224923df956`,
checkpoint threshold 0.054542455822229385, epoch28, ordered-interpolation
representation. Use current `BreathingDataset(normalize=True)` and
`extract_interpolated_subcarriers`, matching frozen offline evaluation; preserve
carrier order, standardize each source carrier and interpolate to [300,232].
No retraining, threshold sweep, input-feature revision, test selection, phase
feature, periodic gate or frontend mutation. Run eval/no_grad deterministic CPU
in batches; report that historic aggregate scores used MPS, not claim bitwise
identity across accelerators. Prediction is positive when p>=threshold.

Use existing loader assessability rules. Record every exclusion/failure and
its class/scenario; do not classify unreadable/invalid samples as empty or quietly
drop from total coverage. A preprocessing failure is explicit unavailable.
All scored probabilities must be finite. File/metadata labels `sleep` and `empty`
are the dataset's breathing-positive and empty-negative proxies, not direct
respiratory sensor measurements. `none` is absence of detected breathing evidence,
not a guarantee of no person. No BPM/physiological rate or apnea accuracy claim.

## Primary report

For each LOS/NLOS × fan on/off scenario: metadata/available counts, TP/FN/TN/FP,
breathing-positive recall (TP/(TP+FN)), empty false-positive rate (FP/(FP+TN)),
specificity, and macro-F1 only when both classes are present. If there are no
empty examples, false-positive rate/specificity/macro-F1 are **None**, never zero.
LOS_FanOn has no local test empty examples; explicitly mark that cell missing.
Report class-specific probability summaries for diagnostics, not calibrated
confidence. Save all sample predictions, reasons and misclassified sample IDs.

Compare fan-on minus fan-off in percentage points separately for LOS and NLOS,
with positive and negative denominators separately; unavailable comparisons
remain None. Do not use pooled overall accuracy as proof of fan robustness.
Class mix differs between scenarios, so compare recall and FPR, not only F1.

## Comparability/limitations

All selected rows have source user field U23 and environment E09, with four
device categories. `empty` user metadata does not imply an occupied room.
Report by scenario × device and equal-device averages of class-conditional
metrics, plus on/off comparisons per device where both conditions have the
relevant class. Include carrier-count/difficulty distributions so device/domain
imbalances are visible. They are not paired randomized captures or independent
people. Source-session IDs/window overlap are not independently verified.
No binomial confidence intervals or significance tests pretending windows are
independent. No retrospective promotion threshold or arbitrary pass/fail score.

## Provenance and outputs

Protected new `outputs/fan_detection_v1/`, refuse overwrite. Save config,
protocol/evaluator/loader/feature hashes, metadata and split hashes, checkpoint
and protected original results/replay/lock hashes before inference. Hash every
selected raw HDF5 before/after; verify ID uniqueness and test-only membership.
Save predictions, metrics, failures and error cases. Do not overwrite old
test_metrics, data cache, model files or frontend artifacts. Local work only,
no GitHub commit/push. Add tests for missing classes, threshold equality, quality
coverage, finite predictions and on/off/device comparison denominators.
