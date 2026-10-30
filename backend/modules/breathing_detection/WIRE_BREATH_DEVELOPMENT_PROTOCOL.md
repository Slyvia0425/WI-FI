# Fixed WiRe-Breath development protocol v1

## Status and artifacts

This protocol is development cross-validation on already-observed data. It
does not create an untouched final test set. New recordings, especially
matching empty-room/interference recordings, are required for a subsequent
independent external binary evaluation.

The authoritative metadata-only split file is
`evaluation/protocols/wire_breath_development_v1.json`. It contains ordered
sample IDs, cache-metadata SHA-256, seed 20261028, window indices, and the
participant/room/environment/posture/quality distribution of every partition.
It contains no CSI arrays, reference waveforms, probabilities, or model files.
Keep this small file with source code. The older ignored
`subject_holdout_folds.json` describes only outer folds and is superseded for
new development by this nested protocol.

## Fixed partitions

Each outer fold assesses one complete participant. From the other four
participants, approximately 20% of windows are assigned to inner validation
by selecting whole recordings. Recordings are stratified by participant and
Quiet/Interference; strata with at least two recordings retain recordings on
both sides. A fixed metadata-only search balances window counts and posture
coverage. Each final inner training and validation partition covers all nine
available postures and both environments. No model prediction, loss, or
respiratory reference was used to select partitions.

| Outer assessment subject | Train windows (records) | Inner validation windows (records) | Assessment windows (records) |
|---|---:|---:|---:|
| 01 | 738 (51) | 189 (14) | 474 (20) |
| 02 | 762 (49) | 186 (14) | 453 (22) |
| 03 | 954 (53) | 219 (14) | 228 (18) |
| 04 | 1,071 (60) | 255 (16) | 75 (9) |
| 05 | 1,014 (55) | 216 (14) | 171 (16) |

Counts include flagged windows so that coverage remains measurable. Keep the
existing 100 ms rule and train only on positive-weight windows. Report recall
among assessable windows alongside assessability coverage and breathing output
fraction over all windows. Quality-passing train/validation/assessment counts
are respectively 725/187/381, 671/183/439, 862/204/227, 973/245/75, and
927/195/171 for the five rows above.

Validation is approximately 17.6--20.4% of the four-person development pool;
whole-record grouping and coverage take precedence over exact window ratios.
Room balance is reported rather than forced: Subject01 appears only in Room1
and Subjects04/05 only in Room2. Person effects and room effects cannot be
fully separated with this dataset.

## How future experiments use the protocol

1. Define the candidate family, metrics, random seeds, threshold policy and
   acceptance limits before running it. Load this same protocol for all arms.
2. Per outer fold, fit using its training partition only. Any learned
   normalization, waveform estimator, calibration, or feature selector uses
   training data only. Select settings and stopping using inner validation.
3. If global settings are needed across folds, use inner-validation results
   only, then freeze settings before scoring any outer assessment partition.
   Start each model from the same approved initial checkpoint; do not carry
   fitted parameters or optimizer state between folds.
4. Report each outer participant, the equal-weight mean across five
   participants, the worst participant, and pooled window counts. Report
   Quiet/Interference and room/posture slices with denominators and coverage.
   Windows from one record are correlated; uncertainty estimates, if used,
   should resample whole recordings rather than individual windows.
5. Do not choose a candidate or adjust thresholds from outer assessment
   failures. The data remain historically observed development data even
   after this procedural separation. Record later exploratory changes as new
   development rounds; do not relabel them as untouched evaluation.

The CSI-Bench official train/validation partitions remain fixed. Training
uses its official training data, and binary acceptance checks use its
validation data with both classes. The already-observed official test is not
reused for development. WiRe-Breath has only breathing-present samples:
it cannot independently train or calibrate a valid empty/breathing decision
boundary, nor can it measure specificity or binary Macro-F1. Any joint
classifier training needs CSI-Bench negatives and explicit class/source
sampling policy. New waveform/rate heads can use the aligned WiRe-Breath
reference within these fixed partitions.

## Integrity and reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.preprocessing.wire_protocol
```

The command checks an existing protocol instead of overwriting a different
one. Dataset changes or algorithm changes require an explicitly reviewed new
version. All future consumers should call:

```python
from backend.modules.breathing_detection.preprocessing.wire_protocol import load_protocol

protocol = load_protocol(protocol_path, cache.manifest['samples'])
fold = protocol['folds']['subject_01']
train_indices = fold['train']
validation_indices = fold['validation']
assessment_indices = fold['assessment']
```

Loading verifies the cache fingerprint/order, complete partition coverage,
unique IDs, and absence of window/recording/outer-person leakage.
