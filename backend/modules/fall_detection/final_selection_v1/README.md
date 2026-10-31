# Final research-model selection (four authorized stages)

Protocol: `protocol.json`. Candidate pipelines are the original 1D-CNN and the
pinned official ResNet18. Every experiment preserves the official Kaggle v12
train/validation/test membership. All prior models remain intact.

Live Colab notebook:
https://colab.research.google.com/drive/1ZslLfeUwnqRkZCF5JxlkI_AQnFNHn70_

The user explicitly approved mounting Drive for persistent experiment storage.
Only the newly created folder
`MyDrive/CSI-Bench_FallDetection_FinalSelection_v1` is used by authored writes.
Raw public data stays on `/content/CSI-Bench-v12` to avoid Drive I/O during training.

## 1. Recover usable ResNet18 weights

The earlier archive download was not verified complete and its runtime expired.
The new seed-42 run is `official_resnet18_seed42_recovery_v1`. This is a fresh
same-configuration run, not a claim that the original exact weights were recovered.
Pinned official model, Dataset, trainer math and the known scheduler mismatch are
retained. New `--seed` support and atomic durable weight backups affect observation
and storage only. Reports, official checkpoints, logs and a portable weight bundle
are saved in Drive. A second Python process replays all validation records after
training and checks scores against those saved during training.

`verify_and_export_recovery.py` prints a bounded compressed validation CSV and
JSON report through the notebook output and offers a small reports download link.
No private authentication data is printed. The large weights remain persistently
stored even if browser download delivery fails. Any locally retrieved weight file
must be checked against the cloud SHA256 before use.

## 2. Compare validation operating points

`compare_thresholds.py` accepts only predictions whose IDs, labels and devices
match all 1,005 official validation records. It writes Precision–Recall curves,
average precision, all threshold candidates, and metrics at 0.5 and the selected
single global threshold. Exact F1 fraction comparison breaks ties by higher
Recall, then higher threshold. Per-device thresholds are not fitted.

Run from the repository root, using actual retrieved prediction files:

```bash
.venv/bin/python backend/modules/fall_detection/final_selection_v1/compare_thresholds.py \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --candidate original_1d_cnn_seed42=backend/modules/fall_detection/reports/final_selection_v1_original_seed42_cpu/validation_predictions.csv \
  --candidate official_resnet18_seed42=PATH_TO_RECOVERED_VALIDATION_CSV \
  --output-dir backend/modules/fall_detection/reports/final_selection_v1_seed42_comparison
```

The baseline-reference calculation already demonstrates that maximizing overall
Fall F1 does not guarantee low ESP32 false positives. At threshold ~0.38939 the
original CNN has overall F1 0.94130 but ESP32 FP=26/30; at 0.5 it has overall
F1 0.94025 and ESP32 FP=16/30. These are validation results only.

## 3. Check seeds 42, 43 and 44

The original CNN reuses the existing seed-42 checkpoint; new seeds use unchanged
`train.py` with the original Adam/weighted-BCE, batch 32, max 100, patience 10
configuration on MPS. ResNet18 uses official AdamW/CE, batch 128, max 100,
patience 15 on Colab T4 for all three seeds. No shared redefinition of learning
rate, loss or preprocessing is introduced across the two different pipelines.

`select_across_seeds.py` refuses to select until all six candidate/seed prediction
files and checkpoint hashes exist. It selects the pipeline by mean validation
Fall F1 at each run's validation-selected global threshold, then mean Recall,
then the smaller model. It reports per-device means and sample standard deviations.
The final artifact seed is fixed to 42 rather than cherry-picking a seed.

## 4. Freeze, independent inference, then official test evaluation

`freeze_final.py` checks selection and protocol hashes and freezes the seed-42
weights, threshold, normalization and inference source hashes in a NEW artifact.
`infer.py` loads that artifact in a new Python process and reads a single raw
HDF5 segment. `evaluate_final.py` only accepts an integrity-checked frozen artifact;
it uses one score per physical segment and reports test_id plus its disjoint
easy/medium/hard partition, including supported per-device groups.

No test inference is authorized to run before the preceding checks and freeze.
Test results from the original model were inspected earlier in the project;
this work cannot claim that the test set has never been examined. Known identical
content pairs across the official splits are preserved and disclosed. Three seeds
measure training variability, not uncertainty from collecting new independent data.

The deliverable is a research final baseline, with its ESP32 limitations stated.
Segment-level metrics do not establish continuous real-world alarm readiness.

## Executed implementation checks

- Single-file official preprocessing exactly matched pinned BenchmarkCSIDataset
  for one actual HP record and one actual ESP32 record, shape `[1,500,232]`.
- Threshold selection matched the established implementation on all 1,005
  actual original-CNN validation predictions.
- The new raw-file original-CNN inference path exactly matched established
  FallDataset inference on HP/ESP32 × Fall/Nonfall (maximum score difference 0).
- These checks did not train models or evaluate test records.

Current progress and actual results must be read from reports and the live
notebook; prepared scripts and queued cells are not evidence of completed runs.
