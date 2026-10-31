# CSI-Bench official-source reproduction, FallDetection only

Pinned upstream source commit: b1beac7b0fa46e01748901195ecfbe309dfc0180.
Upstream source is copied unchanged under upstream/ with its LICENSE and
per-file SHA256 provenance. It is not the previously implemented custom CNN.

Completed scope: official MLP, seed42, official v12 train/val, no test.
This is a source-faithful component run with a validation-only observation
wrapper, not complete reproduction of seven models/three seeds/paper test
numbers. Existing model checkpoints and preprocessing remain untouched.

## Commands (from repository root)

```bash
.venv/bin/python -m pip install -r backend/modules/fall_detection/official_reproduction/requirements.lock.txt
.venv/bin/python backend/modules/fall_detection/official_reproduction/run_baseline.py --model mlp --run-name official_mlp_seed42_sourcefaithful_repeat
.venv/bin/python backend/modules/fall_detection/official_reproduction/check_saved_model.py --run-name official_mlp_seed42_sourcefaithful_repeat
```

Data root defaults to /Users/chenxiaohe/Documents/CSI-Bench-v12. Change
--data-root on the training wrapper if relocating data; the saved-model check
currently uses this same absolute root. Choose fresh run names: earlier results
are never overwritten. Repeat runs are user-triggered, not automatically run.

## Preserved upstream behavior and discovered limitations

- Dataset: BenchmarkCSIDataset, train_id and val_id only, HDF5 CSI_amps.
  All 5695 records passed official-loader finite/shape preflight. Metadata and
  split IDs match the previously audited manifest. Official labels Fall0,
  Nonfall1; score is softmax[:,0]. No random re-splitting or deduplication.
- Each record is normalized by its own full original amplitude mean/std
  (torch.std default correction1), before padding to [1,500,232]. No train-fitted
  custom per-feature normalization is used. Real HP/ESP32 outputs independently
  matched this formula exactly. Added padding is zero.
- MLPClassifier flattens this input but uses only its first10000 values: 43
  full time rows plus24 features from the next time row. This is preserved
  upstream behavior; the classifier does not use the full500-point segment.
- TaskTrainer.train(), train_epoch() and evaluate() run upstream math unchanged.
  AdamW .001, weight decay1e-5, batch128, CrossEntropyLoss without class weights,
  at most100epochs and early stopping15 by global validation CE. Argmax two
  logits is equivalent to Fall softmax threshold0.5 (Fall is class0).
- Scheduler lambda uses warmup_steps=37batches*5=185, but trainer calls
  scheduler.step() once each epoch. This does NOT realize a5-epoch warmup.
  First epoch LR0; epoch2 5.4054054e-6; best epoch32 LR0.000167567568.
  We did not silently fix this. A corrected scheduler must be a separate run.
- Platform deviations: MPS instead of the paper's A10G, current pinned library
  versions instead of unknown original environment, num_workers0/pin_memoryFalse
  for local execution, fail-fast collator instead of silently filtering None
  (no bad data found). Only extra evaluation observation/coverage checks added.
- Official MLP early-stopped47, best32; best valCE0.215373261, overall Fall
  F10.886688669, weightedF10.897817194, accuracy0.897512438.
  HP: TP373 FP55 FN17 TN481, FallF10.911980440, weightedF10.922623493.
  ESP32: TP30 FP12 FN19 TN18, FallF10.659340659, weightedF10.613001206.
  Duration1011.46s. See reports/official_mlp_seed42_sourcefaithful_v1.
- New-process CPU replay safely/strictly loaded the saved model and checked all
  validation IDs, scores, classification decisions and loss against saved MPS
  best-epoch results. checkpoint_check.json records tolerances and differences.
- Official hard/easy test results were not computed; external paper test results
  must not be compared directly with these validation-only numbers. Known
  train/val duplicate content and hardware/environment confounding remain.

## ResNet18 status

Initially investigated official ResNet18. All train/val loader checks and a
2-record forward pass succeeded. The batch128 forward/backward probe session
was interrupted, temporary log disappeared, and completion/failure cause could
not be established. No ResNet formal training or metrics are claimed. On this
8GB host we did not retry the large probe. Record: reports/
official_resnet18_preflight_v1/batch128_probe.json. MLP permits the configured
batch128 without substituting a different architecture under the same name.

## Outputs

Original official checkpoint: outputs/<run>/best_model.pt (zero-based epoch).
Additional portable bundle: outputs/<run>/reproduction_bundle.pt (one-based
best_epoch, config, model weights, source commit, label mapping, split IDs and
library versions). Neither is compatible with the existing custom-CNN inference
script unchanged. Reports retain full validation history, learning rates,
best scores by ID, training curves, hashes and independent reload checks.
