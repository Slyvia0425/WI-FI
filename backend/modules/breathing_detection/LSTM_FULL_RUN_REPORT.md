# Lightweight LSTM full-run report

## Scope

This report records three complete train/validation runs of the promoted
ordered-frequency representation and the subsequent one-way evaluation of the
frozen seed 123 candidate on the official test split. Model and threshold
selection used validation data only.

## Configuration

- Official split: 70,000 train IDs and 15,000 validation IDs
- Representation: ordered subcarrier interpolation to `[300,232]`
- Model: shared `232 -> 32` projection and bidirectional LSTM, hidden size 64
- Optimisation: balanced BCE, batch size 128, learning rate 0.001
- Selection: validation macro-F1, patience 5, maximum 30 epochs
- Seed and device: 42, Apple MPS
- Cache: versioned float16 memory-mapped train and validation arrays

Three training samples were skipped because their source data could not
produce a valid example. All 15,000 validation samples were evaluated.

## Seed 42 result

The selected checkpoint is epoch 27. Its decision threshold, selected only on
validation predictions, is 0.091866.

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 0.9928 | 0.9794 |
| Positive precision | 0.9945 | 0.9839 |
| Positive recall | 0.9966 | 0.9908 |
| Positive F1 | 0.9956 | 0.9873 |
| Macro-F1 | 0.9885 | **0.9663** |
| Empty-room FPR | 0.0226 | **0.0686** |

Validation confusion matrix: TN 2,674, FP 197, FN 112, TP 12,017.

This is a large improvement over both the full periodicity baseline
(validation macro-F1 0.6656, empty-room FPR 0.5517) and the 5,000/1,000-sample
interpolation pilot (0.8187 and 0.4573). The result supports preserving a
consistent ordered frequency axis instead of ranking subcarriers separately
for each sample.

## Seed 42 interpretation

The small train-validation gap justified the two additional full-seed runs
reported below. At this stage the official test split was still untouched.

Generated artifacts are under the ignored directory
`outputs/lstm_interpolated_full_seed42/`: `best_model.pt`, `config.json`, and
`metrics.json`.

## Multi-seed stability

Two additional runs changed only the random seed. All three used the same
official splits, cached inputs, architecture, optimiser, early-stopping rule,
and validation-only threshold selection.

| Seed | Best epoch | Validation Macro-F1 | Empty-room FPR | Threshold |
|---:|---:|---:|---:|---:|
| 42 | 27 | 0.9663 | 0.0686 | 0.0919 |
| 123 | 28 | **0.9680** | **0.0683** | 0.0545 |
| 2026 | 20 | 0.9656 | 0.0714 | 0.1511 |
| Mean | - | **0.9667** | **0.0694** | 0.0992 |

Macro-F1 spans only 0.0024 and empty-room FPR only 0.0031, so the predictive
result is stable across these seeds. The selected probability threshold is
less stable (0.0545 to 0.1511). This does not invalidate the representation,
but it means deployment should not copy a threshold from a different seed.
Use the threshold paired with the selected checkpoint, or add probability
calibration later if probabilities must be comparable across retraining runs.

## Decision after three seeds

The ordered `[300,232]` representation and projected bidirectional LSTM pass
the multi-seed validation gate. Seed 123 is the validation-selected candidate
because it has the highest macro-F1 and lowest empty-room FPR, although its
advantage is small. Freeze that checkpoint together with its threshold
0.054542. The next evaluation step may build the test cache and run a single
test evaluation of this frozen candidate. Test results must not be used to
switch seeds or retune the threshold.

## Frozen test evaluation

The frozen seed 123 checkpoint from epoch 28 was evaluated once with its
validation-selected threshold 0.054542. Its SHA-256 is
`fff3cdb28c7577489400be7cc82506f074703ea22425404be1cc3224923df956`.
No weights or thresholds were changed after observing the test results.

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.9805 | 0.9796 |
| Positive precision | 0.9840 | 0.9839 |
| Positive recall | 0.9920 | 0.9909 |
| Positive F1 | 0.9880 | 0.9874 |
| Macro-F1 | 0.9680 | **0.9670** |
| Empty-room FPR | 0.0683 | **0.0673** |

The test confusion matrix is TN 2,716, FP 196, FN 110, TP 11,977. One of
15,000 samples was not assessable, leaving 14,999 evaluated samples. The close
validation/test results indicate no material aggregate generalisation drop.

| Original K | Evaluated | Empty / sleep | Macro-F1 | Empty-room FPR |
|---:|---:|---:|---:|---:|
| 56 | 1 | 1 / 0 | not defined | 0.0000 |
| 112 | 1,011 | 447 / 564 | 0.9970 | 0.0045 |
| 208 | 10,702 | 1,142 / 9,560 | 0.9259 | **0.1673** |
| 240 | 3,144 | 1,322 / 1,822 | 0.9974 | 0.0023 |
| 416 | 141 | 0 / 141 | not defined | not defined |

The errors are concentrated in K=208, which contributes 191 of 196 false
positives. K=56 has only one sample and K=416 contains no assessable empty
examples, so those slices cannot support balanced-performance claims. Treat
K=208 as the next error-analysis target without retuning against the test set.

The validation-only follow-up is documented in
[`K208_ERROR_ANALYSIS.md`](K208_ERROR_ANALYSIS.md). It identifies a confounded
`HC_near_Fridge` acquisition domain with appliance-like periodic signals, not
evidence that the number 208 itself is causal.
