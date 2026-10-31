# Lightweight LSTM pilot report

## Purpose

Before an expensive full run, the first device-agnostic representation reduced
each `[1,K,300]` sample to ten cross-subcarrier statistics at every time point,
giving an LSTM input of `[300,10]`. This supports all observed values of K
without padding, clipping or interpolation.

## Pilot result

Configuration: first 5,000 official training IDs, first 1,000 validation IDs,
seed 42, bidirectional LSTM, hidden size 64, balanced BCE loss, batch size 64,
validation macro-F1 early stopping.

The best checkpoint occurred at epoch 1:

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 0.7658 | 0.8150 |
| Positive F1 | 0.8603 | 0.8930 |
| Macro-F1 | 0.5676 | 0.6052 |
| Empty-room FPR | 0.7723 | 0.7378 |

This is worse than the full periodicity baseline validation result
(macro-F1 0.6656, empty-room FPR 0.5517). Falling validation macro-F1 across
later epochs also gives no evidence that scaling this representation to the
full dataset is justified.

## Decision

Do not run this `[300,10]` representation on the full dataset. Aggregating all
subcarriers into summary statistics discards individual periodic waveforms and
phase relationships that may distinguish breathing from background variation.

## Top-32 follow-up

The proposed second representation was implemented and evaluated on exactly
the same first 5,000 train and 1,000 validation IDs with the same seed, hidden
size, batch size, loss and early-stopping rule. It selected the 32
highest-variance subcarriers per sample, sorted them by variance, standardized
each stream over time, and supplied `[300,32]` to the LSTM.

| Representation | Validation Macro-F1 | Empty-room FPR |
|---|---:|---:|
| Cross-subcarrier statistics `[300,10]` | 0.6052 | 0.7378 |
| Top-variance streams `[300,32]` | 0.5977 | 0.6280 |
| Full periodicity baseline | 0.6656 | 0.5517 |

Top-32 reduced empty-room false positives but failed the predeclared promotion
criterion and did not improve macro-F1. Selecting and reordering different
physical subcarriers independently for every sample likely destroys feature
position consistency across samples. It must not be scaled to the full split.

## Fixed-frequency-axis follow-up

A third representation preserved subcarrier order, standardized each original
subcarrier over time, and linearly interpolated the ordered frequency axis to
232 positions. A shared per-time-step projection reduced 232 values to 32
before the same bidirectional LSTM. It used the identical 5,000/1,000 IDs,
seed, hidden size, batch size and validation selection rule.

| Representation | Validation Macro-F1 | Empty-room FPR |
|---|---:|---:|
| Cross-subcarrier statistics `[300,10]` | 0.6052 | 0.7378 |
| Top-variance streams `[300,32]` | 0.5977 | 0.6280 |
| Periodicity baseline (full validation) | 0.6656 | 0.5517 |
| Ordered interpolation `[300,232]` + projection | **0.8187** | **0.4573** |

The interpolated model's best checkpoint was epoch 6. Validation accuracy was
0.9180, positive F1 was 0.9529, and its confusion matrix was TN 89, FP 75,
FN 7, TP 829. The result meets the predeclared promotion criteria and strongly
supports consistent frequency-axis positions over per-sample ranking.

## Promotion decision

Promote ordered interpolation to the project LSTM candidate. Before full
training, cache the `[300,232]` representation so each epoch does not reread
and transform all HDF5 files. Train the complete train/validation splits while
retaining validation-only early stopping and threshold selection. Multiple
seeds and K/device slice metrics are required before a one-way evaluation of
the untouched test split.

## Cache implementation

The promoted representation now has a versioned float16 memory-mapped cache.
It stores sequences, labels, assessability weights and a JSON manifest, records
progress every 250 samples, and resumes compatible partial builds. Training
converts one cached batch at a time back to float32. A real 500-train/200-val
cache and one-epoch training smoke test completed successfully; its 93 MB size
matches the projected approximately 13 GB for all 100,000 samples.

The full train and validation caches have since been completed and the first
complete-split seed has been trained. Results and next steps are recorded in
[`LSTM_FULL_RUN_REPORT.md`](LSTM_FULL_RUN_REPORT.md).
