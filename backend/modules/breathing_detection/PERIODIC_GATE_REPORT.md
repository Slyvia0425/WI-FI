# Periodic-interference gate experiment

## Scope

This experiment used only the official training and validation splits. The
official test split was not loaded. The frozen seed 123 LSTM and breathing
threshold 0.054542 were unchanged.

The candidate gate used the existing 18 interpretable signal features. It was
trained only on training windows that the frozen LSTM classified as breathing,
with the risk target defined as an empty-room false positive. No K, device,
user, environment, deployment, or other metadata was supplied. The rejection
threshold was selected on validation data under a maximum 1% rejection budget
for correctly detected breathing windows.

## Result

The selected risk threshold was 0.877799.

| Quantity | Result |
|---|---:|
| Validation windows | 15,000 |
| Rejected windows | 122 (0.81%) |
| Existing false positives rejected | 6 / 196 (3.06%) |
| Correct breathing windows rejected | 116 / 12,032 (0.96%) |
| Useful-error share among rejections | 4.92% |
| Retained Macro-F1 | 0.9686 (before: 0.9680) |
| Retained empty-room FPR | 6.63% (before: 6.83%) |

The small selective-metric improvement is not operationally worthwhile: the
gate discards about nineteen correct breathing windows for every false positive
it removes. Because its threshold and reported result use the same validation
split, even this small gain is an optimistic development estimate.

## Decision

Do not enable this signal-only periodic gate in the default inference path. Its
model and validation output remain under the ignored
`outputs/periodic_interference_gate/` directory for reproducibility, and the
training/inference code remains useful for future experiments.

The result supports the earlier error analysis: single-window periodicity is
shared by real breathing and appliance-like interference. A useful gate will
need at least one additional independent source of evidence, such as a Fan or
appliance classifier, synchronized activity evidence, longer temporal context,
or new labelled interference recordings.
