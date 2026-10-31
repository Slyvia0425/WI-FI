# Confidence threshold on the test rows (budget 20% of the validation windows answered "unknown"; frozen before the test rows were read)

| Split | T | Threshold | Rejected (false rejection of known classes) | Accuracy: all to kept | Macro-F1: all to kept | Errors rejected | Rejected windows that were wrong |
|---|---|---|---|---|---|---|---|
| split_0 | 2.3042 | 0.8062 | 24.2% | 0.836 to 0.919 | 0.450 to 0.478 | 63% | 43% |
| split_1 | 6.3231 | 0.6887 | 21.9% | 0.882 to 0.977 | 0.476 to 0.494 | 85% | 46% |
| split_2 | 0.6876 | 0.5355 | 28.7% | 0.579 to 0.712 | 0.364 to 0.415 | 51% | 75% |
| **mean ± std** | | | 24.9% ± 3.5% | 0.766 to 0.870 | 0.430 to 0.462 | 66% | |

Share of each true class answered "unknown":

| Split | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| split_0 | 14% | 23% | 60% | 61% |
| split_1 | 16% | 16% | 81% | 93% |
| split_2 | 20% | 23% | 49% | 43% |

Recall of the kept windows (all windows in brackets):

| Split | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| split_0 | 1.00 (0.99) | 0.98 (0.95) | 0.00 (0.00) | 0.00 (0.00) |
| split_1 | 0.99 (0.96) | 0.99 (0.97) | 0.00 (0.00) | 0.00 (0.01) |
| split_2 | 0.90 (0.84) | 0.86 (0.73) | 0.03 (0.03) | 0.00 (0.00) |

Calibration on the test rows: ECE 0.119 before, 0.084 after (mean of the splits). Kept windows that are still wrong: split_0 618, split_1 158, split_2 2435.
