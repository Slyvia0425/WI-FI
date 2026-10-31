# Per-class confidence thresholds on the test rows (precision target 90%; frozen before the test rows were read; second report)

Thresholds (calibrated probability; "never" = the class is never claimed, its windows are answered unknown):

| Split | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| split_0 | 0.444 | 0.378 | never | never |
| split_1 | 0.889 | 0.340 | never | 0.540 |
| split_2 | 0.571 | 0.719 | never | never |

| Split | Unknown share | Accuracy: all to kept | Macro-F1: all to kept | Errors turned into unknown | Unknown windows that were wrong | Wrong claims kept |
|---|---|---|---|---|---|---|
| split_0 | 0.4% | 0.836 to 0.839 | 0.450 to 0.451 | 2% | 98% | 1619 |
| split_1 | 31.5% | 0.882 to 0.921 | 0.476 to 0.477 | 54% | 20% | 480 |
| split_2 | 45.9% | 0.579 to 0.812 | 0.364 to 0.439 | 76% | 70% | 1208 |
| **mean ± std** | 25.9% ± 23.2% | 0.766 to 0.857 | 0.430 to 0.456 | 44% | | |

First report (one threshold for all classes, 20% budget): unknown 24.9%, accuracy of kept 0.870, macro-F1 of kept 0.462.

Share of each true class answered unknown:

| Split | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| split_0 | 0% | 0% | 0% | 5% |
| split_1 | 53% | 2% | 68% | 53% |
| split_2 | 30% | 42% | 72% | 67% |

Windows claimed as each class, and the share of those claims that are right:

| Split | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| split_0 | 5614 (77%) | 4449 (93%) | 0 (-) | 0 (-) |
| split_1 | 1851 (100%) | 4125 (91%) | 0 (-) | 111 (0%) |
| split_2 | 4011 (77%) | 2414 (88%) | 0 (-) | 0 (-) |

When the answer is unknown, the lean (best guess) equals the true class in: split_0 2%, split_1 80%, split_2 30% of the unknown windows.
