# Error analysis of Logistic Regression on MiniRocket (test rows)

## Confusion, mean of the three splits (rows = true class, share of its windows)

| True \ called | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| Human | 0.93 | 0.05 | 0.01 | 0.01 |
| Pet | 0.08 | 0.89 | 0.03 | 0.01 |
| IRobot | 0.63 | 0.16 | 0.01 | 0.20 |
| Fan | 0.30 | 0.54 | 0.16 | 0.00 |

Counts per split (rows = true class):

- split_0: Human [4304, 60, 0, 0]; Pet [209, 4141, 16, 0]; IRobot [858, 154, 0, 0]; Fan [256, 99, 6, 0]
- split_1: Human [4076, 160, 2, 3]; Pet [88, 3764, 13, 1]; IRobot [170, 7, 0, 184]; Fan [22, 198, 198, 5]
- split_2: Human [3755, 418, 136, 155]; Pet [657, 3008, 282, 165]; IRobot [1642, 908, 102, 281]; Fan [49, 307, 0, 0]

## Key confusions (mean share of the true class)

| Confusion | Share |
|---|---|
| Human called Fan | 0.012 |
| Fan called Human | 0.300 |
| Pet called Human | 0.077 |
| Human called Pet | 0.048 |
| Human called IRobot | 0.010 |
| IRobot called Human | 0.626 |
| Pet called Fan | 0.013 |
| Fan called Pet | 0.535 |
| IRobot called Fan | 0.202 |
| Fan called IRobot | 0.162 |
| IRobot called Pet | 0.160 |
| Pet called IRobot | 0.025 |

## Is it a person? (Human against everything else)

| Split | Precision | Recall | F1 | Pet called Human | IRobot called Human | Fan called Human | Humans called Pet / IRobot / Fan |
|---|---|---|---|---|---|---|---|
| split_0 | 0.765 | 0.986 | 0.862 | 4.8% (209) | 84.8% (858) | 70.9% (256) | 60 / 0 / 0 |
| split_1 | 0.936 | 0.961 | 0.948 | 2.3% (88) | 47.1% (170) | 5.2% (22) | 160 / 2 / 3 |
| split_2 | 0.615 | 0.841 | 0.711 | 16.0% (657) | 56.0% (1642) | 13.8% (49) | 418 / 136 / 155 |

## Confident mistakes (highest probability at least 0.9)

| Split | Windows that confident | Wrong among them | Share of all errors that are confident | Correct windows below 0.9 | Confident errors by true class |
|---|---|---|---|---|---|
| split_0 | 88.9% | 12.0% | 65.1% | 6.4% | {'Human': 0, 'Pet': 122, 'IRobot': 750, 'Fan': 208} |
| split_1 | 93.6% | 8.1% | 64.3% | 2.5% | {'Human': 102, 'Pet': 55, 'IRobot': 254, 'Fan': 262} |
| split_2 | 0.0% | 0.0% | 0.0% | 100.0% | {'Human': 0, 'Pet': 0, 'IRobot': 0, 'Fan': 0} |

## Rooms with the most errors (per split, top 4)

| Split | Room | In training | Windows | Error rate | Errors | Classes |
|---|---|---|---|---|---|---|
| split_0 | E11 | True | 2121 | 65.6% | 1391 | {'Human': 212, 'Pet': 536, 'IRobot': 1012, 'Fan': 361} |
| split_0 | E16 | True | 211 | 77.3% | 163 | {'Pet': 211} |
| split_0 | E18 | True | 1133 | 7.9% | 90 | {'Human': 499, 'Pet': 634} |
| split_0 | E15 | True | 144 | 4.2% | 6 | {'Human': 144} |
| split_1 | E12 | True | 423 | 98.8% | 418 | {'Fan': 423} |
| split_1 | E11 | True | 1352 | 28.9% | 391 | {'Human': 354, 'Pet': 637, 'IRobot': 361} |
| split_1 | E18 | True | 1140 | 12.4% | 141 | {'Human': 574, 'Pet': 566} |
| split_1 | E16 | True | 495 | 10.3% | 51 | {'Human': 495} |
| split_2 | E19 | False | 2933 | 96.5% | 2831 | {'IRobot': 2933} |
| split_2 | E11 | True | 780 | 80.9% | 631 | {'Human': 144, 'Pet': 636} |
| split_2 | E18 | True | 849 | 51.5% | 437 | {'Human': 283, 'Pet': 566} |
| split_2 | E13 | True | 3043 | 14.0% | 426 | {'Human': 2687, 'Fan': 356} |

Rooms of the dataset and the classes recorded in them: E11 ['Fan', 'Human', 'IRobot', 'Pet']; E12 ['Fan', 'Pet']; E13 ['Fan', 'Human', 'Pet']; E14 ['Human']; E15 ['Human', 'Pet']; E16 ['Human', 'Pet']; E17 ['Human', 'Pet']; E18 ['Human', 'Pet']; E19 ['Human', 'IRobot']; E20 ['Pet']

| Split | Windows | Rooms with one class: windows / error rate | Rooms shared by several classes: windows / error rate |
|---|---|---|---|
| split_0 | all | 3055 / 0.0 | 7048 / 0.235 |
| split_0 | Human and Pet only | 3055 / 0.0 | 5675 / 0.05 |
| split_1 | all | 2024 / 0.001 | 6867 / 0.152 |
| split_1 | Human and Pet only | 2024 / 0.001 | 6083 / 0.044 |
| split_2 | all | 2772 / 0.023 | 9093 / 0.543 |
| split_2 | Human and Pet only | 2772 / 0.023 | 5804 / 0.302 |

## Recording groups: the rare classes, and the three worst groups of the other classes

| Split | Class | Device/user | Recording | Room | Windows | Accuracy | Mostly called | Share |
|---|---|---|---|---|---|---|---|---|
| split_0 | Fan | F01 | Test_1 | E11 | 361 | 0.00 | Human | 0.71 |
| split_0 | IRobot | IR01 | Test_1 | E11 | 1012 | 0.00 | Human | 0.85 |
| split_0 | Pet | P12 | Laser_Pointer_` | E16 | 211 | 0.23 | Human | 0.70 |
| split_0 | Human | U34 | Combined | E18 | 213 | 0.84 | Human | 0.84 |
| split_0 | Pet | P17 | Test_6 | E18 | 211 | 0.89 | Pet | 0.89 |
| split_1 | Fan | F02 | Fan | E12 | 423 | 0.01 | Pet | 0.47 |
| split_1 | IRobot | IR01 | Test_2 | E11 | 361 | 0.00 | Fan | 0.51 |
| split_1 | Human | U34 | Sneaking | E18 | 145 | 0.77 | Human | 0.77 |
| split_1 | Human | U32 | Walking | E16 | 142 | 0.80 | Human | 0.80 |
| split_1 | Human | U30 | Walking | E18 | 143 | 0.85 | Human | 0.85 |
| split_2 | Fan | F03 | Position_1 | E13 | 356 | 0.00 | Pet | 0.86 |
| split_2 | IRobot | IR02 | Test1 | E19 | 2933 | 0.04 | Human | 0.56 |
| split_2 | Human | U25 | sneak | E11 | 144 | 0.10 | IRobot | 0.71 |
| split_2 | Pet | P12 | Laser_Pointer_2 | E16 | 210 | 0.18 | Human | 0.74 |
| split_2 | Pet | P06 | Test_6 | E11 | 212 | 0.19 | IRobot | 0.37 |

## Same device in training or not (VALIDATION rows; rare-class groups; three candidates)

| Split | Class | Device | Recording | Room | Windows | Same device in training (windows) | Logistic Regression on MiniRocket: recall / F1 | Random Forest on MiniRocket: recall / F1 | Random Forest on statistics: recall / F1 |
|---|---|---|---|---|---|---|---|---|---|
| split_0 | Fan | F02 | Fan | E12 | 423 | False (0) | 0.00 / 0.00 | 0.00 / 0.00 | 0.05 / 0.07 |
| split_0 | IRobot | IR01 | Test_2 | E11 | 361 | False (0) | 0.00 / 0.00 | 0.00 / 0.00 | 0.14 / 0.15 |
| split_1 | Fan | F03 | Position_1 | E13 | 356 | True (355) | 0.99 / 0.73 | 0.78 / 0.87 | 0.67 / 0.38 |
| split_1 | IRobot | IR01 | Test_1 | E11 | 1012 | False (0) | 0.00 / 0.00 | 0.00 / 0.00 | 0.13 / 0.18 |
| split_2 | Fan | F03 | Position_2 | E13 | 355 | False (0) | 0.00 / 0.00 | 0.00 / 0.00 | 0.17 / 0.17 |
| split_2 | IRobot | IR01 | Test_2 | E11 | 361 | True (1012) | 0.77 / 0.58 | 0.07 / 0.10 | 0.39 / 0.31 |

F1 is the F1 of that class over the whole validation set of the split; recall is for this group only.
