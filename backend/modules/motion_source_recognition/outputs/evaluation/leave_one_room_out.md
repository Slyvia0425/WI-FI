# Leave one room out (Logistic Regression on MiniRocket, C = 0.1)

| Held-out room | Windows (classes) | Accuracy | Most frequent class | Recall of the classes in the room |
|---|---|---|---|---|
| E11 | 6342 (Human 1344, Pet 3264, IRobot 1373, Fan 361) | 0.363 | 0.212 (Human) | Human 0.69, Pet 0.42, IRobot 0.01, Fan 0.00 |
| E12 | 1792 (Pet 1369, Fan 423) | 0.244 | 0.000 (Human) | Pet 0.31, Fan 0.03 |
| E13 | 18531 (Human 16195, Pet 1481, Fan 855) | 0.107 | 0.080 (Pet) | Human 0.03, Pet 0.97, Fan 0.01 |
| E14 | 1717 (Human 1717) | 0.501 | 0.000 (Pet) | Human 0.50 |
| E15 | 1499 (Human 498, Pet 1001) | 0.560 | 0.332 (Human) | Human 0.52, Pet 0.58 |
| E16 | 2343 (Human 1922, Pet 421) | 0.533 | 0.180 (Pet) | Human 0.54, Pet 0.52 |
| E17 | 2344 (Human 1919, Pet 425) | 0.475 | 0.181 (Pet) | Human 0.43, Pet 0.68 |
| E18 | 4397 (Human 1925, Pet 2472) | 0.543 | 0.438 (Human) | Human 0.48, Pet 0.59 |
| E19 | 5567 (Human 2634, IRobot 2933) | 0.392 | 0.000 (Pet) | Human 0.71, IRobot 0.11 |
| E20 | 16326 (Pet 16326) | 0.106 | 0.000 (Human) | Pet 0.11 |

## All rooms pooled (every window held out once)

Windows 60858, accuracy 0.248, macro-F1 0.159.

| | Human recall | Pet recall | IRobot recall | Fan recall |
|---|---|---|---|---|
| Pooled | 0.26 | 0.28 | 0.08 | 0.01 |

Is it a person? Precision 0.345, recall 0.256, F1 0.294.
