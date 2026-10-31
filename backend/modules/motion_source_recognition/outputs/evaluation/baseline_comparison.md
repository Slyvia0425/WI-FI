# Logistic Regression on MiniRocket against the two reference lines (test rows)

| | Macro-F1 (mean ± std) | Range (95%) | Accuracy (mean ± std) | Human recall | Pet recall | IRobot recall | Fan recall |
|---|---|---|---|---|---|---|---|
| Main model | 0.430 ± 0.059 | [0.415, 0.443] | 0.766 ± 0.164 | 0.93 | 0.89 | 0.01 | 0.00 |
| Always the most frequent class | 0.150 ± 0.012 | [0.143, 0.156] | 0.428 ± 0.050 | 1.00 | 0.00 | 0.00 | 0.00 |
| Room lookup | 0.376 ± 0.014 | [0.337, 0.404] | 0.680 ± 0.053 | 0.83 | 0.81 | 0.00 | 0.00 |

## Gap of the main model to each line (3-split mean)

| Line | Macro-F1 gap | 95% range | Accuracy gap | Share of resamples where the model is better |
|---|---|---|---|---|
| Always the most frequent class | +0.280 | [0.266, 0.294] | +0.338 | 100% |
| Room lookup | +0.054 | [0.03, 0.087] | +0.086 | 100% |

## Per class F1 (3-split mean)

| | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| Main model | 0.84 | 0.86 | 0.02 | 0.01 |
| Always the most frequent class | 0.60 | 0.00 | 0.00 | 0.00 |
| Room lookup | 0.75 | 0.75 | 0.00 | 0.00 |

## Per split

| Split | Main model macro-F1 / accuracy | Most frequent class | Room lookup | Windows in a room unseen in training |
|---|---|---|---|---|
| split_0 | 0.450 / 0.836 | 0.151 / 0.432 (Human) | 0.391 / 0.724 | 0% |
| split_1 | 0.476 / 0.882 | 0.162 / 0.477 (Human) | 0.364 / 0.696 | 0% |
| split_2 | 0.364 / 0.579 | 0.137 / 0.376 (Human) | 0.372 / 0.621 | 25% |
