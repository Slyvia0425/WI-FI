# Test scores (scored once)

Created 2026-10-18T04:52:58+00:00. Fingerprints of the saved test probabilities were checked first. Main model: **Logistic Regression on MiniRocket**.

## Test windows and recording groups

| Split | Human | Pet | IRobot | Fan |
|---|---|---|---|---|
| split_0 | 4364 windows, 18 groups | 4366 windows, 19 groups | 1012 windows, 1 groups | 361 windows, 1 groups |
| split_1 | 4241 windows, 21 groups | 3866 windows, 18 groups | 361 windows, 1 groups | 423 windows, 1 groups |
| split_2 | 4464 windows, 20 groups | 4112 windows, 21 groups | 2933 windows, 1 groups | 356 windows, 1 groups |

## Test scores (mean ± standard deviation over the 3 splits; range = 95% of group resamples within each class, of the 3-split mean; IRobot and Fan have one test group per split, so the range cannot show their uncertainty)

| Model | Macro-F1 | Range | Accuracy | Range | Human recall | Pet recall | IRobot recall | Fan recall | With smoothing (macro-F1) | Validation macro-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression on MiniRocket | 0.430 ± 0.059 | [0.415, 0.443] | 0.766 ± 0.164 | [0.736, 0.791] | 0.93 | 0.89 | 0.01 | 0.00 | 0.435 | 0.552 |
| Random Forest on MiniRocket | 0.435 ± 0.036 | [0.422, 0.446] | 0.788 ± 0.108 | [0.762, 0.811] | 0.96 | 0.93 | 0.00 | 0.00 | 0.436 | 0.530 |
| Random Forest on statistics | 0.434 ± 0.019 | [0.41, 0.451] | 0.713 ± 0.076 | [0.67, 0.748] | 0.84 | 0.84 | 0.08 | 0.03 | 0.431 | 0.518 |
| Random Forest on statistics + physical * | 0.432 ± 0.020 | [0.416, 0.443] | 0.771 ± 0.109 | [0.741, 0.794] | 0.94 | 0.89 | 0.01 | 0.01 | 0.428 | 0.520 |
| Logistic Regression on statistics + physical * | 0.448 ± 0.042 | [0.43, 0.464] | 0.704 ± 0.058 | [0.672, 0.733] | 0.81 | 0.83 | 0.19 | 0.00 | 0.469 | 0.524 |

## Per split: macro-F1 (95% range) and the two rare classes

| Model | Split | Macro-F1 | Range | Accuracy | IRobot precision / recall | Fan precision / recall |
|---|---|---|---|---|---|---|
| Logistic Regression on MiniRocket | split_0 | 0.450 | [0.429, 0.463] | 0.836 | 0.00 / 0.00 | 0.00 / 0.00 |
| Logistic Regression on MiniRocket | split_1 | 0.476 | [0.467, 0.484] | 0.882 | 0.00 / 0.00 | 0.03 / 0.01 |
| Logistic Regression on MiniRocket | split_2 | 0.364 | [0.322, 0.397] | 0.579 | 0.20 / 0.03 | 0.00 / 0.00 |
| Random Forest on MiniRocket | split_0 | 0.440 | [0.414, 0.459] | 0.817 | 0.00 / 0.00 | 0.00 / 0.00 |
| Random Forest on MiniRocket | split_1 | 0.468 | [0.451, 0.481] | 0.879 | 0.00 / 0.00 | 0.00 / 0.00 |
| Random Forest on MiniRocket | split_2 | 0.396 | [0.367, 0.418] | 0.669 | 0.01 / 0.00 | 0.00 / 0.00 |
| Random Forest on statistics | split_0 | 0.444 | [0.404, 0.477] | 0.740 | 0.37 / 0.11 | 0.00 / 0.00 |
| Random Forest on statistics | split_1 | 0.446 | [0.412, 0.474] | 0.771 | 0.07 / 0.06 | 0.03 / 0.03 |
| Random Forest on statistics | split_2 | 0.411 | [0.372, 0.441] | 0.627 | 0.35 / 0.06 | 0.03 / 0.06 |
| Random Forest on statistics + physical * | split_0 | 0.431 | [0.405, 0.45] | 0.802 | 0.00 / 0.00 | 0.00 / 0.00 |
| Random Forest on statistics + physical * | split_1 | 0.452 | [0.433, 0.467] | 0.861 | 0.00 / 0.00 | 0.00 / 0.00 |
| Random Forest on statistics + physical * | split_2 | 0.412 | [0.381, 0.436] | 0.651 | 0.41 / 0.04 | 0.05 / 0.04 |
| Logistic Regression on statistics + physical * | split_0 | 0.493 | [0.457, 0.524] | 0.721 | 0.52 / 0.35 | 0.00 / 0.00 |
| Logistic Regression on statistics + physical * | split_1 | 0.439 | [0.416, 0.461] | 0.752 | 0.17 / 0.13 | 0.00 / 0.00 |
| Logistic Regression on statistics + physical * | split_2 | 0.411 | [0.382, 0.435] | 0.639 | 0.73 / 0.09 | 0.01 / 0.01 |

## Main model minus each other candidate (macro-F1, 3-split mean)

| Compared with | Difference | 95% range | Share of resamples where the main model is better |
|---|---|---|---|
| Random Forest on MiniRocket | -0.005 | [-0.015, 0.005] | 19% |
| Random Forest on statistics | -0.003 | [-0.016, 0.012] | 34% |
| Random Forest on statistics + physical * | -0.001 | [-0.01, 0.006] | 35% |
| Logistic Regression on statistics + physical * | -0.018 | [-0.029, -0.006] | 0% |

\* Added after the model choice was frozen. The physical features were designed after looking at the recordings of the whole dataset (the E11 robot and fan recordings, and the differences between classes over all rooms), which include windows of the validation and test sets, so the scores of these two models are more optimistic than the others'. The frozen choice is not changed by them.
