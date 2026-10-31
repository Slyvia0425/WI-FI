# Model selection (validation rows only)

Created 2026-10-17T03:29:44+00:00. Rule: main model = highest mean validation macro-F1 over the three splits; other candidates are comparison rows.

**Main model: Logistic Regression on MiniRocket**

| Candidate | Macro-F1 mean ± std | Accuracy mean ± std | IRobot recall | Fan recall | Unseen-room accuracy (split_2) | ECE per split | Model MB |
|---|---|---|---|---|---|---|---|
| Logistic Regression on MiniRocket | 0.552 ± 0.087 | 0.838 ± 0.071 | 0.26 | 0.33 | 0.779 | [0.073, 0.121, 0.145] | 0.4 |
| Random Forest on MiniRocket | 0.530 ± 0.108 | 0.852 ± 0.025 | 0.02 | 0.26 | 0.948 | [0.049, 0.057, 0.19] | 13.5 |
| Random Forest on statistics | 0.518 ± 0.047 | 0.769 ± 0.011 | 0.22 | 0.29 | 0.784 | [0.114, 0.139, 0.137] | 6.4 |

Gap between the first and second candidate: 0.022; spread (std) of the main model over the splits: 0.087.

Plan for the test scoring:
- all three candidates are scored on the test rows once, with the settings saved here; the test fingerprints below are checked first
- the main numbers are without smoothing; the smoothed version (6 windows) is an extra row
- confusion matrices, error analysis and the confidence threshold are done for the main model only
- if the test ranking differs from the validation ranking, the report says so and the main model stays
