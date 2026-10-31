# Inference time of one window, one CPU thread (200 validation windows of split 0)

| Candidate | Features | Classify (median ms) | Whole path: median ms | 95th percentile ms | Share of the 5 s | Model MB | Also ship MB |
|---|---|---|---|---|---|---|---|
| Random Forest on statistics | statistics | 7.64 | 9.29 | 9.7 | 0.19% | 6.37 | 0.0 |
| Logistic Regression on MiniRocket | minirocket | 0.22 | 21.82 | 22.29 | 0.44% | 0.38 | 0.08 |
| Random Forest on MiniRocket | minirocket | 7.46 | 29.38 | 29.96 | 0.59% | 13.48 | 0.08 |
| Random Forest on statistics + physical | statistics_physical | 7.59 | 15.02 | 15.54 | 0.3% | 15.17 | 0.0 |
| Logistic Regression on statistics + physical | statistics_physical | 0.19 | 7.52 | 7.92 | 0.15% | 0.09 | 0.0 |

Building blocks (median / 95th percentile, ms):

| Block | Median | 95th |
|---|---|---|
| statistics | 1.79 | 1.98 |
| minirocket | 19.88 | 20.26 |
| spectral_shape | 0.81 | 0.98 |
| physical_raw_numbers | 4.28 | 4.51 |
| assemble_with_dynamics | 0.54 | 0.61 |

MiniRocket first call (cold start, compiling): 4976.1 ms. Reading the file and the memory in use at run time are not measured.
