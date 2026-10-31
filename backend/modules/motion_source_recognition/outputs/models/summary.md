| Model | Split | Parameters | Macro-F1 | Accuracy | Human recall | Pet recall | IRobot recall | Fan recall | Training seconds | Model MB |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0 | {'C': 0.01, 'max_iter': 300} | 0.469 | 0.893 | 0.97 | 0.98 | 0.00 | 0.00 | 644 | 0.4 |
| logistic_regression | 1 | {'C': 1.0, 'max_iter': 300} | 0.643 | 0.863 | 0.97 | 0.95 | 0.00 | 0.99 | 627 | 0.4 |
| logistic_regression | 2 | {'C': 3e-06, 'max_iter': 300} | 0.544 | 0.758 | 0.85 | 0.72 | 0.77 | 0.00 | 664 | 0.4 |
| logistic_regression_statistics_physical | 0 | {'C': 0.001, 'max_iter': 300} | 0.480 | 0.767 | 0.81 | 0.84 | 0.32 | 0.00 | 90 | 0.1 |
| logistic_regression_statistics_physical | 1 | {'C': 1.0, 'max_iter': 300} | 0.513 | 0.734 | 0.85 | 0.80 | 0.07 | 0.43 | 94 | 0.1 |
| logistic_regression_statistics_physical | 2 | {'C': 0.01, 'max_iter': 300} | 0.578 | 0.803 | 0.87 | 0.82 | 0.55 | 0.05 | 89 | 0.1 |
| random_forest | 0 | {'min_samples_leaf': 50, 'max_features': 'sqrt', 'n_estimators': 300} | 0.469 | 0.778 | 0.84 | 0.84 | 0.14 | 0.05 | 420 | 6.4 |
| random_forest | 1 | {'min_samples_leaf': 50, 'max_features': 'sqrt', 'n_estimators': 300} | 0.564 | 0.771 | 0.87 | 0.84 | 0.13 | 0.67 | 420 | 6.4 |
| random_forest | 2 | {'min_samples_leaf': 50, 'max_features': 'sqrt', 'n_estimators': 300} | 0.521 | 0.758 | 0.87 | 0.72 | 0.39 | 0.17 | 355 | 5.2 |
| random_forest_minirocket | 0 | {'min_samples_leaf': 2, 'max_features': 'sqrt', 'n_estimators': 300} | 0.460 | 0.879 | 0.97 | 0.95 | 0.00 | 0.00 | 166 | 13.5 |
| random_forest_minirocket | 1 | {'min_samples_leaf': 5, 'max_features': 'sqrt', 'n_estimators': 300} | 0.655 | 0.829 | 0.95 | 0.91 | 0.00 | 0.78 | 142 | 12.5 |
| random_forest_minirocket | 2 | {'min_samples_leaf': 50, 'max_features': 'sqrt', 'n_estimators': 300} | 0.476 | 0.850 | 0.96 | 0.87 | 0.07 | 0.00 | 79 | 4.3 |
| random_forest_statistics_physical | 0 | {'min_samples_leaf': 5, 'max_features': 'sqrt', 'n_estimators': 300} | 0.447 | 0.855 | 0.94 | 0.92 | 0.00 | 0.00 | 238 | 15.2 |
| random_forest_statistics_physical | 1 | {'min_samples_leaf': 5, 'max_features': 'sqrt', 'n_estimators': 300} | 0.549 | 0.808 | 0.94 | 0.90 | 0.00 | 0.44 | 250 | 15.7 |
| random_forest_statistics_physical | 2 | {'min_samples_leaf': 50, 'max_features': 'sqrt', 'n_estimators': 300} | 0.565 | 0.805 | 0.94 | 0.75 | 0.46 | 0.11 | 204 | 4.9 |

Mean ± standard deviation over the splits (sample standard deviation, n = 3).

| Model | Macro-F1 | Accuracy |
|---|---|---|
| logistic_regression | 0.552 ± 0.087 | 0.838 ± 0.071 |
| logistic_regression_statistics_physical | 0.524 ± 0.050 | 0.768 ± 0.034 |
| random_forest | 0.518 ± 0.047 | 0.769 ± 0.011 |
| random_forest_minirocket | 0.530 ± 0.108 | 0.852 ± 0.025 |
| random_forest_statistics_physical | 0.520 ± 0.064 | 0.823 ± 0.028 |
