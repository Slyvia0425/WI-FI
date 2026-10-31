# Confidence threshold of the main model: calibration and trade-off (validation rows only)

| Split | Temperature T | ECE before | ECE after | Windows with confidence >= 0.99 before / after |
|---|---|---|---|---|
| split_0 | 2.304 | 0.073 | 0.062 | 76% / 0% |
| split_1 | 6.323 | 0.121 | 0.062 | 90% / 2% |
| split_2 | 0.688 | 0.145 | 0.05 | 0% / 0% |

Reject the least confident windows (calibrated confidence); accuracy / macro-F1 of the windows that are kept, share of all errors that are rejected:

| Rejected | split_0 accuracy / macro-F1 / errors rejected | split_1 accuracy / macro-F1 / errors rejected | split_2 accuracy / macro-F1 / errors rejected |
|---|---|---|---|
| 0% | 0.893 / 0.469 / 0% | 0.863 / 0.643 / 0% | 0.758 / 0.544 / 0% |
| 5% | 0.922 / 0.477 / 31% | 0.889 / 0.675 / 23% | 0.775 / 0.559 / 12% |
| 10% | 0.950 / 0.485 / 58% | 0.911 / 0.713 / 41% | 0.794 / 0.572 / 23% |
| 15% | 0.966 / 0.490 / 73% | 0.924 / 0.723 / 53% | 0.809 / 0.575 / 33% |
| 20% | 0.975 / 0.493 / 81% | 0.931 / 0.727 / 60% | 0.824 / 0.571 / 42% |
| 25% | 0.981 / 0.495 / 86% | 0.937 / 0.731 / 66% | 0.835 / 0.569 / 49% |
| 30% | 0.985 / 0.496 / 90% | 0.939 / 0.733 / 69% | 0.850 / 0.577 / 56% |
| 40% | 0.989 / 0.497 / 94% | 0.941 / 0.734 / 74% | 0.869 / 0.554 / 67% |
| 50% | 0.992 / 0.498 / 96% | 0.939 / 0.734 / 78% | 0.892 / 0.513 / 78% |
