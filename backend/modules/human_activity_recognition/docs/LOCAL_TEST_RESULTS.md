# Local test results

Date: 2026-10-22

Environment: local macOS CPU, Python 3.13, Module B real-data subset

## Automated checks

| Check | Result |
|---|---|
| Ruff format | passed |
| Ruff static checks | passed |
| Mypy | passed |
| Bandit | passed |
| Module B tests | 45 passed |
| Repository contract tests | 13 passed |
| Committed HDF5 fixture | loaded and trained |
| Subset extraction test | passed |

## Real csi-bench subset

The local development subset comes from the official `guozhenjennzhu/csi-bench`
`HumanActivityRecognition` metadata and splits:

- 232 subcarriers, 500 frames per session
- five mutually exclusive classes
- 2,000 train / 500 validation / 500 test sessions
- 400 train sessions per class, 100 validation/test sessions per class
- 3,000 referenced HDF5 sessions (about 936 MB)
- only referenced HDF5 sessions were downloaded; the full 81 GB archive was not used

The downloader, HDF5 validation, metadata mapping and split checks are deterministic.
All 3,000 HDF5 files were opened successfully after download.

## Optimization progression

| Stage | Split | Accuracy | Macro-F1 |
|---|---|---:|---:|
| 1D-CNN, 500 train sessions, 3 seeds | official test | 0.652 ± 0.017 | 0.647 ± 0.022 |
| 1D-CNN, 2,000 train sessions | official test | 0.848 | 0.848 |
| Spectral ExtraTrees, 3 seeds | validation | 0.897 ± 0.002 | 0.898 ± 0.002 |
| Spectral ExtraTrees, 3 seeds | official test | **0.907 ± 0.008** | **0.907 ± 0.008** |

The final model uses channel-wise log band power, temporal statistics,
training-only feature selection (`k=2800`), and an ExtraTrees ensemble.
Model and feature-selection parameters live in `config.yaml`.

Seed-42 official-test confusion matrix:

| True \ Predicted | jumping | running | seated-breathing | walking | wavinghand |
|---|---:|---:|---:|---:|---:|
| jumping | 88 | 3 | 0 | 0 | 9 |
| running | 4 | 87 | 6 | 3 | 0 |
| seated-breathing | 2 | 2 | 91 | 0 | 5 |
| walking | 4 | 2 | 0 | 93 | 1 |
| wavinghand | 1 | 1 | 1 | 0 | 97 |

Seed-42 per-class F1: jumping `0.884`, running `0.892`,
seated-breathing `0.919`, walking `0.949`, wavinghand `0.915`.

Artifacts:

```text
backend/modules/human_activity_recognition/local_data/optimized_final/spectral/seed-42-stable/
backend/modules/human_activity_recognition/local_data/optimized_final/spectral/report_seed42/
```

## Synthetic fixture experiment

The committed fixture remains a pipeline test, not a benchmark:

| Model | Accuracy mean ± std | Macro-F1 mean ± std |
|---|---:|---:|
| `cnn_1d` | 1.00 ± 0.00 | 1.00 ± 0.00 |
| `cnn_transformer` | 1.00 ± 0.00 | 1.00 ± 0.00 |

## Interpretation

- The optimized model meets the proposal target of Macro-F1 `>= 0.80` on the held-out test split.
- The largest improvement came from expanding balanced training data from 500 to 2,000 sessions,
  followed by spectral feature selection and ExtraTrees.
- The downloaded subset contains the `HP` device only, so cross-device evaluation remains unverified.
- The synthetic fixture's perfect score must not be reported as real csi-bench performance.
