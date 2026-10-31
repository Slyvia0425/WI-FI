# Module B complete data and model artifacts

The complete Module B dataset and model parameters are published as GitHub Release assets because the dataset is approximately 1 GB and exceeds normal Git repository guidance.

Release: https://github.com/Bartley007/WI-FI/releases/tag/module-b-data-models-v1

## Assets

- `csi_bench_module_b_3000_sessions.tar.zst`
  - 3,000 HDF5 sessions from the official `guozhenjennzhu/csi-bench` HumanActivityRecognition task
  - 2,000 train / 500 validation / 500 test
  - 400 train, 100 validation and 100 test sessions per class
  - metadata, label mapping, official split IDs and subset manifest
- `module_b_models_and_evaluation.tar.zst`
  - final spectral ExtraTrees models for seeds 42, 43 and 44
  - final 1D-CNN checkpoint
  - evaluation JSON, metrics, confusion matrices, replay files and extracted feature cache
  - tuned configuration and optimization summary
- `SHA256SUMS.txt`
- `MODULE_B_ARTIFACTS_README.md`

## Restore

```bash
shasum -a 256 -c SHA256SUMS.txt
tar --zstd -xf csi_bench_module_b_3000_sessions.tar.zst
tar --zstd -xf module_b_models_and_evaluation.tar.zst
```

The data is an unmodified subset of CSI-Bench and is distributed under CC BY-NC-ND 4.0. Attribution belongs to the original CSI-Bench authors. See the release notes for the source and licence details.
