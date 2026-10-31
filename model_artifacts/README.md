# Module B + C trained model files

The matching model assets are published in the same repository's [bc-models-20261031-v1 release](https://github.com/Slyvia0425/WI-FI/releases/tag/bc-models-20261031-v1). `manifest.json` records the SHA256, byte size, role and destination of all 35 checkpoints and their companion files.

The code stays in Git. Binary model files stay in Release assets, preserving the existing model versions and output paths. Raw CSI, NumPy/intermediate feature caches and duplicate older delivery ZIPs are excluded.

## Restore the application models and the new candidate

From the repository root, using Python3.10 or later (the downloader uses only the standard library):

```bash
python3 scripts/restore_bc_models.py
```

This downloads `module_bc_runtime_models.tar.gz` and `SHA256SUMS`. It restores:

- The existing B seed42 spectral ExtraTrees model and its original evaluation/configuration files.
- The existing C CSI-Bench ResNet18 checkpoint and frozen B/C routing configuration.
- The independently trained HP mixed-data ResNet18 candidate and its training metadata. It is **not** substituted for the application model.
- The archived TCS D checkpoint and replay evidence used by the historical validation check.

Every archive and restored file is checked against its SHA256. Identical local files are retained; a different existing file causes an explicit error. To inspect the release separately, add `--output-root /path/to/empty/folder`.

## Restore every local trained-model version

```bash
python3 scripts/restore_bc_models.py --all
```

The additional `module_bc_research_models.tar.gz` contains the other B model/seed versions, C 1D-CNN/SenseFi/CNN+LSTM/MLP/ResNet seed variants, historical TCS models and saved optimizer/resume checkpoints. These remain research artifacts and are not automatically promoted into the application.

For an offline copy, download both archives and `SHA256SUMS` from the Release page, then run:

```bash
python3 scripts/restore_bc_models.py --all --offline --archives-dir /path/to/downloads
```

Install model dependencies from `backend/modules/fall_detection/requirements.txt` before inference. This restore operation does not train, retune, deserialize or run a model. Inference commands and validation boundaries remain in [the B/C delivery guide](../docs/BC_RESNET18_CODE_DELIVERY.md).
