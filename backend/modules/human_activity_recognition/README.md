# Human activity recognition (Module B)

The module converts a five-second csi-bench CSI window into an explainable single-person activity decision:

```text
jumping | running | seated-breathing | walking | wavinghand | transition | unknown
```

`transition` and `unknown` are decision states, not training classes.

## Responsibilities

- Read `HumanActivityRecognition` HDF5 data, metadata and official splits.
- Validate labels, shape, finite values, signal quality and split-group isolation.
- Fit normalisation on training data only.
- Train and compare `cnn_1d` and `cnn_transformer`.
- Evaluate macro-F1, per-class precision/recall, confusion, calibration, latency and OOD robustness.
- Publish a contract-safe replay for the website.

## Boundary

This module depends only on its own folders plus the repository's shared contract/export/path helpers.

```text
api.py / cli.py
  -> service.py
  -> domain.py, preprocessing/, training/, experiments/, evaluation/
  -> config.py, security.py, logging_utils.py, resource.py, transport.py
```

- No frontend code imports backend code.
- No downstream module reads Module B internals; only the replay contract is public.
- External endpoints are used only through `transport.py`.
- Required contract fields stay unchanged. Optional fields are documented in `docs/INTERFACE.md`.

## Start here

```bash
pip install -r backend/modules/human_activity_recognition/requirements.txt
python3 -m backend.modules.human_activity_recognition.cli show-config
python3 -m backend.modules.human_activity_recognition.cli self-check
```

Local fixture only:

```bash
python3 backend/modules/human_activity_recognition/scripts/run_local_fixture.py
```

Download a deterministic Module B subset directly from Kaggle:

```bash
python3 backend/modules/human_activity_recognition/scripts/download_official_subset.py \
  --samples-per-class 100 --num-sub 232 --prune
export HAR_DATA_ROOT="$PWD/backend/modules/human_activity_recognition/local_data/csi_bench_module_b"
python3 -m backend.modules.human_activity_recognition.cli manifest
python3 -m backend.modules.human_activity_recognition.cli train-spectral --seed 42
```

Alternatively, extract the same layout from an existing Kaggle ZIP with
`scripts/prepare_local_subset.py`. The subset contains only
`Multitask/HumanActivityRecognition/` and `Multitask/sub_Human_h5/`; unrelated task
directories are never extracted.

Replay-only development:

```bash
python3 -m backend.api --only human_activity_recognition
```

## Configuration

All paths, endpoints, limits and run parameters live in `config.yaml`; environment variables override only the values documented in `docs/RUNBOOK.md`. No address, secret, dataset path or output path is hardcoded in business logic.

## Tests

```bash
python3 -m unittest discover -s backend/modules/human_activity_recognition/tests -t .
python3 -m unittest discover -s backend/tests -t .
npm --prefix frontend run build
```

## Documentation

- `ARCHITECTURE.md` — boundaries and dependency direction
- `docs/INTERFACE.md` — service, API and replay contract
- `docs/DATA_FORMAT.md` — csi-bench layout and label mapping
- `docs/RUNBOOK.md` — local/remote/hybrid operation and env variables
- `docs/MIGRATION.md` — compatibility and rollback
- `docs/MODEL_CARD.md` — intended use and limitations
- `docs/ACCEPTANCE.md` — acceptance evidence

## Current limitation

The code path and local HDF5 fixture pipeline are implemented and tested. The full
csi-bench dataset is not committed. On the ignored 3,000-session Module B subset
(2,000 train / 500 validation / 500 test), the optimized spectral ExtraTrees route
reaches accuracy `0.907 ± 0.008` and Macro-F1 `0.907 ± 0.008` across three seeds,
meeting the `>= 0.80` target. Cross-device validation remains pending. The bundled
website replay now uses the released seed42 spectral model on the same raw HP FallDetection segments as Module C (`mock: false`). These inputs have no five-class HAR annotations, so no B accuracy or joint improvement is claimed. The original fixture is archived.

The released joblib model uses `integration_v1/runtime.py`; the original `service.py` loads CNN PyTorch checkpoints. For current source, commands and limitations see [ResNet18 + B handoff](../../../docs/RESNET_B_HANDOFF.md).
