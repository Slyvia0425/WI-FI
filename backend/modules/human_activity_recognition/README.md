# Human activity recognition (Module B)

The module converts a five-second csi-bench CSI window into an explainable single-person activity decision:

```text
standing | sitting | walking | lying | transition | unknown
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
python3 backend/modules/human_activity_recognition/scripts/run_local_fixture.py --epochs 2
```

Prepare only the module's real-data subset from an existing Kaggle ZIP:

```bash
python3 backend/modules/human_activity_recognition/scripts/prepare_local_subset.py \
  --archive /path/to/csi-bench.zip \
  --destination backend/modules/human_activity_recognition/local_data/csi_bench_module_b
export HAR_DATA_ROOT="$PWD/backend/modules/human_activity_recognition/local_data/csi_bench_module_b"
python3 -m backend.modules.human_activity_recognition.cli manifest
python3 -m backend.modules.human_activity_recognition.cli train --model cnn_transformer
```

The subset contains only `Multitask/HumanActivityRecognition/` and `Multitask/sub_Human_h5/`. Unrelated task directories are never extracted.

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

The code path and local HDF5 fixture pipeline are implemented and tested. The full csi-bench dataset is not committed; extract only the Module B subset with `prepare_local_subset.py`. Real benchmark numbers remain pending until that subset is available locally, so the bundled website replay remains a fixture (`mock: true`).
