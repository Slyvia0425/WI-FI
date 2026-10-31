# Environment and runbook

## Requirements

- Python 3.10+
- Module dependencies: `pip install -r backend/modules/human_activity_recognition/requirements.txt`
- Development checks: `pip install -r backend/modules/human_activity_recognition/requirements-dev.txt`

## Required environment variables

| Variable | Purpose | Required |
|---|---|---|
| `HAR_DATA_ROOT` or `CSI_DATA_ROOT` | root containing only `Multitask/HumanActivityRecognition` and `Multitask/sub_Human_h5` | for training with real data |
| `HAR_OUTPUT_ROOT` | checkpoints, manifests and metrics | optional; defaults to `human_activity_recognition` |
| `HAR_CONFIG` | alternate config path | optional |
| `HAR_REMOTE_ENDPOINT` | remote inference endpoint | for `remote` mode |
| `HAR_REMOTE_TOKEN` | remote bearer token | only when the endpoint requires it |

Never commit `.env.local`, tokens, dataset files, checkpoints or generated logs.

## Prepare only Module B data

Do not extract the entire 81 GB dataset for development. From an already downloaded Kaggle archive:

```bash
python3 backend/modules/human_activity_recognition/scripts/prepare_local_subset.py \
  --archive /path/to/csi-bench.zip \
  --destination backend/modules/human_activity_recognition/local_data/csi_bench_module_b
export HAR_DATA_ROOT="$PWD/backend/modules/human_activity_recognition/local_data/csi_bench_module_b"
```

The script copies only `Multitask/HumanActivityRecognition/` and `Multitask/sub_Human_h5/`, writes a subset manifest, rejects path traversal and never modifies the source.

## Offline fixture

A deterministic 408 KB HDF5 fixture is committed under `fixtures/mini_csi_bench`:

```bash
python3 backend/modules/human_activity_recognition/scripts/run_local_fixture.py
```

It exercises the real HDF5 reader, metadata and split loader, training loop and replay export without downloading csi-bench. Its metrics are synthetic and must not be reported as benchmark results.

## Modes

### Local

`runtime.mode: local`. Uses local dataset and checkpoint.

### Remote

`runtime.mode: remote`. Uses `HttpTransport` with timeout, retries and classification. A missing endpoint fails before a request is sent.

### Hybrid

`runtime.mode: hybrid`. Tries remote first and falls back to local when `allow_local_fallback` is true. Degraded responses include `degraded: true` and `degradation_reason`.

## Standard checks

```bash
ruff check --config backend/modules/human_activity_recognition/pyproject.toml backend/modules/human_activity_recognition
mypy --config-file backend/modules/human_activity_recognition/pyproject.toml backend/modules/human_activity_recognition
bandit -q -r backend/modules/human_activity_recognition
python3 -m unittest discover -s backend/modules/human_activity_recognition/tests -t .
python3 -m unittest discover -s backend/tests -t .
npm --prefix frontend run build
```

## Operational limits

- Batch size, concurrency, request rate and input bytes are bounded by `runtime` config.
- Export validation happens before files are written.
- A failed module does not take down other API routes.
