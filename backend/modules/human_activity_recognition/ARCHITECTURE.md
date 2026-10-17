# Module B architecture and boundaries

Module id: `human_activity_recognition`  
Owner: Member B  
Dataset task: `csi-bench/HumanActivityRecognition`  
Primary output contract: `activity`, `confidence`, `stable`, plus backward-compatible optional evidence.

## 1. Goals

Recognise four stable single-person states:

| Stable id | Meaning |
|---|---|
| `standing` | standing still |
| `sitting` | seated |
| `walking` | walking |
| `lying` | lying or reclining |

`transition` and `unknown` are decision states, not training classes.

## 2. Architecture layers

```text
HTTP/CLI input
  -> api.py / cli.py                      interface layer
  -> service.py                           application layer
  -> preprocessing/ training/ evaluation/ domain logic
  -> metrics, model, dataset adapters
  -> config.py, security.py, logging_utils.py, transport.py
```

Allowed dependency direction:

- Interface layer may call service layer.
- Service layer may call domain, preprocessing, model, evaluation and infrastructure helpers.
- Domain and preprocessing code must not import `api.py` or `cli.py`.
- Shared repository code is never imported into frontend code.
- External endpoints are reached only through a transport adapter.

## 3. Directory ownership

```text
backend/modules/human_activity_recognition/
  config.yaml              versioned non-secret defaults
  contract.py              public replay frame contract
  domain.py                stable ids and typed results
  errors.py                classified module errors
  service.py               application use cases
  transport.py             local/remote/hybrid adapter
  preprocessing/           dataset discovery, labels, quality, normalisation
  training/                model definitions and training loop
  experiments/             reproducible model comparisons
  evaluation/              metrics, error analysis, replay export
  tests/                   unit, integration, contract, E2E, security, boundary
  scripts/                 CI and smoke helpers

frontend/src/modules/human_activity_recognition/
  meta.js
  plan.js
  data/replay.json
  index.jsx
  components/              module-owned visual components only
```

## 4. Naming and identifiers

- Module id: `human_activity_recognition`.
- Stable activity ids are lowercase snake-less slugs: `standing`, `sitting`, `walking`, `lying`.
- Sample id: deterministic BLAKE2b hash of source-relative path, row index and source record id.
- Model version: caller supplied, for example `b-0.1.0-20261017`.
- Configuration keys use `snake_case`.
- User-facing display text is never used as a key, join key or routing condition.

## 5. Public frame contract

Required fields remain unchanged for compatibility with the existing website and fusion layer:

```json
{
  "t": 12.0,
  "activity": "walking",
  "confidence": 0.91,
  "stable": true
}
```

Optional additive fields:

- `activity_id`: stable zero-based class id.
- `probabilities`: map of stable ids to probabilities.
- `stability`: 0 to 1 continuity score.
- `quality`: `good`, `degraded` or `invalid`.
- `model_version`: immutable model identifier.
- `transition`: boolean transition flag.

Adding optional fields is compatible. Removing or renaming required fields requires updating `frontend/src/modules/human_activity_recognition/`, fusion documentation and contract tests.

## 6. Data format

Expected csi-bench layout, subject to `config.yaml`:

```text
<local_subset_root>/
  Multitask/
    HumanActivityRecognition/
      metadata/sample_metadata.csv
      metadata/label_mapping.json
      splits/*.json
    sub_Human_h5/**/*.h5
```

Only the two `Multitask` directories above are required for Module B; unrelated task data is never extracted into the local subset.

The loader:

- reads HDF5 arrays named by configured key candidates such as `CSI_amps`;
- accepts 2-D `(feature, time)` windows only;
- preserves user, environment, device and session identifiers;
- rejects non-finite arrays, empty windows and shape mismatches;
- computes normalisation statistics from the training split only;
- validates that group identifiers never cross official split boundaries.

## 7. Error model

All expected failures use typed errors with a stable code:

- `CFG_INVALID`
- `INPUT_INVALID`
- `DATA_NOT_FOUND`
- `DATA_INVALID`
- `SPLIT_LEAKAGE`
- `MODEL_UNAVAILABLE`
- `MODEL_INVALID`
- `INFERENCE_FAILED`
- `TRANSPORT_TIMEOUT`
- `TRANSPORT_UNAVAILABLE`
- `AUTH_DENIED`
- `EXPORT_INVALID`

The CLI and service return a structured error object instead of leaking tracebacks to callers.

## 8. Runtime modes

- `local`: use local dataset/checkpoint.
- `remote`: call the configured remote endpoint.
- `hybrid`: try remote first, then local fallback when policy permits.

All modes expose the same `predict` and `replay` methods. Remote access uses a single transport wrapper with timeout, bounded retry, backoff, redaction and explicit degradation.

## 9. Security and privacy

- Raw dataset reads require the `system` or `researcher` role.
- Replay consumers may use the `viewer` role.
- Paths must resolve inside configured allow-listed roots.
- Secrets are read from environment variables by name; they are never stored in YAML or logs.
- Logs redact authorization, token, secret, password and raw CSI payloads.
- Temporary exports are written only under the configured output root.

## 10. Reproducibility and operations

- Random seeds are recorded in training and experiment outputs.
- Checkpoints contain model name, class order, normalisation statistics, feature shape and configuration hash.
- Every run writes a structured operation record.
- Data and interface migrations include a new version, compatibility note and rollback action in `MIGRATION.md`.

## 11. Acceptance gates

- Format, static and type checks pass for this module.
- Unit, integration, contract, end-to-end, error, empty-data, boundary, permission and performance tests pass.
- Existing repository checks (`npm test`, frontend build) remain green.
- Replay export validates against `contract.py`.
- No dataset, checkpoint, local config or temporary log is committed.
