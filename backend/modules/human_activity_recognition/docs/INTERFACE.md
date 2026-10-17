# Interface contract

## HTTP replay endpoint

```text
GET /api/human_activity_recognition/replay
```

Returns the JSON written by `backend.shared.export.write_replay`.

## Required frame fields

| Field | Type | Rule |
|---|---|---|
| `t` | number | non-decreasing replay time in seconds |
| `activity` | string | `standing`, `sitting`, `walking`, `lying`, `transition` or `unknown` |
| `confidence` | number | `[0, 1]` |
| `stable` | boolean | true only for a stable, sufficiently confident state |

## Optional additive fields

| Field | Type | Rule |
|---|---|---|
| `activity_id` | integer or null | 0..3 for stable states, null otherwise |
| `probabilities` | object | stable-id to probability, sums to 1 |
| `stability` | number | `[0, 1]` |
| `quality` | string | `good`, `degraded` or `invalid` |
| `model_version` | string | immutable model identifier |
| `transition` | boolean | true for an explicit transition decision |

Existing required fields are not renamed or removed. Additive fields are safe for older consumers.

## Service API

`HumanActivityService` exposes:

- `health()`
- `replay()`
- `load_checkpoint(path=None, actor="system")`
- `predict(window, actor="researcher")`
- `predict_batch(windows, actor="researcher")`
- `handle(payload)`
- `transport()`

`handle` always returns JSON-serialisable success or a structured error:

```json
{"ok": false, "error": {"code": "MODEL_UNAVAILABLE", "message": "...", "details": {}}}
```

## CLI

```bash
python3 -m backend.modules.human_activity_recognition.cli show-config
python3 -m backend.modules.human_activity_recognition.cli manifest --output outputs/manifest.jsonl
python3 -m backend.modules.human_activity_recognition.cli train-synthetic --epochs 2
python3 -m backend.modules.human_activity_recognition.cli train --model cnn_transformer
python3 -m backend.modules.human_activity_recognition.cli self-check
```
