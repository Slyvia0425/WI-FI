# Data format and mapping

## Source

- Dataset: `csi-bench`
- Task: `HumanActivityRecognition`
- No dataset file is committed to Git.
- `HAR_DATA_ROOT` or `CSI_DATA_ROOT` points to the dataset root.

Expected task layout is configurable. The default is:

```text
<local_subset_root>/
  Multitask/
    HumanActivityRecognition/
      metadata/sample_metadata.csv
      metadata/label_mapping.json
      splits/
    sub_Human_h5/
      **/*.h5
```

Only these two directories are required. Do not extract unrelated task directories. Use
`scripts/prepare_local_subset.py` to copy or extract only this subset and verify the result.

## Sample record

| Field | Meaning |
|---|---|
| `sample_id` | deterministic BLAKE2b id; unique within the manifest |
| `path` | resolved HDF5 path |
| `session_id` | recording session |
| `user_id` | participant |
| `environment_id` | room or environment |
| `device_id` | hardware configuration |
| `activity` | canonical stable id |
| `split` | `train`, `val` or `test` |

## Array rules

- 2-D array with the configured feature dimension equal to 232 by default.
- `(feature, time)` orientation is canonical; `(time, feature)` is transposed only after quality checks.
- Non-finite values, empty arrays, too-short windows and extreme amplitudes are rejected.
- Normalisation statistics are fitted on the training split only.
- Split guard rejects the same user, environment, device or session appearing in more than one split.

## Label mapping

Canonical ids are stable and independent of display text:

```text
0 standing
1 sitting
2 walking
3 lying
```

`transition` and `unknown` are decision outputs, not training classes.
