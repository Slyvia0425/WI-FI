# Breathing Detection data audit

This is a minimal, read-only audit completed before implementing a data loader
or training a model. The downloaded data is the source of truth; values from
the proposal remain assumptions until confirmed here or in the official code.

## Dataset location and integrity

- Local root: `data/BreathingDetection` (ignored by Git).
- Size on disk: approximately 12 GB.
- Metadata rows / HDF5 files: 100,000 / 100,000.
- Every metadata path exists and metadata IDs are unique.
- Labels: `sleep` = 80,580; `empty` = 19,420.
- Official splits: train = 70,000; validation = 15,000; test = 15,000.
- Difficulty-specific test lists partition the test set: easy = 6,689,
  medium = 4,405, hard = 3,906.

The downloaded label mapping defines `empty` as class 0 and `sleep` as class
1. For this project, these are interpreted as `no_breathing_evidence` and
`breathing_evidence`. They do not constitute a clinical respiratory label.

## HDF5 storage schema

A deterministic stratified sample of 66 files was inspected, covering all
available label/user/difficulty/device combinations.

- Every sampled file contains one dataset named `CSI_amps`.
- Dtype is consistently `float32`.
- Stored layout is `[K, T, C]`, where `K` is the number of subcarriers,
  `T = 300` time points, and `C = 1` channel.
- Observed shapes:

| Shape | Sampled files |
|---|---:|
| `[208, 300, 1]` | 47 |
| `[240, 300, 1]` | 14 |
| `[112, 300, 1]` | 4 |
| `[416, 300, 1]` | 1 |

- No NaN or infinite values were found in the 66 sampled files.
- HDF5 root and dataset attributes are empty; labels and scenario information
  must therefore come from `sample_metadata.csv`.
- Sample means and standard deviations are on the raw amplitude scale (for
  example, means around 70--145), so the downloaded arrays have not already
  been per-sample z-score normalized.

Assuming the documented 30 Hz sampling rate, 300 points represent a 10-second
window. The fixed time dimension supports that interpretation, but the HDF5
files contain no timestamps or sampling-rate attributes of their own.

## Metadata available for evaluation slices

The metadata provides sample ID, relative path, user, activity, environment,
device, subcarrier count, frequency band, difficulty, deployment, session,
multi-subject flag, label, and sometimes date.

- Users/environments: U06/E10, U23/E09, U24/E05.
- Frequency band is 5 GHz for every row.
- Deployment text includes explicit `LOS`/`NLOS` and `FanOn`/`FanOff` values for
  part of the dataset. Other deployments do not encode those fields, so they
  must be represented as unknown rather than inferred.
- `fan` is scenario metadata, not a third target class.

## Implementation decisions based on the real data

1. The loader must read `CSI_amps` and validate `[K, 300, 1]`; it must not
   silently reshape unknown layouts.
2. The canonical internal representation will be `[C, K, T]`, obtained with an
   explicit axis permutation. LSTM input will later be `[T, C*K]`.
3. Per-sample z-score normalization, if enabled, must happen in project code;
   it has not already been applied to the downloaded amplitudes.
4. The official split files will be used rather than generating a random
   window-level split.
5. The class distribution is substantially different from the proposal's
   approximate description. Class weighting and decision thresholds must be
   derived from the real training and validation sets.
6. `uncertain` and `not_assessable` remain application states, not training
   labels.

## Remaining decision before model implementation

The subcarrier dimension varies by device (112/208/240/416). A fixed-size LSTM
cannot consume these arrays directly. Before finalizing the loader and LSTM,
the official CSI-Bench preprocessing/code should be checked for its published
padding or clipping rule. If no authoritative rule exists, candidate policies
must be compared on validation data and documented; no arbitrary target size
will be selected from the test set.

A periodicity baseline can be designed to aggregate per-subcarrier features
and therefore tolerate variable `K`, but it should still share the official
split and normalization policy with the LSTM comparison.

## Reproducing this audit

From the repository root:

```bash
.venv/bin/python -m backend.modules.breathing_detection.preprocessing.audit_data \
  --data-root data/BreathingDetection --summary
```

This command only reads metadata and a deterministic stratified sample. It
does not modify the original dataset.
