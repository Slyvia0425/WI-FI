# Model card

## Intended use

Research prototype for single-person indoor WiFi CSI activity recognition. It is not a medical device and does not issue fall alerts.

## Classes

`jumping`, `running`, `seated-breathing`, `walking`, `wavinghand`; `transition` and `unknown` are decision states.

## Models

- `cnn_1d`: local temporal baseline.
- `cnn_transformer`: convolutional front end followed by a Transformer encoder.

## Evaluation

- Macro-F1 primary target: `>= 0.80`.
- Current 3,000-session real-data subset (2,000 train / 500 validation / 500 test), three seeds: accuracy `0.907 ± 0.008`, Macro-F1 `0.907 ± 0.008` on the official test split.
- The result meets the `0.80` Macro-F1 target on this in-distribution HP-device subset; cross-device validation is still required.
- Per-class precision, recall and confusion matrix.
- Expected calibration error.
- End-to-end latency under one second for the five-second window.
- Cross-user, cross-environment and cross-device robustness.

## Known limitations

- The repository does not include the 81 GB csi-bench download; the ignored local subset has 3,000 sessions.
- The subset used for this evaluation contains only the `HP` device; cross-device robustness is unverified.
- Jumping and running account for most remaining errors; all classes exceed 0.88 F1 on the held-out test split.
- The bundled website replay remains a fixture until a reviewed real run is exported.
- The five official classes are treated as mutually exclusive stable states; transitions are a separate decision state.
- Multi-person co-occurrence is out of scope.
