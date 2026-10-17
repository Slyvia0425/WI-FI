# Model card

## Intended use

Research prototype for single-person indoor WiFi CSI activity recognition. It is not a medical device and does not issue fall alerts.

## Classes

`standing`, `sitting`, `walking`, `lying`; `transition` and `unknown` are decision states.

## Models

- `cnn_1d`: local temporal baseline.
- `cnn_transformer`: convolutional front end followed by a Transformer encoder.

## Evaluation

- Macro-F1 primary target: `>= 0.80`.
- Per-class precision, recall and confusion matrix.
- Expected calibration error.
- End-to-end latency under one second for the five-second window.
- Cross-user, cross-environment and cross-device robustness.

## Known limitations

- The repository does not include the 81 GB csi-bench download; real dataset results require the configured local dataset.
- The bundled website replay remains a fixture until a real run is exported.
- The four classes are treated as mutually exclusive stable states; transitions are a separate decision state.
- Multi-person co-occurrence is out of scope.
