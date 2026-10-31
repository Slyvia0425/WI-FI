# Periodicity baseline report

## Scope

This is the first full interpretable baseline for the downloaded CSI-Bench
BreathingDetection dataset. It is a non-medical binary classifier for
`sleep`/breathing evidence versus `empty` and is not a respiratory-rate or
apnoea detector.

The model uses the official train/validation/test ID files. Model parameters
are fitted on train, the decision threshold is selected once on validation
macro-F1, and test is used only for final evaluation.

## Configuration

- Input: project pipeline `[1, K, 300]`, preserving the real 10-second window.
- Up to 64 highest-variance subcarriers per sample.
- Candidate breathing band: 0.1--0.7 Hz.
- Features: 18 FFT, autocorrelation, cross-subcarrier consistency, waveform
  agreement, signal spread, entropy, and high-frequency motion features.
- Classifier: balanced logistic regression with L2 regularisation.
- Validation-selected threshold: `0.3882897939612134`.

Three train files and one test file failed the minimal assessability gate and
were excluded. Validation contained no excluded files.

## Overall results

| Split | N | Accuracy | Precision | Recall | Positive F1 | Macro-F1 | Empty-room FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 69,997 | 0.7925 | 0.8648 | 0.8798 | 0.8722 | 0.6599 | 0.5683 |
| Validation | 15,000 | 0.7960 | 0.8706 | 0.8783 | 0.8744 | 0.6656 | 0.5517 |
| Test | 14,999 | 0.7900 | 0.8661 | 0.8746 | 0.8703 | 0.6591 | 0.5611 |

Test confusion matrix: TN 1,278; FP 1,634; FN 1,516; TP 10,571.

The positive-class F1 is relatively high because `sleep` is the majority
class. Macro-F1 and empty-room FPR reveal the important limitation: periodic
features alone frequently mistake empty-room variation for breathing evidence.

## Test slices

| Slice | N | Macro-F1 | Empty-room FPR |
|---|---:|---:|---:|
| Easy | 6,688 | 0.6360 | 0.4909 |
| Medium | 4,405 | 0.6557 | 0.6120 |
| Hard | 3,906 | 0.6839 | 0.5775 |
| Fan off | 2,358 | 0.6045 | 0.4085 |
| Fan on | 1,798 | 0.6787 | 0.2810 |
| NLoS | 2,386 | 0.6332 | 0.2937 |
| LoS | 1,770 | 0.6065 | 0.5141 |

Fan and LoS fields exist only where deployment metadata states them explicitly;
these rows are subsets rather than the entire test set. The unexpectedly lower
FanOn FPR should not be interpreted as fan robustness without controlling for
device, environment, deployment and label composition.

## Most influential standardized coefficients

Positive coefficients increase predicted breathing evidence; negative
coefficients reduce it.

| Feature | Coefficient |
|---|---:|
| Signal standard deviation median | +0.4843 |
| Maximum breathing-band power ratio | +0.3461 |
| Median spectral entropy | +0.3178 |
| Autocorrelation peak Q75 | -0.2369 |
| Median autocorrelation frequency | -0.2020 |
| Median waveform correlation | +0.1940 |
| Peak-frequency IQR | -0.1586 |
| Peak-power share Q75 | +0.1484 |

The coefficients confirm that the classifier uses more than one spectral peak,
but they also show that simple periodicity is not sufficient to separate all
empty-room windows. This baseline is therefore an interpretable lower bound and
a diagnostic comparison for the LSTM, not the final production detector.

## Reproduction artifacts

Generated, Git-ignored artifacts are stored under:

```text
backend/modules/breathing_detection/outputs/periodicity_baseline_full/
├── feature_cache/
│   ├── features_v1_train_id_nall_k64.npz
│   ├── features_v1_val_id_nall_k64.npz
│   └── features_v1_test_id_nall_k64.npz
├── model.json
└── metrics.json
```

Re-run from the repository root:

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.training.train_periodicity_baseline \
  --data-root data/BreathingDetection \
  --output-dir backend/modules/breathing_detection/outputs/periodicity_baseline_full \
  --evaluate-test
```
