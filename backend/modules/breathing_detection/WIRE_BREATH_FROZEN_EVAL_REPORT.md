# Frozen LSTM evaluation on WiRe-Breath

## Purpose and constraints

The frozen CSI-Bench seed-123 LSTM was evaluated without retraining or
threshold adjustment on the model-ready WiRe-Breath cache. The checkpoint
threshold remained `0.0545424558`; its SHA-256 is
`fff3cdb28c7577489400be7cc82506f074703ea22425404be1cc3224923df956`.

Every WiRe-Breath window contains a breathing participant. This evaluation can
therefore measure positive-class recall and assessability coverage, but it
cannot measure specificity, false-positive rate, or Macro-F1. It is an
external-domain diagnostic, not a replacement for the frozen CSI-Bench test.

## Main result

| Measure | Result |
|---|---:|
| Total 10-second windows | 1,401 |
| Assessable under the initial timestamp rule | 1,293 (92.29%) |
| Marked not assessable | 108 (7.71%) |
| Breathing detected among assessable windows | 1,257 |
| False `none` among assessable windows | 36 |
| Positive-class recall among assessable windows | 97.22% |
| Breathing output over all windows | 89.72% |

The high external recall is encouraging evidence that the frozen classifier
did not depend entirely on CSI-Bench-specific sample dimensions. It does not
show that the model can reject empty rooms in the WiRe-Breath device domain,
because this dataset contains no matching empty-room class.

## Environment and room slices

| Slice | Assessable / total | Recall |
|---|---:|---:|
| Quiet | 670 / 720 | 95.37% |
| Interference | 623 / 681 | 99.20% |
| Room1 | 402 / 510 | 98.51% |
| Room2 | 891 / 891 | 96.63% |

Recorded motion interference did not reduce recall in this dataset. This is
not evidence that interference is harmless in general: environment, posture,
participant, and room are not balanced independently, and WiRe-Breath's
`Interference` condition still contains a breathing person.

All 108 windows rejected by the initial 100 ms timestamp-gap rule are from
Room1. The frozen model nevertheless detects breathing in 106 of those 108
windows when they are scored diagnostically. The quality rule must therefore
undergo a gap-threshold sensitivity study before it can be treated as a final
production gate; the flagged windows remain excluded from the headline recall.

## Participant and posture findings

Participant recalls among assessable windows are:

| Participant | Recall |
|---|---:|
| 01 | 98.43% |
| 02 | 93.62% |
| 03 | 100.00% |
| 04 | 100.00% |
| 05 | 98.83% |

Most postures achieve at least 95% recall. Sitting is the clear exception at
83.33% (140/168). This does not represent a general Sitting failure: 28 of all
36 false-`none` windows come from one recording,
`Room2/Subject02/Sitting/Quiet/0816/Trial01`. That recording contains 30
windows and the model rejects 28, many with probabilities close to zero.

The remaining eight errors are distributed over five recordings:

- Subject01 Soldier Interference: 3;
- Subject05 Semi Quiet: 2;
- Subject01 Freefaller Interference: 1;
- Subject01 Normal Interference: 1; and
- Subject01 Normal Quiet: 1.

The concentration and extreme confidence make threshold lowering an
inappropriate fix. The next analysis should compare the failed recording's
CSI spectrum, temporal variance, respiratory-reference rate/amplitude, and
subcarrier agreement against successful Subject02 Sitting and other-posture
recordings. Any adaptation must use recording- or subject-grouped validation,
not these same failed windows for both diagnosis and model selection.

That signal-level follow-up is complete. The tested periodicity and quality
features do not explain the record's near-zero model probability, while the
timestamp analysis supports retaining the 100 ms gap rule. See
[`WIRE_BREATH_SIGNAL_ERROR_ANALYSIS.md`](WIRE_BREATH_SIGNAL_ERROR_ANALYSIS.md).

## Reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.evaluate_wire_breath \
  --checkpoint backend/modules/breathing_detection/outputs/\
lstm_interpolated_full_seed123/best_model.pt \
  --cache-dir backend/modules/breathing_detection/outputs/wire_breath_cache \
  --output backend/modules/breathing_detection/outputs/\
wire_breath_frozen_seed123/evaluation.json
```

The JSON output includes every false-`none` sample ID and probability, plus
environment, participant, room, posture, and quality slices.
