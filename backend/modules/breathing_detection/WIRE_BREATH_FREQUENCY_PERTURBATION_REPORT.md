# Frozen model frequency-order perturbation study

## Method

All 1,401 WiRe-Breath sequences were scored under the same predefined
transforms: circular shifts of -32, -16, -4, -1, +1, +4, +16, +32 positions,
frequency reversal, and edge-replicated shifts of -4, -1, +1, +4 positions.
The time axis, frozen seed-123 checkpoint, and original probability threshold
were unchanged. Shifts apply to the final 232-position model input. A positive
shift moves streams toward higher output indices. Edge replication avoids
wrapping the opposite frequency boundary, but duplicates boundary streams.

Controls use 1,200 breathing and 1,200 empty windows sampled exclusively from
CSI-Bench training, seed 20261028. They are in-sample sensitivity controls,
not new validation or test results. External headline comparisons exclude
the existing 108 quality-flagged windows. Success/failure comparison groups
are fixed using the original unperturbed predictions.

## Results

The target recording contains 30 windows, originally 2 detected and 28 missed.
The successful external control contains 1,255 quality-passing windows outside
that recording. The table reports newly lost controls relative to identity.

| Transform | Target detection | Recovered / 28 misses | Lost / 1,255 successes | Training empty FPR | Training breathing recall |
|---|---:|---:|---:|---:|---:|
| Identity | 6.67% | 0 | 0 | 1.58% | 99.67% |
| Circular -32 | 100.00% | 28 | 65 | 79.08% | 94.00% |
| Circular -16 | 100.00% | 28 | 39 | 86.42% | 93.08% |
| Circular -4 | 90.00% | 26 | 2 | 64.50% | 96.00% |
| Circular -1 | 96.67% | 27 | 6 | 32.75% | 98.17% |
| Circular +1 | 73.33% | 20 | 7 | 25.00% | 93.42% |
| Circular +4 | 80.00% | 23 | 84 | 42.67% | 89.75% |
| Circular +16 | 76.67% | 21 | 61 | 65.75% | 78.33% |
| Circular +32 | 100.00% | 28 | 38 | 83.92% | 92.25% |
| Reverse | 3.33% | 1 | 134 | 79.25% | 95.25% |
| Edge -4 | 90.00% | 26 | 6 | 64.00% | 95.42% |
| Edge -1 | 76.67% | 21 | 3 | 38.08% | 99.33% |
| Edge +1 | 60.00% | 16 | 5 | 26.17% | 96.50% |
| Edge +4 | 96.67% | 27 | 46 | 49.33% | 96.42% |

Recovered counts need not equal the net increase in detections: a transform
can recover original misses while losing one of the original two successes.

## Interpretation

The frozen classifier is sensitive to even a one-position frequency shift.
This sensitivity persists with edge-replicated shifts, so circular wrapping
does not fully explain it. Position perturbations can recover the target but
also strongly increase breathing scores on training empty rooms. The target's
improvement therefore cannot establish correct physical alignment; part of
the apparent recovery accompanies a general loss of class discrimination.

Reversal fails to recover the target and damages successful controls. These
results do not support a simple reversed input order as the cause. A physical
frequency offset also remains unproven: normalized device coordinates are
not absolute RF frequencies, and these synthetic perturbations are not real
sensor calibration measurements.

## Decision and next work

Keep the deployed representation and frozen model unchanged. None of the
tested global transforms preserves the empty-reference behavior. Do not pick
a per-window transform by maximizing breathing probability: the empty-room
response demonstrates why that would encourage false positives.

The next justified step is a grouped robustness experiment, initially using
CSI-Bench train/validation only: compare the unchanged baseline against a
candidate trained with small frequency perturbations applied to BOTH classes.
Use validation to require retention of empty specificity as well as breathing
recall. If that succeeds, report WiRe-Breath as an already-observed external
development check; a later final generalization claim needs new held-out
recordings. WiRe-Breath alone cannot validate a new binary decision rule.

## Reproduction and artifacts

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.perturb_wire_frequency \
  --output backend/modules/breathing_detection/outputs/\
wire_breath_frozen_seed123/frequency_perturbation.json
```

The ignored JSON includes checkpoint SHA-256, threshold, group-level
recovery/loss counts, all-window assessable results, and training controls.
