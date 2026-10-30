# WiRe-Breath signal error and timestamp-gap analysis

## Question

This analysis investigates the 28 false-`none` windows from
`Room2/Subject02/Sitting/Quiet/0816/Trial01` and tests whether the initial
100 ms timestamp-gap quality rule should change. The frozen seed-123 model and
its original threshold are unchanged.

Signal features are calculated from timestamp-resampled, unnormalized
WiRe-Breath amplitudes. Ten-second FFT resolution is 0.1 Hz, so respiratory
peak rates are quantized in 6 breaths/minute steps; the values support group
comparison rather than precision respiratory-rate claims.

## Signal comparison

The table reports group medians. `GT agreement` is the fraction of CSI
subcarriers whose breathing-band peak lies within 0.1 Hz of the reference
respiratory peak.

| Feature | Target failed (28) | Same subject Sitting Interference (21, successful) | Other Sitting successful (138) | All successful (1,363) |
|---|---:|---:|---:|---:|
| Reference rate, BPM | 24 | 24 | 18 | 12 |
| Reference waveform std | 318.9 | 294.2 | 399.1 | 271.6 |
| Median CSI temporal variance | 385.2 | 357.2 | 35.7 | 18.9 |
| Median relative CSI variability | 0.890 | 0.551 | 0.381 | 0.384 |
| Median 0.1--0.7 Hz energy ratio | 0.160 | 0.137 | 0.178 | 0.154 |
| Subcarrier peak-frequency MAD, Hz | 0.10 | 0.10 | 0.10 | 0.10 |
| Modal peak fraction | 0.350 | 0.295 | 0.346 | 0.338 |
| GT agreement | 0.517 | 0.504 | 0.611 | 0.611 |
| Median model probability | 0.000000072 | 0.9985 | 0.9998 | 0.9998 |

The failed record does not lack a respiratory reference or periodic CSI
content. Its reference rate and amplitude are normal, its breathing-band
energy is comparable to successful data, and its cross-subcarrier peak
consistency is not degraded. Raw CSI variance is high, but the same
participant/posture under Interference has similarly high variance and is
detected successfully. Relative variability is higher in the failed group,
but its broad range overlaps the successful same-subject comparison.

The model's near-zero probability therefore cannot be explained by the tested
scalar signal-quality or periodicity features. The evidence points instead to
a record-specific domain shift in the ordered 232-position subcarrier pattern
or its temporal evolution. This is an inference, not yet a proven cause. The
two successful windows from the target record are too few to support a stable
within-record comparison.

Lowering the frozen probability threshold is not appropriate: many failed
windows are orders of magnitude below the current threshold, and threshold
changes would affect CSI-Bench empty-room specificity without providing a
matched WiRe-Breath negative class.

## Timestamp-gap distribution

| Room | Windows | Median max gap | P90 | P95 | P99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Room1 | 510 | 41.0 ms | 920.8 ms | 921.6 ms | 922.7 ms | 924.1 ms |
| Room2 | 891 | 41.5 ms | 44.2 ms | 48.2 ms | 61.7 ms | 82.8 ms |

The distribution has a useful separation: normal window-level maximum gaps
end below 83 ms, while the affected Room1 windows contain approximately
0.92-second holes. These are not marginal 100--150 ms cases.

| Allowed maximum gap | Coverage | Conditional recall | Room1 coverage | Room2 coverage |
|---|---:|---:|---:|---:|
| 50 ms | 89.36% | 97.36% | 78.24% | 95.74% |
| 75 ms | 92.15% | 97.21% | 78.82% | 99.78% |
| **100 ms** | **92.29%** | **97.22%** | **78.82%** | **100.00%** |
| 150 ms | 92.29% | 97.22% | 78.82% | 100.00% |
| 250 ms | 92.29% | 97.22% | 78.82% | 100.00% |
| 500 ms | 92.29% | 97.22% | 78.82% | 100.00% |
| 1000 ms | 100.00% | 97.29% | 100.00% | 100.00% |

## Decisions

1. Retain the 100 ms maximum-gap rule. It accepts every Room2 window and all
   normal-gap Room1 windows, while cleanly rejecting the separate 0.92-second
   failure mode. Raising it to 150--500 ms recovers nothing; raising it to one
   second would silently interpolate across severe acquisition holes.
2. Do not use LSTM success on 106 flagged positive windows as justification to
   accept them. Positive-only predictions cannot show whether interpolation
   would cause empty-room false positives or inaccurate respiratory rates.
3. Do not change the frozen classifier threshold based on the failed record.
4. Analyze ordered subcarrier-domain shift next. Appropriate tests include
   per-frequency mean/std profiles, adjacent-subcarrier correlation, PCA or
   embedding distance to CSI-Bench training data, and ablations using
   subcarrier shifts or frequency-axis alignment. Any learned correction must
   use recording/subject-grouped validation.

## Reproduction

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.analyze_wire_breath_failures \
  --data-root data/WiReBreath \
  --cache-dir backend/modules/breathing_detection/outputs/wire_breath_cache \
  --checkpoint backend/modules/breathing_detection/outputs/\
lstm_interpolated_full_seed123/best_model.pt \
  --output backend/modules/breathing_detection/outputs/\
wire_breath_frozen_seed123/signal_error_analysis.json
```
