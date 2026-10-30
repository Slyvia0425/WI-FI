# Continuous windows and disconnection handling

## Implemented locally

`inference/streaming.py` provides a bounded, timestamp-aware raw CSI amplitude
buffer. Defaults retain the frozen LSTM's 10-second detection window, a separate
30-second spectral respiratory-rate window, and a 5-second update interval.
The detector input remains `[300,232]`; model weights and threshold are unchanged.
The current rate uses `mean_spectrum_boundary0_v1` in both streaming and research
sample views. The original validation below used the mean-spectrum baseline and
is retained as history. The subsequent fixed-estimator replay preserves all
detection decisions and rate availability; see
[`UNIFIED_RESPIRATORY_RATE_REPORT.md`](UNIFIED_RESPIRATORY_RATE_REPORT.md).

Sensor timestamps and packet-arrival time are separate clocks. A sample gap
greater than 100 ms invalidates affected windows. No arrival for 2 seconds,
a sensor gap of at least 2 seconds, timestamp regression, or a changed
subcarrier count clears the buffer and temporal history. Invalid batches also
clear current evidence and raise an error. The 2-second watchdog is an
engineering default, not a calibrated physiological threshold; delivery batches
must normally arrive more frequently than this.

After reset, detection needs 10 fresh seconds and rate needs 30 fresh seconds.
Historical results remain available with `last_result_stale=true`, but the
current result becomes `not_assessable` and current rate is hidden. Calling
`poll()` regularly, including when no packets arrive, is required. There is no
device/network server integration in this change. The recorded-data frontend
now consumes exported snapshots/events from this same pipeline; see below.

## Real-data replay validation

Run from the repository root:

```bash
.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_streaming
```

The completed run is saved under `outputs/streaming_validation_v2/`:
`config.json`, `metrics.json`, and `stream_events.json`. The script refuses to
overwrite a configured run. It uses all 467 independent WiRe-Breath segments
from 85 source recordings. Each MAT file starts a new stream; files are never
joined into a purported continuous record. Packets are delivered on an injected
250 ms virtual clock, with watchdog polling even during empty intervals.

| Measure | Result |
|---|---:|
| 5-second sliding-window updates | 2,342 |
| Assessable detection windows | 2,156 (92.06%) |
| Breathing recall among assessable windows | 97.26% |
| False `none` predictions | 59 |
| Label flips / valid adjacent pairs | 17 / 1,681 (1.01%) |
| Experimental rate outputs | 384 |
| Unintended resets during replay | 0 |
| End-of-segment timeout checks passed | 467 / 467 |

The 10-second non-overlapping subset has 1,401 windows, 1,293 assessable,
97.22% breathing recall, and 36 false `none` predictions. Its risk scores still
use the same 5-second tracker history, not an independently configured tracker.

The initial `streaming_validation_v1` used 25 samples per packet. Sparse sample
regions then created eight artificial arrival timeouts. It is retained as a
packet-scheduling sensitivity record, not the final evaluation. Fixed-time
packet delivery in v2 separates sampling quality from simulated transport loss.

## Temporal stability: diagnostic only

The existing five-window tracker becomes available after three valid windows;
quality failures and resets clear it. Of 1,234 scored windows, the existing
diagnostic cutoff of 0.8 flagged one correct breathing prediction and no false
`none` prediction. It therefore provides no demonstrated rejection benefit here
and remains disabled. No threshold was fitted. Subject02 accounts for 47 of the
59 false `none` results but only three label flips: a prediction can be stable
and still wrong. Smoothing is not a demonstrated fix for this domain shift.

These overlapping windows are correlated. All records contain breathing, so
specificity and binary Macro-F1 cannot be measured. Roughly 30-second files do
not validate long-duration stability, clinical pauses, motion labels, or actual
network jitter. Next validation needs genuinely continuous recordings, including
breathing-absent and interference conditions, before promoting a temporal gate.

## Integration contract and tests

```python
from backend.modules.breathing_detection.inference import BreathingStream, StreamConfig
from backend.modules.breathing_detection.inference.frozen_detector import FrozenBreathingDetector

detector = FrozenBreathingDetector(checkpoint_path)
stream = BreathingStream(detector, StreamConfig(breathing_threshold=detector.threshold))
events = stream.push(sensor_timestamps, amplitude_time_by_subcarrier)
current = stream.poll()  # also schedule periodically when no input arrives
```

Production calls use the default monotonic arrival clock. Replay may inject
`arrival_time` and `poll(now)` consistently from a separate monotonic clock.
Do not display the historical `last_result` as current when marked stale.

`tests/test_streaming.py` covers cadence, chunk invariance, bounded buffering,
short sample gaps, long gaps, watchdog expiry without new packets, recovery,
malformed input, timestamp/structure resets, and diagnostic-only instability.
The complete module unittest suite passes 73 tests, including two export-level
checks for real packet-arrival gating and discarded-packet recovery.

## Recorded-data frontend integration

Run `evaluation.export_dashboard` first when rebuilding the research data, then
run `evaluation.export_streaming_dashboard` to attach continuous-window output
for all 42 curated segments. The second exporter invokes `BreathingStream` with
the frozen checkpoint, not frontend-derived probabilities. It preserves research
results while adding normal and simulated-loss timelines. Re-running the first
exporter alone removes the streaming extension, so both commands are required.

The English monitoring view reveals snapshots and events by arrival time,
including warming up, valid/invalid windows, timeout, stale-result hiding and
fresh-data recovery. Simulated mode discards packet batches at 12–15 seconds;
these are explicitly labelled artificial, separate from original sampling gaps.
Pause freezes the replay clock and is not a transport failure. Source, segment
and condition changes reset playback; no segments are concatenated.

Replay lasts 34 seconds: a 30-second observation plus real interpolation-bracket
delivery and watchdog observation. A 30s result is not revealed before its actual
last required packet arrives. Current results expire after 2 seconds without
packets, even at recording end. Available rates are retained only in an explicitly
historical end summary. Both views now use the fixed experimental boundary0
estimator; older fold-selected results are labelled historical. Neither is clinical.

This is **backend-computed offline streaming replay**, not browser-triggered
Python inference, a persistent API session, or a connected device. The bundled
timeline works without a running backend. Five frontend tests cover all 42
segments, no future leakage, overlapping cadence, loss/recovery, end timeout
and missing-schema behavior.
