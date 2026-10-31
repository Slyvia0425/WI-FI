# CSI-Bench breathing / empty-room experience

## Fan interference extension (2026-10-30)

The existing recorded-data workspace now has a separate `CSI-Bench · Fan
interference` entry. Seven available fan-on/off × presence × LOS/NLOS cells
contain 28 real official-test samples: the first assessable ID per device and
cell, selected without prediction-correctness filtering. LOS fan-on empty-room
data are missing and are explicitly described as unavailable, not synthesized.
The default fan scenario is fan-on breathing/LOS. Device and scenario changes
reset playback; English/Chinese switching preserves playback state. Separate
on/off files are not presented as paired experiments.

The export reads frozen `fan_detection_v1` predictions and verifies selected raw
file hashes, rather than running a new model or deriving predictions from labels.
Waveforms are original mean CSI amplitude standardized for display, not recovered
respiration. Results remain hidden until a complete nominal ten-second sample;
no BPM is produced. Full 4,156-sample metrics are in a collapsed explanation,
not shown as the current recording's confidence. All old examples, including
the existing false positive, and WiRe streaming/rate content remain intact.

Regenerate this additive extension **last**, after any earlier replay/CSI export:

```bash
.venv/bin/python -m backend.modules.breathing_detection.evaluation.export_fan_dashboard
```

Backend and bundled replay JSON are updated together. This intentionally changes
replay hashes previously protected by offline experiments; those experiment
configs are immutable historical provenance, not rewritten to conceal the UI
update. Model, evaluation outcomes and unrelated lock-file changes are unchanged.
Tests: 169 backend / 20 frontend pass; production build passes (bundle-size
warning remains). Safari accessibility and visual checks verify the entry,
Chinese/English copy, playback, CSI labeling and collapsed metrics/missing class.

The monitoring frontend now offers a separate dataset picker. WiRe-Breath keeps
its 42-segment signal/rate experience; CSI-Bench adds two detection scenarios,
breathing-present (`sleep` label) and empty room (`empty` label), each with three
independent official `test_id` samples. No samples are concatenated or repeated
to construct a 30-second window. This does not change any benchmark, checkpoint,
threshold, training data or model selection.

## Source and selection

`evaluation/export_csi_dashboard.py` reads the actual local HDF5 files through
`BreathingDataset(normalize=False)`. It sorts the official test IDs and chooses
the first three assessable IDs in each class before invoking the frozen seed123
detector with ordered interpolated `[300,232]` inputs. It never filters examples
by prediction score or correctness. IDs, original labels, metadata, threshold,
checkpoint SHA256 and the selection/timing policy are bundled with the results.

The selected breathing examples are all detected; two empty examples have no
breathing evidence, while the third empty example is a false positive (score
approximately 0.447822 above threshold 0.054542). The UI deliberately preserves
this error and labels the disagreement. Six examples are illustrative, not a
representative performance estimate or a new untouched holdout. The official
test was already observed during the frozen evaluation; no refitting is done.

## Timing and interpretation

Each sample contains 300 stored points at nominal 30 Hz. Packet timestamps are
not available through this dataset loader. The interface reveals the original
points over a nominal 10-second observation and shows the frozen prediction only
after sample completion. It does not claim verified packet continuity, run the
arrival watchdog, or simulate transport failure for these pre-windowed samples.
Normal completion retains an explicitly historical result, not a live reading.

Waveforms show independently normalized mean CSI amplitude, not reconstructed
respiration. Empty-room noise can fluctuate after normalization, so wave height
is not an occupancy test. Ground truth is displayed separately from model output.
`none` means no detected breathing evidence, not proof of an empty room. There is
no 30-second rate estimate or respiratory reference in this view. Both English
and the temporary Chinese acceptance copy are supported. Source/scenario/sample
changes reset playback; language switching does not.

## Regeneration and checks

Run exporters in this order from the repository root when rebuilding everything:
The fixed unified rate evaluation (`evaluation.evaluate_unified_rate`) must
already exist; see `UNIFIED_RESPIRATORY_RATE_REPORT.md`.

```bash
.venv/bin/python -m backend.modules.breathing_detection.evaluation.export_dashboard
.venv/bin/python -m backend.modules.breathing_detection.evaluation.export_streaming_dashboard
.venv/bin/python -m backend.modules.breathing_detection.evaluation.export_csi_dashboard
```

The CSI exporter can run alone to refresh only its extension and preserve the
other dashboard content. Results remain bundled, so a live backend is not needed.
Tests cover deterministic quality-only selection, class provenance, no early
results, sample dimensions, reset semantics and preservation of the false positive.
