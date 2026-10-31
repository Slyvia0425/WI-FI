# Breathing detection (backend)

For the included complete WiRe-Breath data, the CSI-Bench download link, and
English setup and usage instructions, see
[`datasets/breathing_detection/README.md`](../../../datasets/breathing_detection/README.md).

Output: breathing evidence or "not assessable".

Everything for this module on the backend lives in this folder. Work on your own branch and edit only here.

| Folder | Put here |
|---|---|
| `preprocessing/` | Data reading, quality checks, splits and features. |
| `training/` | Model definitions and training scripts. |
| `experiments/` | Model comparison, ablations and threshold studies. |
| `evaluation/` | Metrics, confusion matrices, error analysis, and the export of outputs/replay.json. |
| `outputs/` | Generated files: `replay.json` (read by the server), checkpoints, figures. Large files stay out of git. |
| `api.py` | What the server returns for this module |
| `contract.py` | The frame fields this module promises |
| `requirements.txt` | Optional, packages only this module needs |

Frontend counterpart: `frontend/src/modules/breathing_detection/`.
Export a result for the website with `backend.shared.export.write_replay("breathing_detection", payload, bundle=True)`. One call updates the backend output and the copy bundled with the website.
Develop alone with `python -m backend.api --only breathing_detection`.

## Dataset preprocessing

Install this module's dependencies into the project environment:

```bash
.venv/bin/python -m pip install -r backend/modules/breathing_detection/requirements.txt
```

Load an official split lazily (one HDF5 file is read per indexed sample):

```python
from backend.modules.breathing_detection.preprocessing import BreathingDataset

train = BreathingDataset("data/BreathingDetection", "train_id")
sample = train[0]
# project_pipeline output: [channel=1, subcarriers=K, time=300]
```

The default `project_pipeline` keeps the real 300-point window and variable
subcarrier count. Use `mode="official_reproduction"` only for a benchmark
comparison; it reproduces the authors' generic `[1, 500, 232]` padding and
clipping rule. Both modes use the downloaded official split files and
per-sample z-score normalization.

The separately downloaded WiRe-Breath data has also passed a complete
read-only integrity and timing audit. It contains breathing in both Quiet and
Interference conditions, not empty-room negatives, so it will be used for
interference robustness and respiratory-rate work rather than relabelled for
the binary CSI-Bench task. See
[`WIRE_BREATH_DATA_AUDIT.md`](WIRE_BREATH_DATA_AUDIT.md).

Load timestamp-aligned WiRe-Breath windows lazily with:

```python
from backend.modules.breathing_detection.preprocessing import WireBreathDataset

wire = WireBreathDataset("data/WiReBreath")
sample = wire[0]
# x: [channel=1, subcarriers=234, time=300]
# respiratory_reference: [time=300]
```

The default output is a normalized 10-second window at 30 Hz. Windows with a
source timestamp gap greater than 100 ms are retained but marked
`assessable=False`. Use `recording_ids` to apply grouped splits; never divide
segments or windows from the same recording across train and validation.

Prepare the float16 `[300,232]` model-input cache, aligned respiratory
references, quality weights, and leave-one-subject-out fold manifest with:

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.preprocessing.prepare_wire_breath \
  --data-root data/WiReBreath
```

This cache deliberately has no CSI-Bench binary labels. Flagged windows remain
available with zero training weight, and the fold manifest keeps every held-out
participant wholly outside its training partition.

The frozen seed-123 CSI-Bench LSTM reaches 97.22% positive recall on assessable
WiRe-Breath windows, but 28 of its 36 false-`none` decisions are concentrated
in one Subject02 Sitting recording. See
[`WIRE_BREATH_FROZEN_EVAL_REPORT.md`](WIRE_BREATH_FROZEN_EVAL_REPORT.md) for
the positive-only evaluation limits and required error analysis.

Signal-level comparison shows that the concentrated Sitting failure retains
normal respiratory reference and periodic CSI evidence, suggesting an ordered
subcarrier-domain shift rather than missing breathing. The Room1 gap analysis
supports keeping the 100 ms quality threshold. See
[`WIRE_BREATH_SIGNAL_ERROR_ANALYSIS.md`](WIRE_BREATH_SIGNAL_ERROR_ANALYSIS.md).

Frequency-profile and frozen-model embedding analysis supports a record-level
representation shift, while literal frequency-axis misalignment remains
unproven. See
[`WIRE_BREATH_DOMAIN_SHIFT_REPORT.md`](WIRE_BREATH_DOMAIN_SHIFT_REPORT.md).

Controlled circular and edge-filled frequency shifts confirm sensitivity to
input position, but improvements on the failed record accompany large
empty-room false-positive increases. No transform is promoted. See
[`WIRE_BREATH_FREQUENCY_PERTURBATION_REPORT.md`](WIRE_BREATH_FREQUENCY_PERTURBATION_REPORT.md).

A paired two-epoch fine-tuning pilot improved synthetic frequency-shift
validation Macro-F1 but reduced external WiRe-Breath recall. Neither candidate
replaces the frozen baseline; see
[`FREQUENCY_ROBUSTNESS_PILOT_REPORT.md`](FREQUENCY_ROBUSTNESS_PILOT_REPORT.md).

New WiRe-Breath development must use the fixed nested person/recording splits
in `evaluation/protocols/wire_breath_development_v1.json` rather than creating
new random window splits. See
[`WIRE_BREATH_DEVELOPMENT_PROTOCOL.md`](WIRE_BREATH_DEVELOPMENT_PROTOCOL.md)
for the inner selection, outer assessment, binary-label limitations, and
required reporting rules.

The initial respiratory-rate baseline compares three inspectable spectral
estimators at 10 and 30 seconds under this protocol. All five inner folds
select the 30-second mean normalized spectrum. Person-averaged MAE is
1.51 BPM with 84.37% pooled quality coverage, measured against rates derived
from the reference respiratory waveform. See
[`RESPIRATORY_RATE_BASELINE_REPORT.md`](RESPIRATORY_RATE_BASELINE_REPORT.md)
for shorter-window comparisons, interference errors, and reference limitations.

Large-rate-error analysis finds weaker, competing spectral peaks and frequent
near-half-frequency estimates under interference. A validation-selected peak
concentration screen reduces error but rejects too many good estimates on
held-out people, so it remains disabled. See
[`RESPIRATORY_RATE_ERROR_ANALYSIS.md`](RESPIRATORY_RATE_ERROR_ANALYSIS.md).

Paired waveform review finds 23/35 large errors selecting the exact lower
search endpoint, while 34/35 reference peak-interval checks support the
original spectral reference. Automatic doubling is not justified. See
[`RESPIRATORY_RATE_WAVEFORM_REVIEW.md`](RESPIRATORY_RATE_WAVEFORM_REVIEW.md)
and the ignored `outputs/respiratory_rate_baseline_v1/waveform_review/index.html`.

The local-peak experiment reduces pooled rate MAE from 1.162 to 0.935 BPM at
unchanged coverage, but highest-prominence ranking creates severe Subject05
errors. It remains experimental; see
[`RESPIRATORY_RATE_LOCAL_PEAK_REPORT.md`](RESPIRATORY_RATE_LOCAL_PEAK_REPORT.md).

The conservative boundary-only follow-up reduces pooled MAE to 0.823 BPM
with unchanged coverage and no baseline-good estimates spoiled, though 22
large errors remain. It stays experimental; see
[`RESPIRATORY_RATE_BOUNDARY_CORRECTION_REPORT.md`](RESPIRATORY_RATE_BOUNDARY_CORRECTION_REPORT.md).

The subsequent rate-reliability screen withholds eight windows and catches
three of 22 remaining large errors, but exceeds the per-person good-rejection
budget on Subject05. It remains disabled; see
[`RESPIRATORY_RATE_RELIABILITY_REPORT.md`](RESPIRATORY_RATE_RELIABILITY_REPORT.md).

Run this module's tests with:

```bash
.venv/bin/python -m unittest discover \
  -s backend/modules/breathing_detection/tests -t . -v
```

## Interpretable periodicity baseline

The first baseline selects high-variance subcarriers and extracts 18 fixed
features covering breathing-band FFT energy and peaks, spectral entropy,
autocorrelation peaks, cross-subcarrier peak consistency, waveform agreement,
and a high-frequency motion proxy. A small balanced logistic regression keeps
the decision inspectable through its saved coefficients.

Run a short smoke experiment:

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.training.train_periodicity_baseline \
  --data-root data/BreathingDetection --max-train 400 --max-val 200
```

Omit the limits for the full train/validation run and add `--evaluate-test`
for the final test pass. The decision threshold is
selected on validation macro-F1, not the test set. Outputs are written under
the module's ignored `outputs/` directory. Small smoke-run scores are pipeline
checks only and must not be reported as final results.

The completed full-run analysis is recorded in
[`PERIODICITY_BASELINE_REPORT.md`](PERIODICITY_BASELINE_REPORT.md).

## Lightweight LSTM

The LSTM pipeline and its validation-only early stopping are implemented in
`training/train_lstm.py`. Cross-subcarrier statistics and per-sample Top-32
selection were rejected in controlled pilots. Preserving subcarrier order and
interpolating to a fixed 232-position frequency axis passed the promotion
criteria (validation macro-F1 0.8187). See
[`LSTM_PILOT_REPORT.md`](LSTM_PILOT_REPORT.md). Pilot scores are not final
full-split results.

Three complete-split seeds confirmed stable validation performance. The frozen
seed 123 candidate reached test macro-F1 0.9670 and an empty-room
false-positive rate of 0.0673. See
[`LSTM_FULL_RUN_REPORT.md`](LSTM_FULL_RUN_REPORT.md) for the full configuration,
confusion matrices, and subcarrier-count analysis.
The concentrated K=208 false positives are investigated in
[`K208_ERROR_ANALYSIS.md`](K208_ERROR_ANALYSIS.md).

### Interpolated sequence cache

Full float32 `[300,232]` arrays would use about 26 GB across all splits. The
cache stores float16 memory-mapped arrays instead (about 13 GB total) and
converts each batch back to float32 during training. It records progress every
250 samples and resumes a compatible partial build.

Build/resume train and validation caches while starting training:

```bash
.venv/bin/python -m backend.modules.breathing_detection.training.train_lstm \
  --data-root data/BreathingDetection \
  --representation interpolated \
  --cache-dir backend/modules/breathing_detection/outputs/temporal_cache \
  --build-cache
```

On later training runs, omit `--build-cache`; the completed memory-mapped cache
is loaded directly. Do not use the training script to reevaluate the frozen
model because it would train and overwrite a new checkpoint. The completed
one-way test evaluation used the read-only entry point:

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.evaluate_lstm \
  --data-root data/BreathingDetection \
  --checkpoint backend/modules/breathing_detection/outputs/\
lstm_interpolated_full_seed123/best_model.pt \
  --cache-dir backend/modules/breathing_detection/outputs/temporal_cache \
  --output backend/modules/breathing_detection/outputs/\
lstm_interpolated_full_seed123/test_metrics.json
```

The official test result has already been observed. It must not be reused for
threshold selection, seed selection, or development of later interference
gates.

## Assessability gates

`inference/gating.py` implements the ordered decision layer around the frozen
model: low quality, motion, Fan evidence, calibrated periodic interference,
and temporal instability take precedence over `breathing` or `none`. Optional
external evidence remains unavailable when omitted; it is not silently treated
as zero risk. The current implementation and the remaining validation-only
calibration work are described in
[`GATING_DESIGN.md`](GATING_DESIGN.md).

The first signal-only periodic-risk experiment did not pass the promotion
criterion: it rejected 116 correct breathing windows to remove six validation
false positives. It remains disabled; see
[`PERIODIC_GATE_REPORT.md`](PERIODIC_GATE_REPORT.md).

## Offline research dashboard

The default module frontend is a monitoring workspace with two sources:
recorded-data experience and a disabled future device connection. Users choose
a real record/segment, start, pause or resume, and read completed-window results.
The first detection appears at 10 seconds; an experimental rate may appear at
30 seconds. Switching segments resets observation. Rates and future evidence
are never exposed early, and real timestamp gaps are revealed as their position
is reached. No device input, file upload, or continuous cross-segment timeline
is implied. Playback can be accelerated explicitly and pauses in hidden tabs.

The research notebook and original full-segment explorer are retained behind
the secondary "模型与验证" view. Scoped styles live in
`frontend/src/modules/breathing_detection/breathing.css` and `monitor.css`.
It includes five curated recordings (42 independent segments), real frozen
LSTM scores, timestamp-quality flags and unified fixed experimental rates.
Current rate metrics are retrospective development results; historical
fold-selected experiments remain separately labelled.
Waveforms are standardized CSI amplitude summaries, not reconstructed breaths.
The optional respiratory sensor trace is evaluation-only. Reliability screening
remains disabled; browsing coordinates do not imply synchronized continuous
monitoring or cross-module ground truth.

The workspace adds current evidence, rate and sampling status, next-step
guidance, event history and an end-of-segment summary. A single 30-second rate
is not displayed as a long-term trend. Reference traces remain in optional
signal details and only their already-read portion is revealed.

Verify playback-state logic using `node --test
src/modules/breathing_detection/session.test.js` from `frontend/`.

Regenerate backend and bundled frontend data using existing local artifacts:

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.evaluation.export_dashboard
```

Run `npm run dev` from `frontend/`, then open
`http://localhost:5173/#/module/breathing_detection`. The bundled dataset works
without the backend. Rate display is withheld on timestamp failure or when
the final 10-second classifier window does not support breathing. Original
offline estimates remain visible as diagnostic details, not reliable outputs.
Full-set benchmark cards must not be interpreted as metrics of the curated
sample collection. Data attribution and transformation notes appear on-page.

## Data coverage and external validation

The frozen fan-presence subgroup evaluation is complete: all 4,156 relevant
official-test samples were scored without retraining or threshold changes.
Fan-on breathing recall is LOS 588/589 (99.83%) and NLOS 593/597 (99.33%);
NLOS fan-on empty false positives are 1/612 (0.16%). LOS fan-on has no empty
examples, so its FPR is unknown, not zero. These are previously observed U23/E09
in-domain samples, not independent external validation or BPM accuracy evidence.
See [`FAN_DETECTION_REPORT.md`](FAN_DETECTION_REPORT.md) and
[`frozen protocol`](FAN_DETECTION_PROTOCOL.md). No model/UI/gate changes.

The local metadata inventory and primary-source candidate search are documented
in [`DATA_GAP_AND_ACQUISITION_PLAN.md`](DATA_GAP_AND_ACQUISITION_PLAN.md).
Local CSI-Bench already includes 3,884 `empty + NLOS_FanOn` samples, but these are
development samples, not independent new recordings. Eight HDF5 schemas were
spot-checked; local release version remains unverified. External presence/machine
stress data and synchronized-reference respiratory-rate data have separate
acceptance criteria. No new download, author contact, fitting or model/UI change
is performed by this inventory.

## Continuous-window backend

The complex-CSI phase feasibility audit is complete and **not adopted**. Genuine
complex64 data exist, but intermittent zero coefficients and large temporal phase
changes limit simple processing. Raw/adjacent/index-affine diagnostic phase
readouts have MAE 4.812/6.957/7.339 BPM versus amplitude 0.683, harming 262/245/302
originally accurate outputs. All five grouped validations choose amplitude.
Index-affine projection is not verified physical calibration; no antenna map or
physical carrier-frequency coordinates are assumed. No model/UI/fusion changes.
See [`PHASE_FEASIBILITY_REPORT.md`](PHASE_FEASIBILITY_REPORT.md) and its
[`fixed protocol`](PHASE_FEASIBILITY_PROTOCOL.md).

The three-route consistency audit is complete and **stops before adding a fusion
rule**. All six successful relative corrections and all five harmed accurate
outputs have original-spectrum support for the proposed frequency. A successful
double-rate correction and a harmful half-rate change share common/relative
agreement and strong secondary-peak evidence, so these cues alone do not justify
a safe automatic chooser. No new fusion score or production change is claimed.
See [`THREE_ROUTE_REPORT.md`](THREE_ROUTE_REPORT.md) and its
[`audit protocol`](THREE_ROUTE_PROTOCOL.md).

The common-versus-relative representation pilot is complete and **not adopted**.
Relative-only lowers pooled MAE 0.683→0.625 BPM and large errors 19→17, but harms
5/365 accurate outputs and worsens recording-macro MAE 0.702→1.072. Grouped
selection also harms five accurate outputs and fails support. Both common and
relative dynamics can retain useful respiratory components; do not label the
common trace as interference or remove it globally. The original estimator and
frontend remain unchanged. See [`COMMON_MODE_REPORT.md`](COMMON_MODE_REPORT.md)
and its [`fixed protocol`](COMMON_MODE_PROTOCOL.md).

The subsequent candidate-peak audit and one frozen carrier-support correction
experiment are complete and **not adopted**. Of the previous preprocessing
experiment's 18 remaining errors (including its harmed accurate case), 16 have
a reference-near local peak before cleaning and 18 after cleaning; this does not
make the correct peak identifiable without reference. The fixed support rule
worsens MAE 0.683→1.344 BPM and large errors 19→36, harming 16 accurate outputs.
All five inner validations select the unchanged baseline; no application output
changes. See [`PEAK_EVIDENCE_REPORT.md`](PEAK_EVIDENCE_REPORT.md),
[`audit protocol`](PEAK_EVIDENCE_PROTOCOL.md) and
[`single candidate protocol`](PEAK_SUPPORT_CANDIDATE_PROTOCOL.md).

The subsequent raw-spike audit and fixed isolated-spike preprocessing comparison
is complete and **not adopted**. Grouped MAE improves 0.683→0.581 BPM, but large
errors only fall 19→18 and one accurate result is spoiled. Raw complex/amplitude
and independent resampling checks pass. A descriptive source audit confirms
exact overlap in 346/382 adjacent within-record file pairs; these files must not
be considered statistically independent. Existing whole-record grouping already
keeps confirmed overlap in the same partition. Operational per-file stream resets
remain unchanged. See [`SPIKE_PREPROCESSING_REPORT.md`](SPIKE_PREPROCESSING_REPORT.md)
and its [`fixed protocol`](SPIKE_PREPROCESSING_PROTOCOL.md).

The fixed unified estimator's subsequent motion-related pollution experiment is
complete and **not deployed**. Four CSI-only proxies, train-calibrated grouped
rules and same-record controls did not identify large errors safely: selected
rules intercepted 0/19 and withheld 6/365 accurate outputs. The estimator,
frontend and gates remain unchanged. See
[`MOTION_POLLUTION_REPORT.md`](MOTION_POLLUTION_REPORT.md) and its
[`pre-run protocol`](MOTION_POLLUTION_PROTOCOL.md).

**Current rate implementation:** `preprocessing/unified_rate.py` fixes
`mean_spectrum_boundary0_v1` for every participant/recording. Streaming and
research sample values now agree; no fold-dependent method is used at runtime.
Full recomputation gives retrospective MAE 0.683 BPM and 19 large errors, not
an independent-test score. Reliability gates remain disabled. See
[`UNIFIED_RESPIRATORY_RATE_REPORT.md`](UNIFIED_RESPIRATORY_RATE_REPORT.md) for the
fixed rule, remaining failures, artifacts and regeneration order.

Stage-2 respiratory-rate reliability development is complete: three within-file
20s subwindows and half/double spectral competition did not support safe useful
withholding. All five nested folds select no gate; 394 corrected estimates and
22 large errors remain unchanged. The frontend and application algorithms are
not changed. See
[`RESPIRATORY_RATE_RELIABILITY_STAGE2_REPORT.md`](RESPIRATORY_RATE_RELIABILITY_STAGE2_REPORT.md)
and the pre-run
[`protocol`](RESPIRATORY_RATE_RELIABILITY_STAGE2_PROTOCOL.md).

The monitoring view also includes a separate **CSI-Bench · Breathing / empty room**
experience: three real official-test examples per class, independent nominal
10-second samples, frozen-model predictions and separately labelled ground truth.
The deterministic selection includes a real empty-room false positive; it is not
hidden or relabelled. No rate or continuous packet-timing claim is made for these
pre-windowed samples. See [`CSI_BENCH_EXPERIENCE.md`](CSI_BENCH_EXPERIENCE.md).

`inference/streaming.py` now supports 10-second detection and 30-second
experimental rate windows with 5-second updates, bounded buffering, timestamp
quality checks, and a separate arrival watchdog. Disconnection hides stale
outputs and recovery requires fresh windows. All 467 independent WiRe-Breath
segments have been replayed without concatenating files. Temporal risk remains
diagnostic only; no gate was promoted. The recorded-data frontend now consumes
its exported streaming timelines, including an explicitly simulated packet-loss
condition and end-of-input timeout. A real device is not connected. Regenerate
the streaming extension after the base dashboard export:

```bash
.venv/bin/python -m backend.modules.breathing_detection.evaluation.export_streaming_dashboard
```

The monitor uses 10s/30s windows with 5s updates and the same fixed experimental
rate estimator as the current research sample values. See
[`STREAMING_PIPELINE_REPORT.md`](STREAMING_PIPELINE_REPORT.md) for metrics,
integration requirements and limitations.

## Frame fields

Defined in `contract.py`, besides `t` (seconds).

| Field | Meaning |
|---|---|
| `status` | breathing, none or not_assessable |
| `confidence` | 0 to 1 |
