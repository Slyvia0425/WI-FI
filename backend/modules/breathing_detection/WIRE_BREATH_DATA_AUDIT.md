# WiRe-Breath data audit

This is a read-only audit of the locally downloaded WiRe-Breath release. Its
purpose is to decide how the dataset can safely extend the breathing module;
it does not assign CSI-Bench labels to WiRe-Breath recordings.

## Integrity and layout

- Local root: `data/WiReBreath` (ignored by Git).
- MATLAB files: 467; all 467 load successfully and contain finite values.
- Every file contains `csi_complex`, `csi_abs`, `time`, `gt_signal`,
  `gt_time`, and metadata.
- CSI shape is consistently `[1500, 234]`; `csi_abs` agrees with
  `abs(csi_complex)` in every file.
- The 467 segments come from 85 distinct recordings. A recording has between
  1 and 10 segments.
- The archive was extracted with Windows path separators stored literally in
  filenames. The audit loader supports this layout, so the source data does
  not need to be renamed or moved.

The detailed machine-readable result is generated as
`outputs/wire_breath_audit.json`; `outputs/` is intentionally ignored by Git.

The implemented timestamp-aware adapter produces 1,401 non-overlapping
10-second windows at 30 Hz: 720 Quiet and 681 Interference. Under the initial
100 ms maximum-gap quality rule, 1,293 windows are assessable and 108 are
retained with a `LARGE_TIMESTAMP_GAP` flag (50 Quiet and 58 Interference).
Keeping flagged windows visible supports later threshold analysis without
silently contaminating the clean training subset.

## Dataset composition

| Dimension | Distribution |
|---|---|
| Room | Room1 170; Room2 297 |
| Environment | Quiet 240; Interference 227 |
| Subject | 01: 158; 02: 151; 03: 76; 04: 25; 05: 57 |
| Posture | Foetus 52; Freefaller 60; Log 50; Normal 20; Semi 60; Sitting 56; Soldier 50; Starfish 60; Yearner 59 |

Every segment contains a participant in a breathing posture and a reference
respiratory waveform. `Quiet` and `Interference` therefore mean breathing
under two acquisition conditions. They are not equivalent to the
CSI-Bench `empty` and `sleep` classes, and `Interference` is not a fan-only or
empty-room negative class.

## Timing and respiratory reference

Although metadata describes a nominal 50 Hz CSI rate, timestamp-based median
rates are 48.73--48.99 Hz. Segment durations are 30.70--36.68 seconds. The
median maximum timestamp gap per segment is 43 ms, and the worst gap is
924 ms. Consequently, preprocessing must resample by the supplied timestamps;
it must not assume that sample index divided by 50 is elapsed time.

A simple FFT audit of the ground-truth waveform gives dominant rates of
7.59--27.27 breaths/minute, with an overall median of 15.16. Quiet and
Interference medians are similar (15.18 and 15.08 breaths/minute), reducing
the risk that an environment classifier could succeed only from different
breathing-rate distributions. These figures are integrity checks, not final
rate-estimation accuracy measurements.

## Suitable project uses

WiRe-Breath is suitable for:

1. measuring whether breathing evidence and rate estimation remain stable
   under the dataset's recorded interference;
2. developing timestamp-aware preprocessing and temporal consistency checks;
3. pretraining or evaluating a respiratory waveform/rate head; and
4. testing generalization across subject, room, posture, and environment.

It is not sufficient by itself to train the current `not_assessable`
periodic-interference gate. That gate needs matched negative recordings where
periodic machinery, a fan, or another interferer is present without a person
breathing in the sensing area. Treating WiRe-Breath `Interference` as that
negative label would teach the gate to reject valid breathing.

## Preprocessing and evaluation decision

The adapter for this dataset should:

- preserve the physical order of all 234 subcarriers;
- use `time` to interpolate CSI amplitude onto a uniform time grid;
- use `gt_time` independently to align the respiratory reference;
- construct 10-second windows only over acceptable timestamp coverage and
  record a quality flag for large gaps;
- retain room, subject, posture, environment, session, trial, segment, and a
  derived recording ID in every example; and
- map 234 ordered subcarriers to the existing 232-position model input only at
  the model boundary, using the same fixed frequency-axis interpolation as the
  established CSI-Bench pipeline.

These adapter requirements are implemented in
`preprocessing/wire_breath.py`. The 100 ms gap threshold is an explicit
initial engineering rule and must be validated in downstream sensitivity
analysis; it is not claimed as a threshold published by the dataset authors.

The prepared local cache occupies approximately 187 MB and stores model-ready
`[1401, 300, 232]` sequences plus aligned `[1401, 300]` respiratory references
as float16 arrays. A separate uint8 weight vector excludes quality-flagged
windows from fitting without deleting them. No CSI-Bench binary label is
stored in this cache.

No random segment-level or window-level split is allowed. All segments and
windows from one of the 85 source recordings must remain in the same split.
The first evaluation should be grouped by recording and reported separately
for Quiet and Interference. Stronger generalization experiments should hold
out complete subjects and, separately, a complete room. With only five
subjects and two rooms, those experiments should be reported as grouped
cross-validation rather than as a single supposedly definitive split.

The generated leave-one-subject-out folds passed an explicit recording-ID
leakage check. Their held-out window counts are 474, 453, 228, 75, and 171 for
Subjects 01--05 respectively. This imbalance, and the fact that Subjects 01,
04, and 05 occur in only one room, means results must be reported per fold and
with room/environment slices rather than only as one pooled score.

## Reproducing the audit

From the repository root:

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.preprocessing.audit_wire_breath \
  --data-root data/WiReBreath \
  --output backend/modules/breathing_detection/outputs/wire_breath_audit.json
```

The command reads the release and writes only the ignored JSON summary. It
does not modify the downloaded files.
