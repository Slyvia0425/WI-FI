# Candidate-peak evidence and one supported-competitor experiment

## Decision

Completed all three stages: inspect unresolved cases, distinguish candidate
selection from absent/ambiguous evidence, and evaluate **one** frozen CSI-only
candidate. **Do not enable the new rule or spike preprocessing.** The existing
`mean_spectrum_boundary0_v1` rate estimator, LSTM, quality gates, replay and
frontend remain unchanged. No rejection gate or physical interference classifier
has been added, and nothing is committed/pushed.

The audit supports investigating competing CSI components rather than assuming
all failures lack respiratory-frequency energy. It does **not** provide a safe
way to identify the correct component without the reference. A simple
carrier-supported competitor rule fails badly: when fixed across all 394 valid
windows, MAE increases from 0.683 to 1.344 BPM and large errors from 19 to 36.
All five inner validations choose the unchanged baseline fallback.

## Scope and frozen procedures

Stage 1 follows [PEAK_EVIDENCE_PROTOCOL.md](PEAK_EVIDENCE_PROTOCOL.md), written
before feature extraction. All 394 valid 30-second windows are audited; the
73 timestamp-quality exclusions stay excluded. Two CSI representations are
compared: original input and raw-before-resampling Hampel-8. The latter is a
diagnostic contrast, not a newly adopted preprocessing pipeline.

Target cases are the union of the previous preprocessing experiment's 18
fold-selected >5 BPM errors and any originally accurate output it harmed. The
harmed output is already one of those 18, so there are **18, not 19, plots**.
This target set differs from the unchanged production-reference estimator's
19 large errors. Previous fold-selected preprocessing MAE 0.581 is not the
current production-reference MAE; the current baseline remains 0.683.

Every genuine mean-spectrum local peak in 0.1–0.7 Hz is retained using the
0.05–1 Hz context. Save height, prominence and the fraction of per-carrier
dominant frequencies within 2 BPM. The 2 BPM tolerance is the existing accuracy
definition, not extra FFT-padding resolution. Subcarriers are correlated, so
support is not a count of independent observations or a confidence probability.

Reference-only diagnostics are a local peak within 2 BPM, secondary/primary peak
height ratio, and the two 15-second reference spectral rates. A second/first
ratio >=0.5 and half-window disagreement >2 BPM are descriptive warnings, not
exclusion criteria or replacement ground truth. Half-window estimates have
limited frequency resolution. No label is modified, no reference-based candidate
selection is implemented, and no target is removed after inspection.

Stage 2 chose one correction hypothesis, not a rejection gate. Its rationale,
threshold and selection are frozen in
[PEAK_SUPPORT_CANDIDATE_PROTOCOL.md](PEAK_SUPPORT_CANDIDATE_PROTOCOL.md) before
computing its outcomes. Stage 3 reuses whole-record leave-person-out splits,
with no common recording in train/validation/assessment. The previous source
overlap audit remains relevant: segments are not independent experiments.
These are previously observed development data, not an independent final test.

## Stage 1: what the remaining errors actually contain

| Evidence among 18 targets | Original CSI | Raw Hampel-8 CSI |
|---|---:|---:|
| Local mean-spectrum peak within 2 BPM of reference | 16 | 18 |
| No such local peak | 2 | 0 |
| Reference-near candidate support greater than that of the selected rate | 2 | 1 |

Presence does not imply dominance, separability or an identifiable respiratory
source. Original reference-near peak heights range 0.360–1.000 of allowed-band
maximum, and support ranges 5.6–33.8%. After cleaning, heights range 0.228–0.998,
and support ranges 1.3–33.3%. A tiny newly visible local peak is not sufficient
evidence for an automatic correction. No reference-near target candidate has
majority carrier support. One incorrect low-frequency peak does exceed 50%
support (Subject05 Log Interference segment005).

The two original targets without a reference-near local maximum are Subject02
Log Interference segment007 and Subject05 Log Interference segment005. Cleaning
creates/reveals such a peak in both but still selects an incorrect maximum.
For the separate set of **19 original-baseline** large errors, 18 have a
reference-near peak both before and after cleaning. These descriptive counts
cannot be interpreted as an oracle-corrected model performance.

### Reference ambiguity is not the dominant observed explanation

None of the 18 reference spectra has a second local peak at least half the
primary height. Four have >2 BPM half-window disagreement, all 2.637 BPM:
Subject01 Semi segment009, Subject03 Yearner segment002, Subject05 Soldier
segment001 and Subject05 Yearner segment002. Finite duration, frequency
quantization and genuine rate variation can contribute. This does not establish
reference failure, nor eliminate all possible sensor artifacts. The reference
waveforms/labels are preserved.

Original CSI target predictions have six numerical near-half relations and one
near-double relation to their references; the cleaned representations have six
and four respectively. These are overlapping diagnostic relations with a
2 BPM tolerance in reference-frequency units, not physical harmonic labels.
Low-frequency competition can look like half-rate error without a genuine
subharmonic mechanism. No blind halving/doubling rule is justified.

### Why cleaning can make a correct estimate wrong

Subject05 Yearner Interference segment003 originally selects 11.426 BPM,
matching the reference. Cleaning moves the winner to 17.139 BPM, while the
correct-frequency local peak remains at 11.426 with height 0.960 of the cleaned
allowed maximum. This is a **rank reversal**, not disappearance of all useful
signal. Carrier support is 13.2% at the correct peak and 15.4% at the cleaned
winner; voting alone would not repair it. The reference waveform has a clear
dominant spectral component and no predefined competing-peak warning.

### Complete target index

Rates below are reference / original / raw-Hampel-8, in BPM. These per-case
diagnostics are not new outcomes from the frozen candidate experiment.

| Subject / posture / segment (all Interference) | Rates | Figure |
|---|---|---|
| 01 Semi 001 | 19.78 / 6.15 / 6.15 | [01](outputs/peak_evidence_v1/case_01.png) |
| 01 Semi 009 | 18.46 / 7.47 / 7.91 | [02](outputs/peak_evidence_v1/case_02.png) |
| 01 Semi 010 | 18.46 / 8.35 / 8.35 | [03](outputs/peak_evidence_v1/case_03.png) |
| 02 Log 007 | 11.43 / 13.62 / 22.85 | [04](outputs/peak_evidence_v1/case_04.png) |
| 02 Log 010 | 11.43 / 6.15 / 6.15 | [05](outputs/peak_evidence_v1/case_05.png) |
| 02 Yearner 001 | 11.87 / 6.59 / 6.59 | [06](outputs/peak_evidence_v1/case_06.png) |
| 02 Yearner 002 | 12.30 / 6.15 / 6.15 | [07](outputs/peak_evidence_v1/case_07.png) |
| 03 Log 001 | 13.18 / 7.91 / 7.91 | [08](outputs/peak_evidence_v1/case_08.png) |
| 03 Yearner 002 | 12.74 / 19.78 / 23.73 | [09](outputs/peak_evidence_v1/case_09.png) |
| 03 Yearner 003 | 13.18 / 20.21 / 24.61 | [10](outputs/peak_evidence_v1/case_10.png) |
| 04 Soldier 001 | 21.09 / 6.15 / 6.59 | [11](outputs/peak_evidence_v1/case_11.png) |
| 05 Log 003 | 12.30 / 24.61 / 31.20 | [12](outputs/peak_evidence_v1/case_12.png) |
| 05 Log 004 | 12.30 / 6.15 / 6.59 | [13](outputs/peak_evidence_v1/case_13.png) |
| 05 Log 005 | 12.74 / 7.91 / 7.47 | [14](outputs/peak_evidence_v1/case_14.png) |
| 05 Soldier 001 | 14.06 / 7.47 / 7.47 | [15](outputs/peak_evidence_v1/case_15.png) |
| 05 Soldier 002 | 13.18 / 7.91 / 7.91 | [16](outputs/peak_evidence_v1/case_16.png) |
| 05 Yearner 002 | 10.99 / 28.12 / 18.90 | [17](outputs/peak_evidence_v1/case_17.png) |
| 05 Yearner 003 (cleaning harm) | 11.43 / 11.43 / 17.14 | [18](outputs/peak_evidence_v1/case_18.png) |

All 18 figures include both CSI spectra, dominant-carrier frequency distributions,
reference waveform and reference spectrum. Cases 01, 14 and 18 were also visually
inspected. Neither the Interference condition nor the spectra identify a fan,
specific movement, hardware gain effect or periodic external source.

## Stages 2–3: test one relative-support correction

`supported_competitor_v1` uses original CSI only, no cleaning. Among genuine
local peaks with height >=0.5 of the allowed-band maximum, choose greatest
carrier support (height/lower frequency break ties). Change only for support
strictly greater than current rate support and rate distance >2 BPM. Otherwise
preserve the current estimate. The implementation accepts only CSI evidence and
current BPM: no reference, identity, posture or environment. This tests whether
relative support can fix a subset, not whether support certifies physiology.

Exactly two inner choices are allowed: this rule and unchanged baseline. Choose
lower record-macro MAE subject to full coverage and zero harm to baseline-good
outputs. Good means original error <=2 BPM, denominator **365**. Baseline wins
ties. Support further requires >=25% fewer >5 BPM errors and no person's MAE
worsening. No new threshold or second candidate is tried after the outcome.

| Measure | Original baseline | Fixed new rule | Nested selected |
|---|---:|---:|---:|
| Retained valid windows | 394 | 394 | 394 |
| MAE, BPM | 0.683 | 1.344 | 0.683 |
| RMSE, BPM | 2.466 | 4.510 | 2.466 |
| Recording-macro MAE, BPM | 0.702 | 1.916 | 0.702 |
| >5 BPM errors | 19 | 36 | 19 |
| Accurate outputs harmed /365 | 0 | 16 | 0 |
| Original large errors corrected to <=2 BPM | 0 | 2 | 0 |
| Changed windows | 0 | 26 | 0 |

The fixed rule creates 19 new large errors while correcting two original large
errors, hence 19−2+19=36 remaining large errors. This is not an acceptable trade.
It can jump to strong but incorrect high-frequency or half-rate components.
Every fold chooses baseline. Four fold validations reject the candidate for
accurate-output harm; the subject05 fold has no validation harm but ties baseline
record-macro MAE (0.792), so the unchanged baseline wins. The nested result therefore safely
preserves outputs, but has **zero benefit**: `development_support_pass=false`.
`outer_safety_pass=true` is only the no-change fallback, not a successful rule.

| Held-out subject | Original MAE | Fixed rule MAE | Fixed accurate harm |
|---|---:|---:|---:|
| 01 | 0.491 | 0.491 | 0 |
| 02 | 0.418 | 0.602 | 4 |
| 03 | 0.727 | 1.506 | 2 |
| 04 | 0.879 | 1.441 | 1 |
| 05 | 1.519 | 4.356 | 9 |

The 394/467 total coverage remains 84.37%; no good-quality rate is withheld and
no bad timestamp window is upgraded. The unchanged 100 ms quality rule is not
implicated or retuned by this frequency-selection experiment.

## What should happen next

Freeze this negative result. It rules out this particular supported-competitor
rule, not all multichannel methods. Current evidence does not support another
amplitude threshold sweep, enabling cleaning globally, or replacing a BPM with
the nearest reference peak. It also does not justify a reliable/unreliable gate
based on vote count alone.

A meaningful new experiment would need **different signal evidence**, such as
separating cross-carrier common-mode changes from relative multicarrier dynamics,
or assessing whether the stored complex CSI admits physically valid calibrated
phase features. These are hypotheses requiring a fresh protocol, not fixes
already validated here. Common-mode removal could remove respiration too;
uncalibrated phase could introduce artifacts. Better physical interference
identification and reliability validation additionally need independent continuous
recordings with respiratory reference and explicit interference/empty controls.
Existing five-person overlapping segments cannot establish deployment safety.

## Reproduction and verification

```bash
MPLCONFIGDIR=/private/tmp/wifi_motion_mpl_20261029 .venv/bin/python -m backend.modules.breathing_detection.evaluation.audit_peak_evidence
.venv/bin/python -m backend.modules.breathing_detection.evaluation.evaluate_peak_support
.venv/bin/python -m unittest discover -s backend/modules/breathing_detection/tests -q
```

Completed runs intentionally refuse overwriting their directories. Audit artifacts
are in `outputs/peak_evidence_v1/` (`config.json`, `window_evidence.json`,
`audit_summary.json`, 18 PNGs); candidate artifacts in `outputs/peak_support_v1/`
(`config.json`, `window_predictions.json`, `changed_cases.json`, `metrics.json`).
They are local ignored experiment outputs, not frontend data exports.

Configuration records protocol/source/code hashes and protected original unified
predictions/metrics, checkpoint, replay files and existing frontend lock hash.
The full module suite passes **129 tests**, including 13 new tests for candidate
evidence, read-only extraction, spectral labels, support/tie/weak-ripple behavior,
explicit invalid input and baseline fallback. A synthetic limitation test also
shows that most carriers can consistently support a wrong component: the tests
do not mistake internal consistency for physiological correctness.
