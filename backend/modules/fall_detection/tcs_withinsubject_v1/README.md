# TCS-Fall subject01 GCV/CNN pilot

This is an independent within-subject feasibility experiment, authorized by the
user while waiting for a corrected download. Three known-corrupt records are
explicitly excluded. Existing HP/CSI-Bench models and thresholds are preserved.

Only207 intact single-action recordings are used:165 training (72Fall,93Nonfall)
and42 validation (18Fall,24Nonfall). Seed42 fixes six complete recordings per
activity in validation. No recording or exact audited canonical-IQ content crosses
the split. Independence of neighboring repeats/sessions is not established.
The12 continuous recordings are reserved, and activity16/17 is excluded.

## Inputs and reconstructed preprocessing

`prepare.py` reads actual audited `.dat` members from Downloads/1.zip, checks
their content against the raw audit, corrects antenna order, applies reviewed
RF scaling, takes native Tx0 (3Rx x30 positions), and retains the first8000 packets
as in the public visualization notebook. It logs the omitted tail and actual
timestamp span rather than pretending packets are an exact1000Hz clock.

The reconstruction uses an8th-order Butterworth normalized cutoff0.2 and
`scipy.signal.filtfilt` within that segment, then nonoverlapping20-packet groups:
population std/mean becomes `[400,90]`. Exactly zero means return zero; no NaN
filling, padding, resampling, augmentation or HP normalization is performed.
No train-fitted normalization parameters are used. The CNN receives `[B,1,400,90]`.
This is an offline fixed-segment recipe, not a completed continuous alarm system.

`preprocess.ipynb` is absent from the public/downloaded code. The reconstruction
is explicitly defined and must not be called a faithful complete-paper
reproduction. SA event times are unavailable, so truncation to the public example's
first8000 packets cannot guarantee every entire fall trajectory is retained.

## Model and training

`model.py` contains the exact CNN-class AST from the downloaded official
`code/models.py`, initialized randomly. It has1,233,474parameters and five2D
convolution groups, followed by a1280-to2 classifier. See model_provenance.json.
The upstream GPL-3.0 license is retained in LICENSE.upstream.

Adam lr0.0001, unweighted cross-entropy, batch8 for local M2 8GiB memory, seed42,
maximum100epochs, patience10 based on validation loss. Best checkpoint selection
uses validation loss, not F1. Fall is class1, Nonfall class0. Scores are Softmax
column1 and threshold stays0.5. No test data is created or evaluated.

## Commands

From the repository root with its existing virtual environment; use fresh output
paths for reruns:

```bash
.venv/bin/python backend/modules/fall_detection/tcs_withinsubject_v1/prepare.py --output backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat
.venv/bin/python backend/modules/fall_detection/tcs_withinsubject_v1/train.py --data-dir backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat --output backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat/run --report backend/modules/fall_detection/reports/tcs_subject01_gcv_cnn_seed42_repeat
.venv/bin/python backend/modules/fall_detection/tcs_withinsubject_v1/verify_reload.py --data-dir backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat --checkpoint backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat/run/best.pt --report backend/modules/fall_detection/reports/tcs_subject01_gcv_cnn_seed42_repeat
```

The source archive must still match the audited archive fingerprint. A newly
downloaded/repaired archive needs its own audit and experiment rather than silent
reuse of the original cohort. The preparation and native-decoder hashes are pinned.

`shape_smoke_check.json` records actual layer outputs and two finite backward
updates on a balanced8-record training subset. Formal training reinitializes the
model and seed after this smoke check. `history.csv` records every epoch;
`training_summary.json` is written only after the run actually finishes.

The checkpoint retains weights, recipe, label mapping, data/split/source hashes,
excluded/reserved IDs, best epoch and threshold. A fresh CPU process checks all42
validation scores against saved MPS scores; independent single-record inference
starts from raw ZIP data using the same recipe.

## Scientific scope

A high score on42 same-person repeated activities does not establish cross-user,
cross-session, cross-hardware or continuous-alarm generalization. No comparison
with CSI-Bench test scores is justified by this pilot. More subjects and explicit
continuous validation are still needed. No event recall, false alarms/hour or
event-to-alert latency is reported here.

## Completed seed42 run

The actual MPS run completed100epochs in443.6seconds; best validation loss0.23128879
was at epoch100. Patience10 did not trigger. At the fixed threshold0.5, the42
within-subject validation recordings gave TP18/FP0/FN0/TN24. This is a small
single-subject pilot result, not a test or continuous-alarm score.

The new-process CPU replay of all42 validation scores differed from saved MPS
scores by at most1.1920928955078125e-7, with no changed classes. Two additional
fresh-process inferences began with actual ZIP .dat data. To reproduce the
verified Fall example:

```bash
.venv/bin/python backend/modules/fall_detection/tcs_withinsubject_v1/infer.py --checkpoint backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_v1/run/best.pt --archive /Users/chenxiaohe/Downloads/1.zip --member 1/a_01_00_07.dat --backend cpu
```

The actual Fall score was0.9611639380. The tested Nonfall member
`1/a_01_03_01.dat` scored0.2964239717. These were the first sorted validation IDs
per class, not selected for their scores. They are uncalibrated classification
scores and do not establish a live alarm's performance.

The detailed Chinese report and curves are in
`reports/tcs_subject01_gcv_cnn_seed42_v1/report_zh.md`. The best checkpoint is
`outputs/tcs_subject01_gcv_cnn_seed42_v1/run/best.pt`; its SHA256 is
`4d5d40831f37d3a226f7552811487c2e74b6333cfce46a6478953cf1d931fe39`.
