# Motion source recognition (backend)

Output: Human, Pet, IRobot or Fan, or unknown when the evidence is weak.

Everything for this module on the backend lives in this folder. Work on your own branch and edit only here.

| Folder | Put here |
|---|---|
| `preprocessing/` | Data reading, quality checks, splits and features. |
| `training/` | Model definitions and training scripts. |
| `experiments/` | Model comparison, ablations and threshold studies. |
| `evaluation/` | Metrics, confusion matrices, error analysis, and the export of outputs/replay.json. |
| `outputs/` | Generated files: `replay.json` (read by the server), checkpoints, figures. Large files stay out of git. |
| `api.py` | What the server returns for this module |
| `inference.py` | One window in, one output frame out (`Detector`); the interface is described in `INTERFACE.md` |
| `calibration.py` | Probability helpers shared by `inference.py`, the experiments and the evaluation (temperature, reliability) |
| `contract.py` | The frame fields this module promises |
| `config.json` | Every tunable number of the module (data shape, split sizes, seeds, parameter grids, thresholds, website settings); read through `preprocessing/config.py` |
| `requirements.txt` | Optional, packages only this module needs |

Frontend counterpart: `frontend/src/modules/motion_source_recognition/`.
Export a result for the website with `backend.shared.export.write_replay("motion_source_recognition", payload, bundle=True)`. One call updates the backend output and the copy bundled with the website.
Develop alone with `python -m backend.api --only motion_source_recognition`.

## Architecture

```text
config.json  ->  preprocessing/config.py          the only reader: constants, paths, the model registry (names, labels, routes)

preprocessing/   data in, features out            manifest, quality, splits, baselines, features, minirocket, physical, windows (reads the files)
      |  outputs/prep/
training/        one model per split              random_forest*, logistic_regression, summary; common.py (loading, scoring, metrics), rooms.py (folds inside one room)
      |  outputs/models/
experiments/     what was tried, validation rows  one script per question; results change nothing downstream unless marked "choice"
      |  outputs/experiments/
evaluation/      choice, test rows, website       model_selection -> test_scoring -> ... -> export_replay (the last step)
         outputs/evaluation/, outputs/replay.json

calibration.py, inference.py   the live module: window in, frame out; they use preprocessing and training only
```

One direction only: a folder imports from the folders above it, never from below (checked by `tests/test_architecture.py`). Only the evaluation calls the live module, to show its answers.
Every number you may want to change is in `config.json`; every script writes to its own file in `outputs/` and reads the files of the steps above it.

## Frame fields

Defined in `contract.py`, besides `t` (seconds).

| Field | Meaning |
|---|---|
| `label` | Human, Pet, IRobot, Fan or unknown |
| `confidence` | probability of the chosen label, 0 to 1 |
| `probs` | probability of each class; null when the window is invalid |
| `quality` | optional: `{ok, level: ok / degraded / invalid, reasons}` |

The `frames` of the replay are what the home page and the fusion page read: five invented demonstration frames (`config.json`, `website.landing_demo`), not results of the detector; they only drive the floor-plan animation. Their time stamps are those of the other modules' demonstration data (the home page walks through the union of all modules' time stamps, so keep them equal). The real answers are in the preview of this module's own page, `dashboard.preview_frames`; those frames also carry `truth` (the true class, or `Made-up`) and, for the made-up damaged windows, `note`. These two are for the demo only and are not part of the interface (see `INTERFACE.md`).

## Python environment

This module has its own virtual environment, so its packages (aeon and numba for MiniRocket, scikit-learn, scipy, pandas, h5py) do not touch other modules or your system Python. It lives in `.venv/` here and is not committed.

```bash
python3 -m venv backend/modules/motion_source_recognition/.venv
backend/modules/motion_source_recognition/.venv/bin/python -m pip install -r backend/modules/motion_source_recognition/requirements.txt
```

Run every script of this module with `backend/modules/motion_source_recognition/.venv/bin/python -m ...` from the repository root. The dataset location is set in `.env.local` (copy `.env.example`).

## Pipeline

Run from the repository root with the module's Python (`-m backend.modules.motion_source_recognition.<name>`). All results go to `outputs/` (not committed). Training and the experiments score the validation rows; the test rows are scored in `evaluation.test_scoring`, after the model choice is frozen.

| Step | Module | Result |
|---|---|---|
| Data preparation | `preprocessing.manifest`, `quality`, `splits`, `baselines`, `features`, `minirocket`, `physical` (raw pass, then assembly) | `outputs/prep/` |
| Random Forest on statistics | `training.random_forest` | `outputs/models/random_forest/` |
| Logistic Regression on MiniRocket | `training.logistic_regression` | `outputs/models/logistic_regression/` |
| Random Forest on MiniRocket | `training.random_forest_minirocket` | `outputs/models/random_forest_minirocket/` |
| Random Forest / Logistic Regression on statistics + physical features | `training.random_forest --route statistics_physical`, `training.logistic_regression --route statistics_physical` | `outputs/models/*_statistics_physical/` |
| Table of all saved models | `training.summary` | `outputs/models/summary.md` |
| 1 Freeze the model choice (validation only) | `evaluation.model_selection` | `outputs/evaluation/model_selection.*` |
| 2 Score the test rows | `evaluation.test_scoring`; with `--with-extra` also the two statistics + physical models | `outputs/evaluation/test_scores.*`, `test_scores_extra.*` |
| 3 Main model against the two reference lines | `evaluation.baseline_comparison` | `outputs/evaluation/baseline_comparison.*` |
| 4 Error analysis | `evaluation.error_analysis` | `outputs/evaluation/error_analysis.*` |
| 5 Inference time and size, one thread | `evaluation.latency` | `outputs/evaluation/latency.*` |
| 6 Leave one room out (main model); one fixed room | `evaluation.leave_one_room_out`; `evaluation.single_room` | `outputs/evaluation/leave_one_room_out.*`, `single_room_E11.*` |
| 7 Confidence threshold and unknown | `evaluation.confidence_threshold` (`--stage 1` curve, `--stage 2` one threshold for all classes, `--stage 3` one threshold per class) | `outputs/evaluation/confidence_threshold*.{json,md}` |
| 8 Interface | `inference.Detector`, `evaluation.example_frames` | `INTERFACE.md`, `outputs/evaluation/example_frames.json` |
| Training curves for the website (forest growth, Logistic Regression refits) | `evaluation.training_curves` | `outputs/evaluation/training_curves.json` |
| What is inside the five models (one tree and the 300 votes; weights over the iterations) | `evaluation.model_anatomy` | `outputs/evaluation/model_anatomy.json` |
| One real window per class turned into features, step by step (data stage of the website) | `evaluation.feature_demo` | `outputs/evaluation/feature_demo.json` |
| Website | `evaluation.export_replay` (run it last, after `training_curves`, `model_anatomy` and `feature_demo`; real replay frames plus six made-up damaged windows, tables and the numbers of the dashboard; writes `outputs/replay.json` and the copy in the frontend module folder) | `outputs/replay.json`, `frontend/src/modules/motion_source_recognition/data/replay.json` |

Each model folder holds `model.joblib`, `probabilities.npz` (validation and test rows) and `run.json` (parameters, scores, timing). Tests: `python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .`

## Where to find each result

| Result | File in `outputs/evaluation/` |
|---|---|
| The chosen classifier and why | `model_selection.md` |
| Scores of every model on the 3 splits (mean, standard deviation, IRobot and Fan per split) | `test_scores.md`, `test_scores_extra.md` |
| Confusion matrices, Human against Fan and Pet against Human, errors by room and recording | `error_analysis.md` |
| Comparison with the two reference lines | `baseline_comparison.md` |
| Time and size of one window | `latency.md` |
| Confidence threshold: the curve, the frozen threshold used by the interface, the per-class alternative | `confidence_threshold.md`, `confidence_threshold_test.md`, `confidence_threshold_per_class_test.md` |
| New room and new people: leave one room out; one fixed room | `leave_one_room_out.md`, `single_room_E11.md` |
| Output fields, unknown rule, quality, downstream use, limits | `INTERFACE.md` (in this folder) |

How the test rows were used, for the final report: the model choice was frozen on the validation rows before any test score (`model_selection.json` holds fingerprints of the saved test probabilities, checked before scoring). The test rows were then used for the test scores (computed twice: the second run only changed the uncertainty ranges, the scores are identical), the two statistics + physical models (their features were designed after looking at the whole dataset, so their scores are optimistic and marked), the reference-line comparison, the error analysis, and two reports of the confidence threshold (one threshold for all classes, which the interface uses; then one per class, chosen afterwards). Leave one room out and the one-room test hold every window out once and are separate from the frozen protocol.

## Experiments (`experiments/`)

Those marked "choice" fed the model choice or the evaluation; those marked "report" only document what was tried and change nothing downstream. They use the validation rows, except `physical_evaluation`, which also uses whole rooms of the dataset.

| Module | Question | Result (3 splits, mean) | Role |
|---|---|---|---|
| `fair_comparison` | Is the algorithm or the feature set responsible for the difference? (2 x 2: Random Forest / Logistic Regression on statistics / MiniRocket) | Macro-F1 RF+statistics 0.518, LR+statistics 0.442, RF+MiniRocket 0.530, LR+MiniRocket 0.552: the features matter more | choice |
| `model_comparison` | Does averaging the two official models help? How do they do on seen and unseen users? | No gain (0.550 against 0.552); rare classes are recognised only when the user is in the training rows | choice |
| `calibration_check` | Does the model's confidence mean what it says? | Logistic Regression is over-confident in splits 0 and 1 and flat in split 2; the forests are steadier | choice (confidence threshold) |
| `temporal_smoothing` | Does averaging the probabilities of the last windows help? | About +0.005 to +0.01 macro-F1 with 6 windows; no harm | choice (optional post-processing) |
| `forest_checks` | Are 300 trees enough? What do leaf size and class weights do? | The curve is flat beyond 100 trees; class weights trade rare-class recall against Human and Pet accuracy | report |
| `level_features` | How much do the level features (mean, min, max) matter? | Removing them costs 0.01 to 0.02 (Random Forest) and 0.015 to 0.045 (Logistic Regression) | report |
| `complementary` | Does gradient boosting on statistics make different mistakes? | Macro-F1 0.504, accuracy 0.818; no gain when averaged with the other models | report |
| `physical_evaluation` (features from `preprocessing.physical`) | Do physically motivated features (spread over channels, periodicity, 20 to 50 Hz content, bursts, dynamics over recent windows) help, alone or added to the statistics or MiniRocket features? | In one home (E11): robot recall 0.10 with MiniRocket and 0.19 with statistics, 0.64 with a Random Forest on 52 physical numbers (Human 0.95, Pet 0.94); added to MiniRocket it changes nothing; leave one room out: no gain (accuracy about 0.25) | report |
| `spectral_features`, `spectral_evaluation` | Do spectral-shape features, or fewer antenna links, help? | No gain over MiniRocket; one link alone scores within 0.01 of all four | report |

Also tried and dropped, without a script in the repository: one forest per antenna link, scale-free ratio features, forests on random halves of the recording groups (none improved on the existing models), and MiniRocket after removing each window's own mean (cancelled).
