# Paired frequency augmentation fine-tuning pilot

## Design fixed before running

Three arms start from the frozen seed-123 model: unchanged checkpoint, clean
fine-tuning, and frequency-augmented fine-tuning. Both fine-tuned arms use the
same 6,000 stratified CSI-Bench training windows, seed 20261028, batch order,
two epochs, AdamW learning rate 0.0001, weight decay 0.0001, and class-weighted
BCE loss. The augmented arm independently samples an edge-replicated shift
from {-4,-1,0,+1,+4} for every training window, in both classes. The original
probability threshold 0.0545424558 is retained for all arms.

Evaluation uses all 15,000 original validation windows. A fixed stratified
subset of 2,000 validation windows is also scored under each of four
edge-replicated shifts (-4,-1,+1,+4). Stress results average the four
Macro-F1 values. No official test data is read.

The preliminary acceptance criteria were saved before execution: original
validation Macro-F1 and breathing recall may each decrease by at most 0.005,
empty FPR may increase by at most 0.005, and shifted Macro-F1 must improve
against both controls. Passing these criteria permits an external development
check; it does not promote the model to production.

## CSI-Bench validation

| Arm | Original Macro-F1 | Original empty FPR | Original breathing recall | Mean shifted Macro-F1 |
|---|---:|---:|---:|---:|
| Frozen | 0.9679 | 6.83% | 99.19% | 0.7918 |
| Clean fine-tune | 0.9650 | 9.30% | 99.60% | 0.7563 |
| Augmented fine-tune | 0.9668 | 7.04% | 99.16% | 0.8407 |

Augmentation passes all preliminary validation checks. Its shifted Macro-F1
increases by 0.0489 relative to frozen and by 0.0844 relative to clean
fine-tuning. However, residual shifted empty FPR remains substantial:
46.74%, 27.42%, 21.15%, and 38.64% for -4,-1,+1,+4 respectively.
This is an improvement in synthetic shift robustness, not shift invariance.

## WiRe-Breath external development check

The augmentation candidate was evaluated only after passing validation. Clean
fine-tuning was then scored as a matched diagnostic control. WiRe-Breath has
already informed earlier diagnostics and is positive-only; these results
are development observations, not a new independent final test.

| Arm | Detected / 1,293 assessable windows | Recall | Target record detected / 30 |
|---|---:|---:|---:|
| Frozen | 1,257 | 97.22% | 2 |
| Clean fine-tune | 1,260 | 97.45% | 2 |
| Augmented fine-tune | 1,222 | 94.51% | 0 |

The augmented candidate loses 2.71 percentage points of external recall
relative to frozen, and target-record performance worsens. Its Interference
recall declines from 99.20% to 95.02%; Quiet declines from 95.37% to 94.03%.
The clean control does not show the same external deterioration. This
supports an adverse effect associated with this augmentation recipe, although
one seed and a small fine-tuning subset cannot establish a universal result.

## Decision

Do not replace the frozen model. The augmented candidate improves the
specific synthetic perturbations used in training but does not solve the
observed external-domain failure. The clean fine-tune is also not promoted:
its original validation empty FPR exceeds the permitted increase.

Keep all three checkpoints and metrics isolated. Avoid further optimization
against this single known failed recording. A subsequent model experiment
should first define recording/subject-grouped development and untouched
evaluation data, then test weaker augmentation or preservation of the frozen
model's outputs. Matched external negative recordings remain necessary before
claiming improved external binary discrimination. There is no evidence from
this pilot to change the 100 ms timestamp-gap rule.

## Local artifacts and reproduction

Artifacts are under the ignored directory
`outputs/frequency_robustness_pilot_seed20261028/`:

- `config.json`: settings and acceptance criteria written before execution;
- `sample_indices.npz`: reproducible train and validation stress subsets;
- `metrics.json`: three-arm validation results and preliminary checks;
- `clean_finetune.pt` and `augmented_finetune.pt`: isolated candidates;
- `wire_clean.json` and `wire_augmented.json`: external diagnostic results.

`metrics.json` field `passes_pilot=true` means only the predefined CSI-Bench
validation screen passed. The final decision after external evaluation is
**not promoted**, as documented above.

```bash
.venv/bin/python -m \
  backend.modules.breathing_detection.training.train_frequency_robustness \
  --output-dir backend/modules/breathing_detection/outputs/NEW_EXPERIMENT_DIRECTORY
```

The runner refuses an existing configured experiment directory. The original
checkpoint is only read and is never overwritten.
