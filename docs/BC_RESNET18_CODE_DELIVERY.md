# B/C integration and HP mixed ResNet18 code delivery

This branch keeps the current B/C behavior and adds the completed fall-detection research code and validation evidence. It does not replace any trained model or change a threshold.

## Current application

- B uses the author's released seed42 spectral ExtraTrees model on the same HP input as C. Its five activity classes do not include Fall.
- B confidence below 0.6 selects the existing CSI-Bench ResNet18 C. Otherwise C is not evaluated and its score/decision remain null.
- The existing C threshold is 0.574872225522995. The routing limitations, including skipped real falls in the validation replay, remain explicit in [B_LOW_CONFIDENCE_GATE.md](B_LOW_CONFIDENCE_GATE.md).
- The website displays saved predictions. Selecting a sample in the page does not run the models again.

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/modules/fall_detection/requirements.txt
npm ci --prefix frontend
PYTHON=.venv/bin/python npm run dev
```

Open `http://localhost:5173/#/module/fall_detection`. The committed backend and frontend replay JSON files allow the recorded demonstration to run without model weights or raw CSI files. `scripts/serve_demo.py` additionally requires a frontend build (`npm run build`).

## Independent HP mixed-data candidate

Code: [mixed_hp_resnet_v1](../backend/modules/fall_detection/mixed_hp_resnet_v1/README.md).

- HP only, complete segments `[B,1,500,232]`, pinned official per-segment normalization, no padding or resizing.
- Training: 5,341 segments, including 1,000 official HAR training segments (200 per activity). Fall=1,727 and Nonfall=3,614.
- Validation: 1,176 fixed segments; no training/validation ID or raw-content overlap after the documented training exclusions.
- Seed42, AdamW, maximum100 epochs, patience10; stopped at epoch28, best checkpoint epoch18.
- The candidate is separate from the model currently used by the application. No test-set evaluation or new threshold optimization was performed.

At the common fixed threshold0.5:

| Measure | Existing ResNet18 | HP mixed candidate |
|---|---:|---:|
| Falls detected | 384/390 | 384/390 |
| Original Nonfall false positives | 27/536 | 16/536 |
| Ordinary activity triggers (weak Nonfall labels) | 192/250 | 0/250 |
| Fall F1 | 95.88% | 97.22% |

The six missed falls are entirely different between models: six original misses are corrected and six new misses appear. The activity labels are weak segment-level Nonfall labels, and source environments differ. This is validation evidence, not independent generalization or continuous false alarms/hour.

Full tables and per-sample evidence: [comparison report](../backend/modules/fall_detection/reports/mixed_hp_resnet_v1/comparison/report_zh.md), [error transitions](../backend/modules/fall_detection/reports/mixed_hp_resnet_v1/comparison/error_transition_by_label.json), and [standalone HTML report](../backend/modules/fall_detection/reports/mixed_hp_resnet_v1/report_zh.html).

## Model files and data

Raw CSI, NumPy caches, checkpoints, joblib models, temporary browser outputs and model ZIPs are not Git source files. All 35 local B/C checkpoints and their companion files are versioned separately in the [matching model Release](https://github.com/Slyvia0425/WI-FI/releases/tag/bc-models-20261031-v1). See [model restoration](../model_artifacts/README.md) for checksum-verified downloads (`python3 scripts/restore_bc_models.py`, or add `--all` for every research version).

| Artifact | Expected repository-relative path | SHA256 |
|---|---|---|
| Existing C ResNet18 | `backend/modules/fall_detection/outputs/official_resnet18_seed42_recovery_v1/reproduction_bundle.pt` | `e3aabc34810aee4a647555eb7b1970bbf19a063b9d76f31d4ff59bfc5521a0a0` |
| New HP candidate | `backend/modules/fall_detection/outputs/hp_mixed_resnet18_v1/best.pt` | `9ec1016f6d3fbce2f76b686ba839a90e1baa6ba882102f9af28fd5599ae9ae9d` |
| Released B | `backend/modules/human_activity_recognition/outputs/released_v1/optimized_final/spectral/seed-42-stable/model.joblib` | `bb991b90504f5520cdf83fb9a07b1b39cf3d1c20a6b875ff1a2ffcf24dec7687` |

B originally comes from the author's [release](https://github.com/Bartley007/WI-FI/releases/tag/module-b-data-models-v1). The matching B/C Release preserves the exact paths and hashes above before real inference. Its restore script refuses to overwrite a different local model. The candidate remains separate from the frozen application model.

Dataset download and audit scripts are included. Official metadata/split snapshots and selected sample identities are in the reports; actual CSI arrays are excluded. Some historical research scripts retain the author's local paths and require path configuration on another machine. For the mixed-data preparation, `MIXED_HP_DATA_ROOT` sets a portable raw-data root while preserving the predeclared selection.

## Verification

```bash
.venv/bin/python -m unittest discover -s backend/tests -t .
OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s backend/modules/human_activity_recognition/tests -t .
npm run build
```

The historical TCS regeneration test is optional and skips when the original research artifacts are not included. Current B/C contract and routing tests still run. Static checks follow the existing Module-B workflow. Formatting, explicit existing `zip` behavior and NumPy type annotations in the publishing copy do not change numerical operations, thresholds or model weights.
