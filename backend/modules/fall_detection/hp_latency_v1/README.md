# HP latency measurement

This experiment times existing seed-42 checkpoints on HP validation input only.
It does not train an HP-only model or change the earlier mixed-device training.
No test-set evaluation, ESP32 inference, threshold selection, or alarm evaluation
is performed. Frozen sources and model files are checked for changes.

From the repository root, using its existing virtual environment:

```bash
.venv/bin/python backend/modules/fall_detection/hp_latency_v1/benchmark.py --output backend/modules/fall_detection/reports/hp_latency_v1_repeat
.venv/bin/python backend/modules/fall_detection/hp_latency_v1/report.py --results backend/modules/fall_detection/reports/hp_latency_v1_repeat/results.json
```

Choose a new result directory; existing experiments are preserved. This version
requires macOS with working MPS as well as CPU. Both models run sequentially,
FP32, batch one, CPU threads two, 30 warmups and 3 x 200 timings per scope.
The 16 audited HP validation files must remain at the manifest's actual paths.
Checkpoints are loaded from their existing final-selection locations.

`results.json` contains hardware, source/checkpoint hashes, all scope summaries,
correctness checks and three independent-process startup observations per model
and backend. Each model/backend CSV contains the raw 1,800 timing observations
across three scopes. `report_zh.md` explains interpretation and limitations.

Forward latency, in-memory processing and warm-cache HDF5 processing have
different measurement boundaries. None includes signal collection/window waits
or establishes event-to-alert latency. Whole-process peak RSS includes imports
and checkpoint loading and is not isolated model/GPU memory.
