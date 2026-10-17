# Mini CSI-Bench fixture for Module B

This is a synthetic, deterministic fixture — not real csi-bench data and not a benchmark result.

It mirrors the required subset layout:

```text
Multitask/HumanActivityRecognition/{metadata,splits}
Multitask/sub_Human_h5/*.h5
```

- 16 samples: 4 samples per class
- 8 train / 4 validation / 4 test
- 232 features × 24 frames
- 408 KB total

Regenerate with:

```bash
python3 backend/modules/human_activity_recognition/scripts/make_local_fixture.py
```

Run the full training/replay path with:

```bash
python3 backend/modules/human_activity_recognition/scripts/run_local_fixture.py --epochs 2
```
