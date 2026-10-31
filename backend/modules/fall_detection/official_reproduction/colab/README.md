# Official ResNet18 on Google Colab

Live notebook: https://colab.research.google.com/drive/1NJ603AiueaTaTEh-e_DBywZzKA2bPnop

This is an independent CUDA run. Existing local checkpoints and code are preserved.
The notebook copies official source at commit
`b1beac7b0fa46e01748901195ecfbe309dfc0180`, downloads only the public Kaggle
v12 FallDetection portion, and verifies all 6,708 file sizes and ZIP CRC32 values.
It constructs only `train_id` (4,690 records) and `val_id` (1,005 records).
No test Dataset is constructed, and no threshold is tuned.

The free Tesla T4 has 14.56 GiB memory. The batch 128 forward/backward/AdamW
probe passed with peak allocated memory 6.3465 GiB. Probe weights are discarded;
formal training starts in a new Python process with seed 42.

Training uses official ResNet18, no pretrained weights, FP32, unweighted
CrossEntropyLoss, AdamW lr 0.001 / weight_decay 1e-5, batch size 128,
maximum 100 epochs, patience 15, and lowest validation loss checkpoint selection.
The official batch-count warmup lambda is advanced once per epoch exactly as
in the pinned source; this known schedule mismatch is deliberately retained.
The first epoch's learning rate is zero. The input is `[B,1,500,232]`;
per-record normalization precedes feature padding, and the official mapping is
Fall=0, Nonfall=1. Fall score is softmax channel 0. Metrics use official argmax.
T4 hardware and num_workers=0 differ from the paper setup and are recorded.
One known identical train/validation content pair remains in the official v12
split; results should be interpreted with that limitation.

Run cells in order: environment → setup/probe → download → write training
script → training → independent reload/export. The live run has started;
check the notebook outputs for current progress. Do not press Run All again
while training is active. Existing run output paths intentionally cannot be
overwritten; a new experiment needs a fresh run name.

The final cell is queued after training. It verifies the saved checkpoint in a
new Python process on validation, prints per-device metrics, displays curves,
and requests download of `CSI-Bench_ResNet18_seed42_results.zip`. That result
archive contains checkpoints, configuration, predictions, metrics, curves,
source wrapper, data receipt, ZIP index, and full training log. It excludes the
raw dataset. Download completion has not yet been verified. The notebook does
not mount Drive: `/content` artifacts last only for the current Colab runtime,
so retain the result archive before ending that runtime.

A portable local notebook copy and all authored helper scripts are in this
folder. `session_status.json` describes the last UI-observed status; it is a
snapshot, not a live monitor. No unobserved training metrics are asserted.
