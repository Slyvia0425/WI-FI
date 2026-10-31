# Module: human activity recognition

| | |
|---|---|
| Data | CSI-Bench task `Multitask`, folders `HumanActivityRecognition` (metadata and splits) and `sub_Human_h5` (the windows) |
| Source | Kaggle `guozhenjennzhu/csi-bench`, version 12 (see the main README for the link and the citation) |
| Licence | CC BY-NC-ND 4.0 |
| Size | 80,915 files, 13.1 GB (the rest of `Multitask`, 14 GB of `.mat` files and two other tasks, is not read) |
| Read by | `backend/modules/human_activity_recognition/` (`HAR_DATA_ROOT` or `CSI_DATA_ROOT`) |

## Get it

```bash
pip install remotezip requests
python3 datasets/human_activity_recognition/get_data.py
```

`sample/` holds a few original files of the two folders (metadata, split lists and h5 windows). The module branch also has
`fixtures/mini_csi_bench/`; its README says it is a synthetic, deterministic fixture for tests, not real CSI-Bench data.
