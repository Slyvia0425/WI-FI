# Module: fall detection

| | |
|---|---|
| Data | CSI-Bench task `FallDetection` |
| Source | Kaggle `guozhenjennzhu/csi-bench`, version 12 (see the main README for the link and the citation) |
| Licence | CC BY-NC-ND 4.0 |
| Size | 6,708 files, 0.9 GB |
| Read by | `backend/modules/fall_detection/` |

## Get it

```bash
pip install remotezip requests
python3 datasets/fall_detection/get_data.py
```

This does the same as `backend/modules/fall_detection/download_data.py` on the module branch: version 12, only the `FallDetection` folder,
a size and checksum check of every file.

The module's activity probe and mixed-data candidate also read single windows from `Multitask/sub_Human_h5` of the same Kaggle dataset;
its own scripts fetch those files. The folder `Multitask/sub_Human_h5` can be downloaded completely with
`datasets/human_activity_recognition/get_data.py`.
