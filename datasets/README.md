# Datasets

Where the data of every module comes from, and how to get it again. The data itself is not in git: it is large, and the CSI-Bench licence
(CC BY-NC-ND 4.0) lets us share the original files but not changed copies. So each module has a description (source, version, licence),
a script that fetches the original files and checks them, and a few original files as a sample.

```
datasets/
  README.md                  this file
  download_csi_bench.py      fetches and verifies CSI-Bench files from the official archive
  <module>/README.md         source, version, licence, size, what the module reads, how to get it
  <module>/get_data.py       the command for this module (calls download_csi_bench.py)
  <module>/sample/           a few original files
  <module>/data/             where get_data.py puts the files (git-ignored)
```

## Modules and sources

| Module | Data | Size | Licence |
|---|---|---|---|
| `motion_source_recognition` | CSI-Bench `MotionSourceRecognition` | 9.3 GB, 60,867 files | CC BY-NC-ND 4.0 |
| `human_activity_recognition` | CSI-Bench `Multitask`: `HumanActivityRecognition` and `sub_Human_h5` | 13.1 GB, 80,915 files | CC BY-NC-ND 4.0 |
| `fall_detection` | CSI-Bench `FallDetection` | 0.9 GB, 6,708 files | CC BY-NC-ND 4.0 |
| `breathing_detection` | CSI-Bench `BreathingDetection`, and WiRe-Breath (not CSI-Bench) | 12.8 GB, 100,008 files, and 467 MATLAB files | CC BY-NC-ND 4.0, and WiRe-Breath under its authors' terms |
| `space_proximity_sensing` | simulated by the group (not CSI-Bench) | 11.5 GB | produced by our own code |

## CSI-Bench

CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing. WiFi channel state information recorded continuously in real
users' homes with existing IoT devices. Kaggle `guozhenjennzhu/csi-bench`, **version 12** (81.5 GB, six tasks, 352,000 files):
https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench. Licence: Attribution-NonCommercial-NoDerivatives 4.0 International
(CC BY-NC-ND 4.0). Our use is non-commercial (university coursework). The files belong to the authors of CSI-Bench.

Citation (DOI https://doi.org/10.34740/kaggle/dsv/16896361):

```bibtex
@misc{guozhen_zhu_yuqian_hu_weihang_gao_wei-hsiang_wang_beibei_wang_k__j__ray_liu_2026,
  title={CSI-Bench},
  url={https://www.kaggle.com/dsv/16896361},
  DOI={10.34740/KAGGLE/DSV/16896361},
  publisher={Kaggle},
  author={Guozhen Zhu and Yuqian Hu and Weihang Gao and Wei-Hsiang Wang and Beibei Wang and K. J. Ray Liu},
  year={2026}
}
```

## Getting the files

```bash
pip install remotezip requests
python3 datasets/motion_source_recognition/get_data.py          # or the folder of any other module
python3 datasets/motion_source_recognition/get_data.py --verify-only
```

- No Kaggle account and no copy of the whole archive are needed: only the named files are read from the online archive.
- Every file is checked against the size and CRC32 checksum stored in the official archive, so the files are byte for byte the official ones.
  The script stops if a task does not have the number of files of version 12.
- `--sample` fetches a few files only. A receipt (`download_receipt.json`) is written next to the data.
- Other tasks or folders: `python3 datasets/download_csi_bench.py --destination <folder> --task <Task> [--only <Task>/<folder>]`.
  `--all` fetches all six tasks (81.5 GB).
- A module reads the folder of its task. For module A, set `CSI_DATA_ROOT` to the absolute path of
  `datasets/motion_source_recognition/data/MotionSourceRecognition`.

## For the submission

The person in charge of the datasets downloads this folder; it holds everything needed to get the CSI-Bench data and the samples.
Data from outside CSI-Bench (WiRe-Breath for breathing detection, the simulated data of space and proximity sensing) is sent to that
person separately by its owner, who puts it into the module's folder.
