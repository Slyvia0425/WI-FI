# WiRe-Breath Dataset

This package provides the dataset associated with **WiRe-Breath: A Sustainable WiFi-Based Real-Time Respiratory Monitoring Solution**.

The data are stored as MATLAB `.mat` files. Each file represents one respiratory monitoring data point of approximately 30 seconds and contains WiFi CSI measurements together with the corresponding ground-truth respiratory signal.

## Dataset Contents

```text
WiRe_Breath_Release_v1.0/
  README.md
  LICENSE
  main.m
  load_wire_breath_data.m
  data/
    Room1/
      Room1_50Hz_Subject01_Foetus_Interference_Trial01_seg001.mat
      Room1_50Hz_Subject02_Normal_Quiet_Trial01_seg001.mat
      ...
    Room2/
      Room2_50Hz_Subject02_Foetus_Interference_Session0816_Trial01_seg001.mat
      Room2_50Hz_Subject03_Sitting_Quiet_Session0817_Trial01_seg001.mat
      ...
```

The dataset contains 467 data files from five participants:

| Room | Quiet | Interference | Total |
|---|---:|---:|---:|
| Room1 | 80 | 90 | 170 |
| Room2 | 160 | 137 | 297 |
| **Total** | **240** | **227** | **467** |

| Participant | Room1 | Room2 | Total |
|---|---:|---:|---:|
| Subject01 | 158 | 0 | 158 |
| Subject02 | 10 | 141 | 151 |
| Subject03 | 2 | 74 | 76 |
| Subject04 | 0 | 25 | 25 |
| Subject05 | 0 | 57 | 57 |
| **Total** | **170** | **297** | **467** |

Subject02 and Subject03 have data in both rooms. The uncompressed dataset is approximately **725 MB**.

`main.m` is a script that loads the first Room1 data file and extracts its variables directly into the MATLAB workspace. `load_wire_breath_data.m` is the underlying lightweight reader. Neither file contains respiratory sensing, signal-processing, or estimation methods.

## File Names

Files in `Room1` follow this format:

```text
Room1_50Hz_Subject<subject_id>_<Posture>_<Environment>_Trial<trial_id>_seg<segment_id>.mat
```

Files in `Room2` follow this format:

```text
Room2_50Hz_Subject<subject_id>_<Posture>_<Environment>_Session<session_id>_Trial<trial_id>_seg<segment_id>.mat
```

| Field | Meaning |
|---|---|
| `Room1`, `Room2` | Experimental room, using the room labels from the paper. |
| `50Hz` | CSI sampling rate in this release. |
| `Subject01`--`Subject05` | Participant identifier assigned in this release. |
| `<Posture>` | Recorded posture label. |
| `Quiet` | Data collected in a quiet environment. |
| `Interference` | Data collected with motion interference in the environment. |
| `Session0816`, `Session0817` | Recording session identifier used for Room2 data. |
| `TrialXX` | Trial index. |
| `segXXX` | Data segment index within the corresponding recording. |

The posture labels are:

```text
Foetus, Freefaller, Log, Normal, Semi, Sitting, Soldier, Starfish, Yearner
```

`Normal` is a general resting-posture label used in Room1.

## MAT File Format

Each MAT file contains the following variables:

| Variable | Shape | Type | Description |
|---|---:|---|---|
| `csi_complex` | `1500 x 234` | complex `single` | Complex CSI values. |
| `csi_abs` | `1500 x 234` | `single` | CSI amplitude values. |
| `time` | `1500 x 1` | `single` | CSI timestamps in seconds, relative to the start of the data point. |
| `gt_time` | variable length | `single` | Ground-truth respiratory timestamps aligned with the data point. |
| `gt_signal` | variable length | `single` | Ground-truth respiratory signal. |
| `metadata` | struct | MATLAB struct | Room, participant, posture, environment, recording index, sampling rate, and data dimensions. |

Ground truth is included in every MAT file through `gt_time` and `gt_signal`. The CSI matrices contain 234 usable subcarriers.

## Citation

If you use this dataset, please cite:

```bibtex
@article{ge2025wirebreath,
  author  = {Yunpeng Ge and He Wang and Ivan Wang-Hei Ho},
  title   = {WiRe-Breath: A Sustainable WiFi-Based Real-Time Respiratory Monitoring Solution},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025},
  pages   = {1--12},
  doi     = {10.1109/JBHI.2025.3619664}
}
```

## License

This dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). See `LICENSE` for details.
