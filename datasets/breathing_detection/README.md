# Module: breathing detection

| | |
|---|---|
| Data | CSI-Bench task `BreathingDetection`, and WiRe-Breath (not CSI-Bench) |
| Source | Kaggle `guozhenjennzhu/csi-bench`, version 12 (see the main README for the link and the citation); WiRe-Breath: https://github.com/Yunpeng-Ge/WiRe-Breath |
| Licence | CSI-Bench: CC BY-NC-ND 4.0. WiRe-Breath: the terms on its authors' page |
| Size | CSI-Bench: 100,008 files, 12.8 GB. WiRe-Breath: 467 MATLAB files |
| Read by | `backend/modules/breathing_detection/` (`data/BreathingDetection` and `data/WiReBreath`) |

## CSI-Bench

```bash
pip install remotezip requests
python3 datasets/breathing_detection/get_data.py
```

`sample/` holds a few original files. The module's own documents say the local copy of CSI-Bench was not checked against a release
version; `get_data.py --verify-only` on that copy does it.

## WiRe-Breath

Used for development and error analysis of the breathing module. 467 MATLAB segments from 85 recordings, 5 subjects, 2 rooms (Quiet 240,
Interference 227 segments); every segment has complex CSI of shape 1500 by 234 and a reference respiratory waveform. The module's audit
(`WIRE_BREATH_DATA_AUDIT.md` on the module branch) found all files loadable with finite values. The files are used unchanged, as released
by the authors. They are not downloaded by the script here: the owner of the module sends them separately, to be placed in
`datasets/breathing_detection/data/WiReBreath`.
