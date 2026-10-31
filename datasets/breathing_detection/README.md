# Breathing module: datasets and usage

This module uses **two different datasets for different tasks**. The complete
WiRe-Breath copy used locally is included in this branch. CSI-Bench is obtained
from its original download page; no additional CSI-Bench raw data is added by
this contribution.

Run all commands below from the repository root. Use a branch that includes
`backend/modules/breathing_detection/` and this dataset directory together
(`main` now includes both).

## 1. What is included

| Dataset | Purpose in this project | Availability |
|---|---|---|
| CSI-Bench / BreathingDetection | Supervised breathing-evidence vs. empty-room detection; frozen LSTM evaluation and fan-condition analysis | Download separately from [Kaggle](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench) |
| WiRe-Breath v1.0.0 | External positive-only detection checks, respiratory-rate development, timestamp-quality and interference analysis | All 467 original MAT segments included in [`WiReBreath/`](WiReBreath/) |

```text
datasets/breathing_detection/
  README.md
  get_data.py                  existing CSI-Bench downloader wrapper
  sample/BreathingDetection/    existing original CSI-Bench examples
  prepare_wire_data.py          verification and optional local installation
  wire_manifest.json           file sizes and SHA-256 checksums
  WiReBreath/
    LICENSE                    original CC BY 4.0 license
    UPSTREAM_README.md         original dataset documentation
    main.m                     original MATLAB loading example
    load_wire_breath_data.m     original MATLAB reader
    Room1/                     170 original MAT segments
    Room2/                     297 original MAT segments
```

The MAT file contents have not been converted, cleaned, resampled, or relabeled.
Only the packaging paths have been normalized from a Windows-style extraction
to portable `Room1/` and `Room2/` folders. The upstream README is preserved under
the name `UPSTREAM_README.md`. The original MATLAB example assumes the upstream
`data/Room1` layout; use the Python adapter below with this repository's layout.
The manifest fingerprints this project's local originals, not an independently
authenticated upstream checksum list.

WiRe files are ordinary Git files, so a ZIP downloaded from **this branch**
contains the actual data rather than Git LFS pointers. Allow approximately
726 MiB for the uncompressed dataset, plus space for dependencies and caches.

## 2. Python environment

For data processing, use Python 3.11 or newer with a compatible PyTorch build.
Python 3.10 is not a supported choice for this module's current SciPy dependency.
Create an environment and install the module requirements:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell, instead of the line above:
# .venv\Scripts\Activate.ps1
python -m pip install -r backend/modules/breathing_detection/requirements.txt
```

The WiRe file-verification script itself uses only the Python standard library.
The replay server also does not require the scientific packages merely to serve
the existing exported JSON; processing raw data does.

## 3. WiRe-Breath: verify and use the included data

Official source: [WiRe-Breath repository](https://github.com/Yunpeng-Ge/WiRe-Breath).
Fixed release: [v1.0.0](https://github.com/Yunpeng-Ge/WiRe-Breath/releases/tag/v1.0.0).

### Verify the delivered files

```bash
python datasets/breathing_detection/prepare_wire_data.py
```

This checks the complete inventory, Room1/Room2 counts, byte sizes and SHA-256
checksums. No data is downloaded or changed.

### Prepare the path used by existing experiment scripts

Some historical experiment scripts use the fixed path `data/WiReBreath`.
Prepare it without editing those scripts:

```bash
python datasets/breathing_detection/prepare_wire_data.py --install
```

This copies verified MAT files to `data/WiReBreath/Room1` and `Room2`. It reuses
identical files and refuses to overwrite different data. A pre-existing flat
Windows-style extraction is recognized without creating duplicate segments.
The additional working copy needs approximately another 726 MiB. It is ignored
by Git. Do not use `--write-manifest` for normal verification; that is only for
the initial preparation of this contribution.

### Read data directly without making a working copy

```python
from backend.modules.breathing_detection.preprocessing import WireBreathDataset

dataset = WireBreathDataset("datasets/breathing_detection/WiReBreath")
sample = dataset[0]
print(len(dataset))                 # 1401 independent 10-second windows
print(sample.x.shape)               # (1, 234, 300), at 30 Hz
print(sample.assessable)            # timestamp and signal-quality decision
print(sample.quality_flags)
print(sample.respiratory_reference.shape)
```

Raw segments are approximately 30 seconds at a nominal 50 Hz: 1500 CSI points,
234 usable subcarriers, aligned respiratory reference, timestamps and metadata.
There are 5 participants and 85 recordings. Quiet has 240 segments; Interference
has 227. Room1 has 170; Room2 has 297. These are all breathing-present records,
not empty-room negatives. Do not assign artificial empty-room labels.

### Build the LSTM input cache

```bash
python -m backend.modules.breathing_detection.preprocessing.prepare_wire_breath --data-root datasets/breathing_detection/WiReBreath
```

This writes the ignored cache under
`backend/modules/breathing_detection/outputs/wire_breath_cache/`: ordered
`[300,232]` CSI sequences, aligned references, quality weights and grouped folds.
A source timestamp gap greater than 100 ms marks a window as not assessable.
Do not relax this rule just to increase coverage. Keep all windows from one
recording in the same split; follow the fixed person/recording development
protocol in the module, rather than randomly splitting neighboring windows.

### Run the shared experimental rate estimator on a 30-second segment

```python
from backend.modules.breathing_detection.preprocessing import WireBreathDataset
from backend.modules.breathing_detection.preprocessing.unified_rate import estimate_unified_rate

segments = WireBreathDataset(
    "datasets/breathing_detection/WiReBreath",
    window_seconds=30, stride_seconds=30, normalize=False,
)
sample = next(segments[i] for i in range(len(segments)) if segments[i].assessable)
result = estimate_unified_rate(sample.x)
print(result)  # experimental BPM, baseline BPM, rule ID and correction status
```

The rate estimator uses 30 seconds at 30 Hz (`[1,234,900]`), separately from
the 10-second LSTM detector. A valid sampling window is not a guarantee of an
accurate rate. Reliability rejection remains disabled. This direct example
illustrates rate estimation; the application additionally checks breathing
evidence before presenting a rate.

### Frozen detector evaluation requires its trained checkpoint

After building the cache, a supplied frozen checkpoint can be evaluated with:

```bash
python -m backend.modules.breathing_detection.evaluation.evaluate_wire_breath --checkpoint PATH_TO_BEST_MODEL_PT --cache-dir backend/modules/breathing_detection/outputs/wire_breath_cache --output backend/modules/breathing_detection/outputs/wire_reproduction.json --device cpu
```

Replace `PATH_TO_BEST_MODEL_PT` with the actual model file. **The checkpoint is
not included in this dataset contribution.** Adding raw data does not by itself
make every historical experiment reproducible: some audit scripts also require
earlier ignored experiment outputs. See the backend module reports for those
dependencies. The displayed dashboard already includes exported predictions;
it does not retrain the model when opened.

## 4. CSI-Bench: download separately and use the breathing task

Original download: [CSI-Bench on Kaggle](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench).
Author's code: [CSI-Bench repository](https://github.com/guozhen-jenn-zhu/CSI-Bench-Real-WiFi-Sensing-Benchmark).

Download the **BreathingDetection** task, retaining the HDF5 files, metadata and
official split files. Do not download unrelated tasks solely for this module.
The team's `main` dataset documentation targets Kaggle version 12; our original
local copy was not independently confirmed to be that release. Do not claim a
byte-identical reproduction across versions without checking the files.

The existing team downloader can fetch only this task into the path used by
our experiment scripts:

```bash
python -m pip install remotezip requests
python datasets/download_csi_bench.py --destination data --task BreathingDetection
python datasets/download_csi_bench.py --destination data --task BreathingDetection --verify-only
```

This downloader targets version 12 and checks the size and CRC32 of each file.
It requires network access. Alternatively, use the Kaggle page above.
The existing `datasets/breathing_detection/get_data.py` wrapper is also retained;
its default destination is `datasets/breathing_detection/data/BreathingDetection`,
so pass that task root explicitly to the adapter if you use the wrapper.
The committed CSI samples are format examples, not a full evaluation dataset.

Place the task root at `data/BreathingDetection/`. The correct root is the
directory containing its `metadata/`, `splits/` and source data directories,
not the parent CSI-Bench archive directory. Preserve relative paths from the
official split files. If downloaded to another location, pass that path using
`--data-root` or the Python constructor.

```python
from backend.modules.breathing_detection.preprocessing import BreathingDataset

test = BreathingDataset("data/BreathingDetection", "test_id")
sample = test[0]
print(len(test), sample.x.shape, sample.label, sample.assessable)
```

The project maps `empty` to 0 and `sleep` to 1 (breathing-present supervision).
The adapter preserves the original 300 time points and subcarrier order; the
detector's representation interpolates to 232 ordered input positions. This is
breathing-evidence detection, not a general occupancy classifier. CSI-Bench
10-second examples do not supply the 30-second window required by our rate
estimator or the respiratory reference used in WiRe-Breath.

See [the backend module README](../../backend/modules/breathing_detection/README.md)
for training commands, frozen-model results, preprocessing options and reports.
Do not select models or thresholds on the observed test set. Existing CSI-Bench
samples and download scripts are preserved from the team's dataset contribution.

## 5. Open the existing frontend experience

With Node.js 20+ installed:

```bash
npm run install:frontend
npm run dev:frontend
```

Open `http://localhost:5173/#/module/breathing_detection` (or the port printed by
Vite). Choose Breathing detection, Respiratory rate, or Fan interference.
The frontend uses bundled, backend-computed replay results. It works without
downloading CSI-Bench, installing the raw-data processing dependencies, or
loading a trained checkpoint. Adding raw MAT files does not automatically add
every record to the curated frontend selector.

## 6. Attribution and licenses

WiRe-Breath authors: **Yunpeng Ge, He Wang, and Ivan Wang-Hei Ho**.
*WiRe-Breath: A Sustainable WiFi-Based Real-Time Respiratory Monitoring Solution*,
IEEE Journal of Biomedical and Health Informatics, 2025.
[DOI: 10.1109/JBHI.2025.3619664](https://doi.org/10.1109/JBHI.2025.3619664).
Data are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
with the original [license text](WiReBreath/LICENSE), upstream documentation and
MAT files retained. See the upstream license for its disclaimer of warranties.
The authors do not endorse this project.

CSI-Bench remains the property of its authors and is distributed from its
original source under its published terms (the team's version-12 documentation
identifies CC BY-NC-ND 4.0). The WiRe license does not apply to CSI-Bench.
