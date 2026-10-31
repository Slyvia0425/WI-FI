# Module: space and proximity sensing

| | |
|---|---|
| Data | simulated by the group; no CSI-Bench data and no recorded data |
| Source | our own generator on the module branch `wifi-space` (ray tracing with Sionna RT 1.2.2, seed 20261029) |
| Licence | produced by our own code |
| Size | about 11.5 GB (11 parts, SHA256 of the whole archive `ae9341a818090f5cbbbdcb66384df2c2bc5cc61f7892dc5e030103b2d0f3a231`) |
| Read by | `backend/modules/space_proximity_sensing/` |

## What it is

The active multi-object workspace uses a 6 by 6 by 3 m furnished room, four transmitters, eight receivers (32 links), and 64 frequency samples at 5.24 GHz with 20 MHz bandwidth. One to three wooden cuboids occupy a 3 by 3 zone grid; objects can share a centre zone if they remain physically separated. Small, medium and large size bands allow dimensions within ±8%. Default and T1 transmitter layouts have separate validated models.

The full archive preserves earlier single-object experiments as well as the active multi-object project. The root-level legacy two-transmitter/six-receiver configuration is not the active web application's configuration. The output is coarse zone counts, not measured furniture geometry or human proximity.

## Files

This folder belongs only to Module E (Chen Siyi). It contains no other member's measured CSI data.

```text
space_proximity_sensing/
  manifest.json          source, layout, counts and Release checksums
  file_index.csv.gz      all 179,064 data-file paths, sizes and CRC32 checksums
  get_data.py            download / restore / verify only this module's data
  sample/                six original held-out simulated NPZ files and labels
  data/                  complete restored data (Git-ignored)
    single_object/       14,096 files from earlier single-object experiments
    multi_object/        164,968 files, including active default and T1 data
```

The data files total **2,777,549,987 bytes**. The **11,515,006,241-byte full project archive** also includes trained models, outputs and source code; it is not the size of the signal data alone. The complete archive remains in the [wifi-space-20261031 Release](https://github.com/Slyvia0425/WI-FI/releases/tag/wifi-space-20261031). Raw data is distributed through those assets, rather than adding 179,064 files to Git history.

## Restore or verify

From the repository root, using Python 3.10+ (standard library only):

```bash
python datasets/space_proximity_sensing/get_data.py
python datasets/space_proximity_sensing/get_data.py --verify-only
```

The first command downloads and SHA256-checks the 11 existing Release parts and the combined archive, then extracts **only signal data** into this folder's `data/` directory. Each extracted file is checked against the committed file index by size and CRC32. It never writes into another module's dataset folder. A complete local archive can avoid downloading:

```bash
python datasets/space_proximity_sensing/get_data.py --archive "/path/to/wifi space.zip"
```

The six samples are for inspecting the format, not a complete training set. `sample/manifest.json` records their true object placements, split, layout and SHA256. The active default examples use spatial V6; the T1 examples use the matched T1 V3 data.

## Application paths

The application's existing model/source layout is preserved. In the current local installation, its two data paths are directory junctions to `data/single_object` and `data/multi_object` here, so the running application reads the organised data without changing any model-bound source files. On a fresh machine, the original `wifi space/install.bat` still restores the complete standalone project at its expected paths. The dedicated command above organises and verifies the dataset independently; it does not install ML dependencies or relocate trained models.
