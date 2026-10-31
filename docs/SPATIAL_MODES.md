# Spatial run modes

The two modes share frozen inference code, models, thresholds and signal simulation. They differ in data access and replay scope. No model is retrained or substituted.

| | Light demo | Full research |
|---|---|---|
| Model / calibration files | Required | Required |
| Full signal dataset | Not required; access is blocked by the light runtime | Required for full replay and research scripts |
| Object placement and transmitter editor | Yes | Yes |
| Fresh ray-traced signal simulation | Yes | Yes |
| Uploaded signal inference | Yes | Yes |
| Included example scenes | 6, from both validated layouts | Full held-out catalog |
| Evaluation displayed in the UI | Previously saved evaluation | Previously saved evaluation |
| New full evaluation or training | Not included | Run the research scripts with complete data |

Light-mode examples are not a replacement for the held-out test set. Its displayed historical metrics retain the original experimental scope. Unsupported transmitter positions remain unvalidated in both modes.

## Local review

Start `python scripts/serve_integrated.py` to offer both profiles. The spatial page defaults to Light demo and has two mode cards. The same navigation supports `?mode=lite` or `?mode=research` inside the hash route.

- `START_LIGHT.bat`: light-only service.
- `START_RESEARCH.bat`: research-only service.
- `START_TEAM.bat`: both, where both sets of assets are installed.

The default internal ports are 8781 (light) and 8780 (research). The unified website is on port 5173.

## Light runtime assets

The light assets include only three required model bundles, their frozen metadata and six original example channels. Calibration references and model parameters are embedded in those unchanged bundles. They do not include full training/test arrays or intermediate experiment caches.

Create assets from an existing complete project:

```sh
python scripts/prepare_spatial_lite.py --from-project "/path/to/wifi space"
```

The result is `runtime_assets/spatial_lite/`. Its `bundle.json` records every payload file's size and SHA256 and the expected frozen source hashes. This directory is Git-ignored. To create or restore an asset-only ZIP:

```sh
python scripts/prepare_spatial_lite.py --from-project "/path/to/wifi space" --archive spatial-lite-assets.zip
python scripts/prepare_spatial_lite.py --restore spatial-lite-assets.zip
```

The local standalone light package includes the unified source, a prebuilt website and these assets. It still needs Python 3.12 and the ML/ray-tracing libraries. `INSTALL_LIGHT.bat` creates a virtual environment and installs the existing pinned requirements without downloading the full dataset. Then use `START_LIGHT.bat`. Node.js is unnecessary when using the prebuilt package.

## Research assets

The existing complete project, data layout and Release restoration remain available. The dataset-specific restore command is documented in `datasets/space_proximity_sensing/README.md`; the complete project restore remains in the spatial source folder. The full mode keeps the entire held-out replay catalog and original research scripts.

## Verification

`scripts/verify_spatial_modes.py` sends the exact same signal arrays to both services and compares all returned fields except compute timing. It covers both validated layouts, six example scenes, two noise levels and missing-signal rejection (14 cases). This is an equivalence check, not a new accuracy evaluation.

Download the [standalone light demo](https://github.com/Slyvia0425/WI-FI/releases/tag/spatial-light-20261031) for a prebuilt website, model assets and six examples. Extract it, run `INSTALL_LIGHT.bat` once, then run `START_LIGHT.bat`.
