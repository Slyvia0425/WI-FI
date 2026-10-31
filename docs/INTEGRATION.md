# Group 66 integration

The unified application combines `A_BRANCH` (already in `main`), `codex/bc-resnet18-hp-mixed`, `D-breathing`, and `wifi-space`. The owner approved publication of the integrated application to `main`.

The shared React shell retains Fang Meiheng's layout, typography, colours and room illustration. Home links to all five module workspaces and the fusion page. A–D retain their own experiments and recorded outputs. Module E opens the interactive static-object simulator through a same-origin proxy, without changing its model-bound source files.

## Start

```sh
npm ci --prefix frontend
npm run build
python scripts/serve_integrated.py
```

Open `http://127.0.0.1:5173/`. Windows users can also use `START_TEAM.bat` after building.

For Module E, first run `install.bat` in `backend/modules/space_proximity_sensing/wifi space`. This restores its checksum-verified model/data release and creates its Python environment. The integrated launcher starts this environment automatically. If models are absent, the four replay workspaces still run; the spatial page offers a reconnect control.

An existing full spatial installation may be selected with `WIFI_SPACE_ROOT`. An explicit interpreter may be selected with `WIFI_SPATIAL_PYTHON`. `--port` changes the website port; `--spatial-port` changes the private loopback model port. `npm run dev` uses the same spatial bridge with Vite's development proxy.

Model B/C assets remain available through the existing `scripts/restore_bc_models.py` command and `model_artifacts/manifest.json`. Recorded-result browsing does not execute those models or require their weights.

## Scope

- A: motion-source features, experiments, classifications and evaluation.
- B: released activity-model outputs and sample timelines.
- C: ResNet18 validation, B-confidence routing and archived TCS experiments.
- D: breathing replay, streaming controls and research views.
- E: object placement, transmitter controls, signal simulation and zone-count inference.
- Fusion: existing rule/interface demonstration, not a simultaneous multi-sensor experiment.

The home room animation is explicitly labelled illustrative. Module E does not estimate human proximity or feed simulated furniture locations into a human-monitoring recording. No joint five-module accuracy is reported. Source models, thresholds and existing evaluation results are unchanged.

## Spatial visual effects

The spatial workspace adds ambient particles, decorative TX/RX signal streams, hover lighting, button ripples and an effects toggle. The visual streams illustrate signal flow; they are not recovered propagation paths or model evidence. Animations follow the processing status, pause when the page is hidden or outside the viewport, and respect reduced-motion preferences. Local overlay assets live in `backend/api/assets/`; model-bound spatial source files, predictions and thresholds are unchanged.
