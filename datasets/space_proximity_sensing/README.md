# Module: space and proximity sensing

| | |
|---|---|
| Data | simulated by the group; no CSI-Bench data and no recorded data |
| Source | our own generator on the module branch `wifi-space` (ray tracing with Sionna RT 1.2.2, seed 20261029) |
| Licence | produced by our own code |
| Size | about 11.5 GB (11 parts, SHA256 of the whole archive `ae9341a818090f5cbbbdcb66384df2c2bc5cc61f7892dc5e030103b2d0f3a231`) |
| Read by | `backend/modules/space_proximity_sensing/` |

## What it is

A fixed 6 by 6 by 3 m room with one wooden cuboid (1.2 by 0.8 by 1.2 m), two transmitters and six receivers, 64 subcarriers at 5.24 GHz with
20 MHz bandwidth, 100,000 samples per source, zones on a 3 by 3 grid with jitter. The settings are in `config.json` of the branch.

## Get it

The archive is published as the release `wifi-space-20261031` of the project repository (11 parts). The branch's `fetch_data.py` downloads
the parts, checks every part and the whole archive by SHA256, and unpacks them. The owner of the module sends the data separately, to be
placed in `datasets/space_proximity_sensing/data`.
