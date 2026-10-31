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

## Get it

The archive is published as the release `wifi-space-20261031` of the project repository (11 parts). The branch's `fetch_data.py` downloads
the parts, checks every part and the whole archive by SHA256, and unpacks them. Run `fetch_data.py` inside `backend/modules/space_proximity_sensing/wifi space`; files are restored there at the paths expected by its models. Do not relocate the model-bound project to a different data layout.
