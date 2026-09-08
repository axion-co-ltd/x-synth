# Design-rule style files

Each file is a copy of the backbone's own rule file,
`third_party/autocellgen/DATA/input/placement_file.style`, with a few keys
changed. Nothing else is edited. The upstream header stays intact, authorship
notice included. AutoCellGen is BSD-3-Clause; see [`NOTICE`](../NOTICE).

| File | Changed from upstream | Used by |
|---|---|---|
| `place_only.style` | `GEN_GDS false`, `ROUTE_SOL 0` | `run_place_pools.sh` (R0): placement pools, no routing |
| `place_only_rx1.style` | the above + `RELAXATION 1` | `run_place_pools.sh` (R1) |
| `place_only_rx2.style` | the above + `RELAXATION 2` | `run_place_pools.sh` (R2) |
| `label_r0.style` | `ROUTE_SOL 1000` (GDS emission left on) | labelling and tile hints for the `r0_*` presets (the path README.md documents), and the default of `label_batch.py`, `tile_batch.py` and `recheck_unknown.py` |
| `label_r0_cap200.style` | `ROUTE_SOL 200` (GDS emission left on) | the two large sequential cells of the cross-scale split |
| `label_r1.style` | `ROUTE_SOL 1000`, `RELAXATION 1` (GDS emission left on) | the alternative width-relaxed dataset |

`ROUTE_SOL` is the per-cell candidate cap. Upstream stops at 5. A run left at
that cap succeeds quietly and produces 200x too few candidates, so no file here
keeps it.

`RELAXATION` admits placements wider than the minimum.

`GEN_GDS` stays at the upstream `true` in the labelling styles. It costs nothing
here. It writes a `.ascii` dump of the routed geometry, which nothing in this
repository reads: pin access and the M1/M2 counts come from
`route/<cell>_w<N>_<k>.txt`, written outside that guard. The GDS step after it
calls a jar through a relative path that does not resolve from the repository
root, so `java` fails and the failure is ignored. The placement-only styles turn
it off: a jar that did resolve would write `DEFAULT.gds` into the working
directory, and parallel workers would collide over it.
