# X-Synth

An open-source reimplementation of the MLCAD '26 paper
*X-Synth: A Fast Synthesis Framework for Cross-Scale Standard Cells via
Pin-Access-Aware Multi-Task Routability Prediction*.

The work the paper describes was done on an industrial flow. This repository
rebuilds the method on public tools and a public PDK, so that it can be released
and reproduced without carrying anything out of that flow. No placer, router,
cell library or rule deck from it is used here.

---

## What this is

The paper's contribution is the **predictor and ranker** between a placer and a
router. Those two are reimplemented here; the placer and router are replaced
with open-source equivalents.

```
placer ──> {c1 ... cN}  ──> [tile hints] ──> [Tile-Transformer] ──> [Pareto ranker] ──> router
                                                                         (top-K under Φ)
```

One intervention point: the backbone normally routes candidates in placer-cost
order, and this replaces that order with a prediction-driven Pareto ranking.

## Where it differs from the paper

The paper runs on an industrial flow. Everything below is a substitution, and the
numbers this produces are not comparable with the paper's.

| Item | Paper | Here |
|---|---|---|
| Placer / router | Industrial P&R flow | [AutoCellGen](https://github.com/The-OpenROAD-Project/AutoCellGen) (BSD-3) |
| Cell netlist | Industrial cell library | ASAP7 `asap7sc7p5t` (BSD-3) |
| Design rules | Industrial advanced-node deck | ASAP7-based rules |
| Pin-access criterion | Defined by that deck | Redefined on the open lattice: an access point counts when its M2 position is free |
| Multi-height cells | Evaluated | Implemented, but the backbone emits only single-height placements, so it is exercised by tests rather than by data |
| Comparison baselines | NVCell 2, LatticeGNN+PDA | Not reimplemented (neither is public). Compared against orderings computable from the labels: oracle, placer cost, width, random |

Training hyperparameters (loss weights, layer count, hidden size, heads,
dropout, optimizer, learning rate, epochs, batch size) were chosen here for this
backbone and this data. Nothing is carried over from the original setup.

## Paper to code

| Paper | Component | Where |
|---|---|---|
| §2.1 | Multi-candidate P/R interface | [`patches/autocellgen/`](patches/autocellgen/) (`--xsynth-route`) |
| §2.2 | Tile decomposition, `w=7`, `s=2` | [`tiles/decompose.py`](src/xsynth/tiles/decompose.py) |
| §2.2 | Per-tile routing hints | `--xsynth-tiles` in [`patches/autocellgen/`](patches/autocellgen/) |
| §2.3 | Tile-hinted heterogeneous graph | [`graph/build.py`](src/xsynth/graph/build.py) |
| §2.3 | Multi-height: stacked tile rows, up/down relations | same file, covered by `tests/test_graph.py` |
| §2.4 | Tile-Transformer, 3 heads + 2 auxiliary | [`model/tile_transformer.py`](src/xsynth/model/tile_transformer.py) |
| §2.4 | Loss: BCE for routability, MSE for M1/M2, weighted sum | `multitask_loss`, same file |
| §2.4 | Pin-access-aware label | [`backend/pin_access.py`](src/xsynth/backend/pin_access.py) |
| §2.5 | Pareto dominance and the ranking queue | [`rank/pareto.py`](src/xsynth/rank/pareto.py) |
| §2.5 | Queue consumption under a wall-clock budget Φ | `under_budget` in [`baselines/orderings.py`](src/xsynth/baselines/orderings.py) |
| §3.1 | Cell-disjoint split | `PRESETS` in [`scripts/label_batch.py`](scripts/label_batch.py) |
| §3.2 | `t1` / `t*` | [`baselines/orderings.py`](src/xsynth/baselines/orderings.py) |

`python3 -m pytest tests` covers the tile decomposition, the graph in single-
and multi-height form, the ranker and the budget rule against the properties
stated in the paper.

## Running it

### 0. Prerequisites

Python 3.10+, and `cmake`, `g++`, `curl` for the backend. Only the first line
needs root:

```bash
sudo apt-get install -y cmake g++ curl python3-venv

python3 -m venv .venv && . .venv/bin/activate       # Debian and Ubuntu refuse
pip install torch pytest                            # pip outside a venv (PEP 668)
```

### 1. Build the backend

```bash
git submodule update --init third_party/autocellgen
./scripts/build_backend.sh                          # fetch Z3 4.8.11, patch, build
```

The submodule is pinned to an upstream commit; every local change lives in
[`patches/autocellgen/`](patches/autocellgen/) and is applied with `git apply`,
which fails rather than building unpatched sources. Seconds to a few minutes,
depending on how fast the one-off 44 MB Z3 download runs; the backend itself is
13 source files. Z3 is taken as a prebuilt release binary, never compiled.

### 2. Generate the data

Routing labels are derived output and are not shipped. Cells come from the
presets in [`scripts/label_batch.py`](scripts/label_batch.py); no pool
measurement is needed first.

```bash
for p in train test; do
  python3 scripts/label_batch.py --preset "r0_$p" --out "work/run_r0_$p" \
      --style configs/label_r0.style --jobs 5 --z3-ms 300000 --resume
  python3 scripts/tile_batch.py --run "work/run_r0_$p" \
      --style configs/label_r0.style --jobs 5 --z3-ms 300000 --resume
done
```

The cross-scale split is capped. `SDFHx1` has a pool of a thousand placements
and routes none of them, and `DFFHQx4` routes 12%, so both are labelled to 200
candidates only. `configs/label_r0_cap200.style` is `label_r0.style` with
`ROUTE_SOL 200`. The tile pass has to be told the same cap, or it hints
candidates that carry no label and runs three to five times longer for nothing.

```bash
python3 scripts/label_batch.py NOR3x2 FAx1 DFFHQNx1 --out work/run_r0_cross \
    --style configs/label_r0.style --jobs 3 --z3-ms 300000 --resume
python3 scripts/label_batch.py SDFHx1 DFFHQx4 --out work/run_r0_cross \
    --style configs/label_r0_cap200.style --jobs 2 --z3-ms 300000 --resume

python3 scripts/tile_batch.py --run work/run_r0_cross --cells NOR3x2 FAx1 DFFHQNx1 \
    --style configs/label_r0.style --jobs 5 --z3-ms 300000 --resume
python3 scripts/tile_batch.py --run work/run_r0_cross --cells SDFHx1 DFFHQx4 \
    --max-k 200 --style configs/label_r0.style --jobs 5 --z3-ms 300000 --resume

cat work/run_r0_*/tiles.jsonl > work/tiles_r0.jsonl
```

Tens of CPU-hours, most of it in the cross-scale cells. Every batch script
resumes, so an interrupted run continues where it stopped. Three settings decide
whether the labels are correct rather than merely fast:

- `--z3-ms` is the per-candidate solver cap. It is wall-clock based, so raising
  `--jobs` effectively tightens it. Too low and "unroutable" becomes
  indistinguishable from "not solved in time". The 300 s used here is what the
  slowest tile in `SDFHx1` needs at five workers. The defaults (60 s in
  `label_batch.py`, 120 s in `tile_batch.py`) truncate that without any warning,
  producing a wrong label or a wrong hint.
- `--jobs > 1` inflates the reported routing time through CPU contention. Within
  a cell every candidate sees the same conditions, so the ordering comparison
  still holds, but absolute times are not comparable across configurations.
- the cap above is part of the dataset definition, not a shortcut: the sample
  counts in [`data/expected.json`](data/expected.json) assume it.

### 3. Train, rank, verify

```bash
./scripts/reproduce.sh
```

Tens of minutes on a CPU, and the model is small enough that this is the
expected way to run it. Arguments are passed through to
[`scripts/run_xsynth.py`](scripts/run_xsynth.py), so `./scripts/reproduce.sh
--device cuda` uses a GPU; the default `--device auto` takes one if it is there.
The script checks the run against [`data/expected.json`](data/expected.json),
which records what this implementation produces on this setup: split sizes and
cell counts exactly, `t*/oracle` within a loose bound, since routing times
depend on the machine.

What that comes out as:

| split | `cost` | `random` | `xsynth` |
|---|---|---|---|
| train | 27.04× | 20.67× | 20.37× |
| test | 30.47× | 27.28× | 15.89× |
| cross | 6.77× | 9.85× | 14.18× |

`t*/oracle` is cumulative router time to reach the lexicographic-best routed
candidate, over the same quantity under an oracle ordering. Lower is better.
`cost` is the placer's own congestion ordering; `random` is the mean over 200
shuffles. The oracle routes exactly one candidate, so this denominator is much
harsher than the baselines the paper reports against, and the two sets of
multiples are not comparable.

Routing is much easier on this lattice than in the paper's environment: 96.5% of
the training candidates carry a positive label, against an in-distribution
routable rate of 36-37% there. The routability head is fitted on a label that
barely varies, and the paper's cross-scale result is not reproduced here.

## License

BSD-3-Clause. Third-party notices are in [`NOTICE`](NOTICE).

## Citation

Please cite the original paper:

> Sehun Yu, Byungho Choi, Junbin Lee, Kijae Hong, and Younggwang Jung. 2026.
> X-Synth: A Fast Synthesis Framework for Cross-Scale Standard Cells via
> Pin-Access-Aware Multi-Task Routability Prediction. In *2026 ACM/IEEE
> International Symposium on Machine Learning for CAD (MLCAD '26)*. ACM,
> 7 pages. https://doi.org/10.1145/3831599.3840350

```bibtex
@inproceedings{xsynth_mlcad26,
  author    = {Yu, Sehun and Choi, Byungho and Lee, Junbin and Hong, Kijae
               and Jung, Younggwang},
  title     = {X-Synth: A Fast Synthesis Framework for Cross-Scale Standard Cells
               via Pin-Access-Aware Multi-Task Routability Prediction},
  booktitle = {2026 ACM/IEEE International Symposium on Machine Learning for CAD
               (MLCAD '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3831599.3840350}
}
```
