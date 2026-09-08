#!/usr/bin/env python3
"""X-Synth end to end: labels -> graph -> training -> ranking -> t1/t* comparison.

    python3 scripts/run_xsynth.py work/run_smoke [--epochs 30]

Follows the flow of paper Figure 2.

To use hints, extract them with the backbone first and pass `--tiles <jsonl>`:

    placement --xsynth-tiles <placement.txt> --cell <name> ... > tiles.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xsynth.backend.formats import parse_jsonl                    # noqa: E402

# The paper's cumulative wall-clock budgets Phi (§3.2): 5 and 30 minutes.
BUDGETS_MS = (5 * 60_000, 30 * 60_000)
from xsynth.baselines.orderings import Candidate, evaluate        # noqa: E402
from xsynth.flow.dataset import Sample, load_run, split_by_cell, stats  # noqa: E402
from xsynth.rank.pareto import Prediction, queue                  # noqa: E402
from xsynth.tiles.decompose import from_jsonl                     # noqa: E402


def load_tiles(path: Path, samples: list[Sample]) -> dict:
    """Key `--xsynth-tiles` JSONL by (cell, file_width, candidate).

    WARNING: the backbone restarts `k` at 0 for every placement file. With
    `RELAXATION > 0` one cell emits several width tiers, so `(cell, k)` alone
    does not identify a candidate. `file_width`, which `tile_batch.py` adds,
    is required. Without it we assume a single width and infer it from the
    samples (compatibility with older JSONL).
    """
    if not path or not path.is_file():
        return {}
    recs = parse_jsonl(path.read_text())
    fallback = {(s.cell, s.index): s.file_width for s in samples}
    widths = {(s.cell, s.file_width): s.cell_width for s in samples}

    out = {}
    keyed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in recs:
        cell, k = r.get("cell"), r.get("k")
        fw = r.get("file_width") or fallback.get((cell, k))
        if fw is None:
            continue
        keyed[(cell, fw)].append(r)

    for (cell, fw), rs in keyed.items():
        cw = widths.get((cell, fw))
        if cw is None:
            continue
        for k, tc in from_jsonl(rs, cell_width=cw).items():
            out[(cell, fw, k)] = tc.hints
    return out


def bare(cell: str) -> str:
    """Sample cell names are subckts (`NOR3x2_ASAP7_75t_R`); presets are bare."""
    return cell.replace("_ASAP7_75t_R", "")


def by_preset(samples: list, prefix: str) -> dict[str, list]:
    """Split by the `<prefix>_train/_test/_cross` presets of `label_batch.py`.

    Unlike the width heuristic (`split_by_cell`) this uses explicit cell lists:
    the boundary is folded TR, which is not monotone in width."""
    from label_batch import PRESETS
    out = {}
    for part in ("train", "test", "cross"):
        want = set(PRESETS.get(f"{prefix}_{part}", []))
        out[part] = [s for s in samples if bare(s.cell) in want]
        missing = want - {bare(s.cell) for s in samples}
        if missing:
            print(f"  warning: {prefix}_{part} has no samples for {sorted(missing)}")
    return out


def _group(recs, key):
    out = defaultdict(list)
    for r in recs:
        out[r.get(key)].append(r)
    return out


def train(samples: list[Sample], epochs: int, hidden: int, seed: int,
          layers: int = 0, device: str = "cpu"):
    import torch

    from xsynth.model.encode import encode, targets
    from xsynth.model.tile_transformer import (
        ModelConfig, TileTransformer, multitask_loss,
    )

    torch.manual_seed(seed)
    # The paper does not give a value for the layer count L ("Stacking L such
    # layers"), so follow the `ModelConfig` default rather than overriding it.
    cfg = ModelConfig(hidden=hidden, heads=4)
    if layers:
        cfg = ModelConfig(hidden=hidden, heads=4, layers=layers)
    model = TileTransformer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    encoded = []
    for s in samples:
        g = s.graph()
        f, e = encode(g, device)
        t = targets(y_route=s.y_route, m1=s.label.m1, m2=s.label.m2,
                    y_naive=s.y_naive, hc=s.hc_target(), device=device)
        encoded.append((f, e, t, g.tile_shape))

    model.train()
    first = last = float("nan")
    for ep in range(epochs):
        total = 0.0
        for f, e, t, sh in encoded:
            opt.zero_grad()
            loss = multitask_loss(model(f, e, sh), t)
            loss.backward()
            opt.step()
            total += loss.item()
        last = total / max(len(encoded), 1)
        if ep == 0:
            first = last
    return model, first, last


def main() -> int:
    ap = argparse.ArgumentParser(description="X-Synth end-to-end")
    ap.add_argument("run_dir", type=Path, nargs="+",
                    help="labelling run directories; several are merged")
    ap.add_argument("--tiles", type=Path, help="--xsynth-tiles JSONL")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=0,
                    help="transformer layer count; 0 uses the ModelConfig default")
    ap.add_argument("--k-access", type=int, default=1)
    ap.add_argument("--no-free-m2", action="store_true",
                    help="drop the M2-occupancy part of the pin-access criterion, "
                         "leaving track count alone. The paper's y_route requires "
                         "the external pin-access constraints to hold, so this is "
                         "on by default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="torch device: auto (cuda when present), cpu, cuda, ...")
    ap.add_argument("--group-by", choices=["cell", "width"], default="cell",
                    help="ranking unit; `cell` merges the width tiers (default)")
    ap.add_argument("--preset", help="preset prefix from `label_batch.py` (e.g. r0), "
                                     "splitting by <p>_train/<p>_test/<p>_cross; "
                                     "without it the width heuristic is used")
    ap.add_argument("--extra-labels", type=Path, nargs="*", default=[],
                    help="`labels.jsonl` from runs that hold labels but no "
                         "placement (`route_shards.py` output); without it those "
                         "candidates are dropped silently")
    args = ap.parse_args()

    samples, seen = [], set()
    for d in args.run_dir:
        got = load_run(d, k_access=args.k_access,
                       extra_labels=args.extra_labels,
                       require_free_m2=not args.no_free_m2)
        for s in got:
            key = (s.cell, s.file_width, s.index)
            if key not in seen:        # on overlap keep the first run read
                seen.add(key)
                samples.append(s)
        print(f"  {d}: {len(got):,} samples")
    if not samples:
        print("no samples found", file=sys.stderr)
        return 1

    if args.tiles:
        tiles = load_tiles(args.tiles, samples)
        for s in samples:
            s.hints = tiles.get((s.cell, s.file_width, s.index), [])

    st = stats(samples)
    print("=" * 70)
    print(f"samples {st['n']} · cells {st['cells']} · "
          f"y_route {st['y_route']}/{st['n']} ({st['y_route_rate']:.1%})")
    print(f"with hints {st['with_hints']}  ·  contrastive (y_naive != y_route) {st['contrastive']}")
    if st["contrastive"] == 0:
        print("  note: no contrastive signal; the auxiliary naive head has nothing to learn")
    print("=" * 70)

    if args.preset:
        parts = by_preset(samples, args.preset)
        print(f"\nsplit ({args.preset}_* presets, cell-disjoint): " + " · ".join(
            f"{k} {len(v):,}/{len({bare(s.cell) for s in v})} cells"
            for k, v in parts.items()))
        train_set = parts["train"]
        if not train_set:
            print(f"no samples for {args.preset}_train", file=sys.stderr)
            return 1
    else:
        ind, cross = split_by_cell(samples)
        print(f"\nsplit (width heuristic): in-dist {len(ind)} · cross-scale {len(cross)}")
        parts = {"in-dist": ind, "cross-scale": cross}
        train_set = ind or samples

    import torch
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        print("--device cuda: this torch build reports no CUDA device; "
              "use --device cpu or install a CUDA build", file=sys.stderr)
        return 1

    print(f"\ntraining: {len(train_set)} samples x {args.epochs} epochs "
          f"on {args.device}")
    model, first, last = train(train_set, args.epochs, args.hidden, args.seed,
                               args.layers, args.device)
    print(f"  loss {first:.4f} -> {last:.4f}")

    # ---- predict -> Pareto ranking -> baseline comparison ----
    from xsynth.model.encode import encode

    def _run_table(title: str, subset: list) -> None:
        print("\n" + "=" * 70)
        print(f"{title}: per-cell t* (ms, lower is better) · {len(subset):,} samples")
        print("=" * 70)
        print(f"  {'cell':28s} {'oracle':>8s} {'cost':>8s} {'random':>9s} {'xsynth':>8s}")
        print("  " + "-" * 66)

        # The ranking unit is **one cell**. Grouping by width tier makes width
        # constant inside a group, which kills the ranker's width objective
        # (§2.5, "do not buy routability with cell area") and wastes the width
        # axis that `RELAXATION > 0` opens up. `--group-by width` is the older
        # per-tier behavior.
        groups: dict[tuple, list[Sample]] = defaultdict(list)
        for s in subset:
            groups[(s.cell,) if args.group_by == "cell" else (s.cell, s.file_width)].append(s)

        rows = []
        for key, ss in sorted(groups.items()):
            cell = key[0]
            if len(ss) < 2:
                continue
            preds = []
            for s in ss:
                g = s.graph()
                f, e = encode(g, args.device)
                y, m1, m2 = model.predict(f, e, g.tile_shape)
                preds.append(Prediction(cid=s.index, y_route=y, m1=m1, m2=m2,
                                        width=s.cell_width))
            order = queue(preds)

            cands = [Candidate(cid=s.index, cost=s.label.cost, width=s.cell_width,
                               routable=s.y_route, m1=s.label.m1, m2=s.label.m2,
                               runtime_ms=s.label.runtime_ms) for s in ss]
            r = evaluate(cands, predicted=order, budgets_ms=BUDGETS_MS)
            if r["oracle"]["tstar"] is None:
                continue

            def f2(v):
                return f"{v:8.0f}" if v is not None else "       -"

            print(f"  {cell[:28]:28s} {f2(r['oracle']['tstar'])} {f2(r['cost']['tstar'])} "
                  f"{f2(r['random']['tstar'])} {f2(r['xsynth']['tstar'])}")
            rows.append(r)

        if rows:
            def tot(name):
                return sum(r[name]["tstar"] for r in rows if r[name]["tstar"] is not None)

            o, c, x = tot("oracle"), tot("cost"), tot("xsynth")
            print("  " + "-" * 66)
            print(f"  {'total':28s} {o:8.0f} {c:8.0f} {tot('random'):9.0f} {x:8.0f}")
            print()
            print(f"  t*  cost/oracle = {c/o:.2f}x   ·   xsynth/oracle = {x/o:.2f}x")

            # The paper reports t1 (first routable) alongside t*, so do the same.
            def tot1(name):
                return sum(r[name]["t1"] for r in rows if r[name]["t1"] is not None)

            o1, c1, x1 = tot1("oracle"), tot1("cost"), tot1("xsynth")
            if o1:
                print(f"  t1  cost/oracle = {c1/o1:.2f}x   ·   "
                      f"xsynth/oracle = {x1/o1:.2f}x")

            # Routable rate under the cumulative wall-clock budgets Phi (§3.2):
            # the share of cells for which the ordering reaches a routable
            # layout before the budget runs out.
            print()
            hdr = "  ".join(f"{b // 60000:>2}min" for b in BUDGETS_MS)
            print(f"  {'routable within':22s} {hdr}      (best within)")
            for name in ("oracle", "cost", "random", "xsynth"):
                got, bst = [], []
                for b in BUDGETS_MS:
                    ok = [r[name]["budget"][b]["routable"] for r in rows]
                    bb = [r[name]["budget"][b]["best"] for r in rows]
                    got.append(sum(ok) / len(ok))
                    bst.append(sum(bb) / len(bb))
                print(f"  {name:22s} " + "  ".join(f"{v:4.0%}" for v in got)
                      + "      " + " ".join(f"{v:4.0%}" for v in bst))
            print()
            # One machine-readable line per split, for scripts/verify_repro.py.
            print("  RESULT " + json.dumps({
                "split": title, "samples": len(subset),
                "cells": len({bare(s.cell) for s in subset}),
                "oracle_ms": round(o), "cost": round(c / o, 4),
                "random": round(tot("random") / o, 4), "xsynth": round(x / o, 4),
                "t1_oracle_ms": round(o1), "t1_cost": round(c1 / o1, 4) if o1 else None,
                "t1_xsynth": round(x1 / o1, 4) if o1 else None,
            }, sort_keys=True))
        else:
            print("  (no cell has two or more candidates; nothing to rank)")

    for _name, _sub in parts.items():
        if _sub:
            _run_table(_name, _sub)
    return 0



if __name__ == "__main__":
    sys.exit(main())
