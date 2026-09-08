#!/usr/bin/env python3
"""Measure the candidate pool size per cell by running placement only.

Labelling is expensive because each candidate costs one router invocation. But
"how many candidates does this cell produce" is answered by placement alone, and
that number decides whether a cell can go into the dataset at all: a cell with
one candidate has nothing to rank.

With `ROUTE_SOL 0` the routing loop in `main.cpp` never runs. The placement
output (`<cell>_w<N>.txt`) is written inside `Placer::run()` and therefore
survives.

    python3 scripts/place_pools.py --max-tr 14 --out work/place_pools/in_dist.jsonl
    python3 scripts/place_pools.py --min-tr 16 --max-tr 20 --style configs/place_only_rx2.style \\
        --out work/place_pools/tr16_20_rx2.jsonl
    python3 scripts/place_pools.py FAx1 DFFHQNx1 --out work/place_pools/cross.jsonl

Records accumulate in JSONL as each cell finishes, so an interrupted run keeps
what it had, and pointing at the same file again skips the finished cells.

WARNING: pool size is a property of the configuration. `NUM_SOL` caps it and
   `RELAXATION` sets how many width tiers are produced, so both are recorded on
   every row, because numbers from different configurations must not share a table.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "third_party/autocellgen/MAKE/PLACE_ROUTE/csyn_fp/build/placement"
Z3LIB = ROOT / "third_party/z3-prebuilt/z3-4.8.11-x64-glibc-2.31/bin"
NETLIST = ROOT / "third_party/autocellgen/DATA/input/asap7sc7p5t.sp"

sys.path.insert(0, str(ROOT / "scripts"))

SOL_RE = re.compile(r"^-------- Solution", re.M)
CPP_RE = re.compile(r"^Min #CPP = (\d+)$", re.M)
PLACE_MS_RE = re.compile(r"^Placement time : (\d+)ms$", re.M)
WIDTH_RE = re.compile(r"_w(\d+)\.txt$")
DEV_RE = re.compile(r"(?:NMOS|PMOS) : (?!dummy)\S+\(")
COL_RE = re.compile(r"^\[Column", re.M)
NFIN_RE = re.compile(r"nfin=(\d+)")


def min_columns(nfin: int, max_fin: int) -> int:
    """The minimum number of columns one transistor occupies.

    `PlaceUnit.cpp:343` splits `nfin` into columns using a divisor no larger than
    `max_fin`. The fewest columns comes from the largest such divisor. With no
    divisor (a prime) it falls back to fin=1 and takes `nfin` columns, so `nfin=5`
    with `max_fin=3` gives 5 columns.
    """
    for fin in range(min(max_fin, nfin), 0, -1):
        if nfin % fin == 0:
            return nfin // fin
    return nfin


def folded_tr(subckt_body: str, max_fin: int = 3) -> int:
    """Post-folding device count. The paper's TR, determined by the netlist.

    `INVx8` has netlist TR=2, but each transistor has `nfin=24` and folds into 8
    columns of 3 fins, giving 16 devices, exactly the 16 in paper Table 1.

    WARNING: do not count a single placement solution. `FOLDING_STYLE dynamic`
    also emits more finely folded shapes (an `nfin=2` as two 1-fin columns), so
    the device count differs between solutions: of `HAxp5`'s 42 solutions, 36
    have 10 and 6 have 11. A cell's TR is the value at the coarsest folding, and
    that value matches this formula on all 180 cells.
    """
    return sum(min_columns(int(n), max_fin) for n in NFIN_RE.findall(subckt_body))


def folded_tr_map(max_fin: int = 3) -> dict[str, int]:
    """{subckt name: folded TR} for the whole netlist at once."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_subset_netlist import parse_cells
    return {name: folded_tr(body, max_fin)
            for name, (body, _tr) in parse_cells(NETLIST).items()}


def devices_in_first_solution(placement_file: Path) -> tuple[int, int] | None:
    """(device count, column count) of the first solution; validates `folded_tr`."""
    blocks = placement_file.read_text(errors="replace").split("-------- Solution ")
    if len(blocks) < 2:
        return None
    return len(DEV_RE.findall(blocks[1])), len(COL_RE.findall(blocks[1]))


def style_knobs(style: Path) -> dict:
    knobs = {}
    for line in style.read_text(errors="replace").splitlines():
        t = line.split("#")[0].split()
        if len(t) == 2 and t[0] in ("NUM_SOL", "ROUTE_SOL", "RELAXATION", "ROUTE_ACCEPT",
                                    "NMOS_MAX_FIN", "PMOS_MAX_FIN"):
            knobs[t[0].lower()] = int(t[1])
    return knobs


def place_one(cell: str, workdir: Path, style: Path, timeout: int) -> dict:
    from make_subset_netlist import parse_cells, resolve

    cells = parse_cells(NETLIST)
    full = resolve(cell, cells)
    if full is None:
        return {"cell": cell, "status": "not_in_netlist"}

    # One directory per configuration: running the same cell under a different
    # configuration would otherwise overwrite the placement files, leaving no way
    # to tell which configuration produced them.
    d = workdir / style.stem / cell
    pdir = d / "placement"
    pdir.mkdir(parents=True, exist_ok=True)
    for f in pdir.glob("*.txt"):          # so leftovers from an earlier run do not mix in
        f.unlink()
    sp = d / "cell.sp"
    sp.write_text(cells[full][0] + "\n")

    t0 = time.monotonic()
    status = "ok"
    try:
        subprocess.run([str(BIN), "-i", str(sp), "-d", str(style), "-o", str(pdir)],
                       env={"LD_LIBRARY_PATH": str(Z3LIB), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        status = "timeout"
    wall = time.monotonic() - t0

    pools: dict[str, int] = {}
    for f in sorted(pdir.glob("*_w*.txt")):
        n = len(SOL_RE.findall(f.read_text(errors="replace")))
        if n:
            m = WIDTH_RE.search(f.name)
            pools[m.group(1) if m else f.name] = n

    rec = {"cell": cell, "subckt": full, "tr": cells[full][1], "status": status,
           "wall_s": round(wall, 2), "pools_by_width": pools,
           "pool_total": sum(pools.values()),
           "n_width_tiers": len(pools), "style": style.name}
    max_fin = max(style_knobs(style).get("nmos_max_fin", 3),
                  style_knobs(style).get("pmos_max_fin", 3))
    rec["tr_folded"] = folded_tr(cells[full][0], max_fin)
    if pools:
        rec["min_width"] = min(int(w) for w in pools)
        d = devices_in_first_solution(pdir / f"{full}_w{rec['min_width']}.txt")
        if d:
            rec["devices_first_sol"], rec["n_columns"] = d
    summary = pdir / "summary.txt"
    if summary.is_file():
        text = summary.read_text(errors="replace")
        if m := CPP_RE.search(text):
            rec["min_cpp"] = int(m.group(1))
            rec["cell_width"] = int(m.group(1)) - 2
        if m := PLACE_MS_RE.search(text):
            rec["place_ms"] = int(m.group(1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="measure pool sizes with placement only")
    ap.add_argument("cells", nargs="*")
    ap.add_argument("--min-tr", type=int)
    ap.add_argument("--max-tr", type=int)
    ap.add_argument("--style", type=Path, default=ROOT / "configs/place_only.style")
    ap.add_argument("--out", type=Path, required=True, help="JSONL result file")
    ap.add_argument("--workdir", type=Path, default=ROOT / "work/place_pools")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell cap in seconds")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    if not BIN.is_file():
        print(f"backend missing: {BIN}\n  ./scripts/build_backend.sh", file=sys.stderr)
        return 1

    knobs = style_knobs(args.style)
    if knobs.get("route_sol", 0) != 0:
        print(f"{args.style.name} has ROUTE_SOL={knobs.get('route_sol')}, so it is "
              "not placement-only and would route as well.", file=sys.stderr)
        return 1

    from make_subset_netlist import parse_cells, resolve

    catalog = parse_cells(NETLIST)
    cells = list(args.cells)
    if args.min_tr is not None or args.max_tr is not None:
        lo = args.min_tr if args.min_tr is not None else -1
        hi = args.max_tr if args.max_tr is not None else 10**9
        by_tr = sorted(((v[1], k) for k, v in catalog.items()), key=lambda x: (x[0], x[1]))
        cells += [name.replace("_ASAP7_75t_R", "") for tr, name in by_tr if lo <= tr <= hi]
    if not cells:
        print("name some cells or use --min-tr/--max-tr", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.out.is_file():
        for line in args.out.read_text().splitlines():
            if line.strip().startswith("{"):
                r = json.loads(line)
                if r.get("status") == "ok" and r.get("style") == args.style.name:
                    done[r["cell"]] = r
    todo = [c for c in cells if c not in done]
    if done:
        print(f"skipping {len(done)} finished cells ({args.out.name})")
    if not todo:
        print("everything is already done")
        return 0

    print(f"placement only, {len(todo)} cells -> {args.out}")
    print(f"  style {args.style.name} · NUM_SOL {knobs.get('num_sol')} · "
          f"RELAXATION {knobs.get('relaxation')} · jobs {args.jobs}\n")

    t_start = time.monotonic()
    rows: list[dict] = list(done.values())

    def run(cell: str) -> dict:
        r = place_one(cell, args.workdir, args.style, args.timeout)
        with args.out.open("a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows.append(r)
        mark = {"ok": "✓", "timeout": "⏱"}.get(r["status"], "✗")
        print(f"  {mark} {cell:<16} TR={r.get('tr','?'):>2} width {r.get('cell_width','?'):>2} "
              f"· candidates {r.get('pool_total',0):>5} "
              f"({r.get('n_width_tiers',0)} tier) · {r.get('wall_s','?')}s", flush=True)
        return r

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(run, todo))
    else:
        for c in todo:
            run(c)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\ndone {len(ok)}/{len(rows)} · {sum(r['pool_total'] for r in ok):,} candidates · "
          f"{(time.monotonic()-t_start)/60:.1f} min")

    single = [r for r in ok if r["pool_total"] <= 2]
    if single:
        print(f"\n{len(single)} cells have <=2 candidates - nothing to rank, unusable:")
        print("   " + ", ".join(sorted(r["cell"] for r in single)))
    print(f"\nresult: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
