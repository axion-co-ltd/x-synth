#!/usr/bin/env python3
"""Find candidates that hit the Z3 cap and re-judge them with a larger cap.

    python3 scripts/recheck_unknown.py --run work/run_train --limit-ms 120000   # inspect
    python3 scripts/recheck_unknown.py --run work/run_train --limit-ms 120000 \\
        --rerun --z3-ms 600000 --jobs 7                                          # re-run

Why this exists
-----------
`Unroutable` conflates two different things:

    unsat    proven unroutable; more time changes nothing
    unknown  the solver gave up; the label is provisional

Patch P7 records `solver`, but only on the `--xsynth-route` JSONL path; the
`summary.txt` that batch labelling writes has no such field. Instead, a
`Routing time` sitting at the cap means a timeout, the same test one has to use
anyway, since `z3::optimize` has no `reason_unknown()`.

`--rerun` re-runs only those candidates through `--xsynth-route`, which does
report `solver`, so the outcome is definite: `sat` flips the label, `unsat`
confirms the original, and another `unknown` means the cap must go higher still.

WARNING: candidates that succeeded near the cap are reported too. `z3::optimize`
   usually returns `unknown` when time runs out mid-optimisation, but a `sat`
   close to the cap may have had its metal minimisation cut short, leaving
   `m1`/`m2` short of optimal.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "third_party/autocellgen/MAKE/PLACE_ROUTE/csyn_fp/build/placement"
Z3LIB = ROOT / "third_party/z3-prebuilt/z3-4.8.11-x64-glibc-2.31/bin"
NETLIST = ROOT / "third_party/autocellgen/DATA/input/asap7sc7p5t.sp"

sys.path.insert(0, str(ROOT / "scripts"))

CELL_RE = re.compile(r"^\(\d+\) Cell name : (\S+)", re.M)
ROW_RE = re.compile(
    r"^Width (\d+), Solution (\d+) \(Cost = (\d+)\): (.+?), Routing time: (\d+)ms", re.M)


def scan_summary(run: Path) -> list[dict]:
    """summary.txt -> per candidate: (cell, file_width, k, verdict, runtime)."""
    f = run / "placement" / "summary.txt"
    if not f.is_file():
        return []
    rows, cell = [], None
    for line in f.read_text(errors="replace").splitlines():
        if m := CELL_RE.match(line):
            cell = m.group(1)
        elif (m := ROW_RE.match(line)) and cell:
            rows.append({"cell": cell, "file_width": int(m.group(1)),
                         "k": int(m.group(2)), "cost": int(m.group(3)),
                         "result": m.group(4), "runtime_ms": int(m.group(5))})
    return rows


def scan_tiles(run: Path) -> list[dict]:
    f = run / "tiles.jsonl"
    if not f.is_file():
        return []
    out = []
    for line in f.read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def rerun_cell(cell: str, fw: int, ks: list[int], run: Path,
               style: Path, z3_ms: int, out_jsonl: Path) -> int:
    from make_subset_netlist import parse_cells

    cells = parse_cells(NETLIST)
    if cell not in cells:
        return 0
    pf = run / "placement" / f"{cell}_w{fw}.txt"
    if not pf.is_file():
        return 0

    d = run / "recheck"
    d.mkdir(parents=True, exist_ok=True)
    sp = d / f"{cell}.sp"
    if not sp.is_file():
        sp.write_text(cells[cell][0] + "\n")

    proc = subprocess.run(
        [str(BIN), "--xsynth-route", str(pf), "--cell", cell,
         "-i", str(sp), "-d", str(style),
         "--order", ",".join(map(str, ks)), "--out", str(d / f"{cell}_w{fw}")],
        env={"LD_LIBRARY_PATH": str(Z3LIB), "XSYNTH_Z3_TIMEOUT_MS": str(z3_ms),
             "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True)

    n = 0
    with out_jsonl.open("a") as f:
        for line in (proc.stdout or "").splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                rec["file_width"] = fw
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="re-judge candidates that hit the cap")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--limit-ms", type=int, required=True,
                    help="the Z3 cap used by that run")
    ap.add_argument("--near", type=float, default=0.95,
                    help="flag runtimes at or above this fraction of the cap")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--z3-ms", type=int, default=600000, help="cap for the re-run")
    ap.add_argument("--style", type=Path, default=ROOT / "configs/label_r0.style")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    thr = args.limit_ms * args.near
    rows = scan_summary(args.run)
    if not rows:
        print(f"no labels: {args.run}/placement/summary.txt", file=sys.stderr)
        return 1

    hit = [r for r in rows if r["runtime_ms"] >= thr]
    unroutable = [r for r in hit if r["result"].startswith("Unroutable")]
    routable = [r for r in hit if not r["result"].startswith("Unroutable")]

    print(f"{len(rows):,} cell labels · {len(hit)} at or above {args.near:.0%} of the "
          f"{args.limit_ms}ms cap ({len(hit)/len(rows):.1%})")
    print(f"  Unroutable -> probably unknown: {len(unroutable)}")
    print(f"  routable   -> metal minimisation possibly cut short: {len(routable)}")

    tiles = scan_tiles(args.run)
    if tiles:
        # A tile runtime is the sum of the pin-aware and naive passes
        thit = [t for t in tiles if t.get("runtime_ms", 0) >= thr]
        print(f"\n{len(tiles):,} tiles · {len(thit)} reached the cap "
              f"({len(thit)/len(tiles):.1%})")

    by_cell: dict[tuple[str, int], list[int]] = defaultdict(list)
    for r in unroutable:
        by_cell[(r["cell"], r["file_width"])].append(r["k"])
    if by_cell:
        print("\nsuspected unknown, by cell:")
        for (c, fw), ks in sorted(by_cell.items(), key=lambda x: -len(x[1]))[:15]:
            print(f"  {c}_w{fw:<3} {len(ks):>5}")

    if not args.rerun:
        if hit:
            print(f"\nto re-run: --rerun --z3-ms {args.z3_ms}")
        else:
            print("\nno candidate hit the cap; every label is definite")
        return 0
    if not by_cell:
        print("\nnothing to re-run")
        return 0

    out_jsonl = args.run / "recheck.jsonl"
    print(f"\nre-running {sum(len(v) for v in by_cell.values())} candidates · "
          f"Z3 {args.z3_ms}ms · jobs {args.jobs} -> {out_jsonl}")
    t0 = time.monotonic()

    def go(item):
        (c, fw), ks = item
        n = rerun_cell(c, fw, sorted(ks), args.run, args.style, args.z3_ms, out_jsonl)
        print(f"  {c}_w{fw:<3} {n}/{len(ks)}", flush=True)
        return n

    items = sorted(by_cell.items())
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(go, items))
    else:
        for it in items:
            go(it)

    recs = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.startswith("{")]
    sat = sum(1 for r in recs if r.get("solver") == "sat")
    uns = sum(1 for r in recs if r.get("solver") == "unsat")
    unk = sum(1 for r in recs if r.get("solver") == "unknown")
    print(f"\n{len(recs)} resolved · sat {sat} (label flipped) · unsat {uns} · "
          f"unknown {unk} (cap still too low) · {(time.monotonic()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
