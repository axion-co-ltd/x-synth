#!/usr/bin/env python3
"""Label a list of cells, resumably.

Written for environments where the session can be cut off. Each cell's results
are appended to the destination and recorded in `manifest.json` as it finishes,
so `--resume` continues from the cells that are left.

    # local
    python3 scripts/label_batch.py --preset mid --out work/run_batch

    # writing results straight to mounted storage
    python3 scripts/label_batch.py --preset mid \\
        --out work/run_mid --resume

The output has the usual run layout, so `flow/dataset.load_run()` reads it directly:

    <out>/placement/summary.txt      labels for every cell, accumulated
    <out>/placement/<cell>_w<N>.txt
    <out>/route/, <out>/IOnet/
    <out>/manifest.json              which cells finished, the settings, timings

WARNING: with `--jobs > 1` the labels are still correct but `Routing time`
   inflates through CPU contention. Use `--jobs 1` for data that will feed t1/t*.
   The manifest records `jobs`, so runs can be told apart afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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

# Cell sets aimed at the width transition region.
PRESETS = {
    # Dataset 1 (superseded by the `r0_*` sets below, and kept only for
    # reference): ROUTE_ACCEPT 2, RELAXATION 1, split boundary folded TR 16.
    # The published styles all keep the upstream ROUTE_ACCEPT 12, so these
    # presets cannot be rerun at their original setting from this repository.
    # Selection criteria: in-distribution (folded TR < 16), width >= 7, and an
    # R1 pool of at least 3. Width 7 is the tile window size; narrower cells
    # produce no tile hints at all (the `cellWidth < tw` guard in
    # `xsynth_route.cpp`). The split is cell-disjoint and the test width
    # range (7-9) sits inside train's (7-11).
    "indist_train": [
        "OAI21x1", "AOI21x1", "AO221x1", "OR3x4", "OAI222xp33",
        "AO32x2", "MAJx2", "AOI222xp33", "OAI331xp33",           # width 7
        "NOR2x2", "XNOR2x2", "AO221x2", "NAND2x2", "AOI31xp67",  # width 8
        "AOI211x1",                                              # width 9
        "OA31x2",                                                # width 10
        "AOI221x1",                                              # width 11
    ],
    "indist_test": [
        "AO22x2", "OA22x2", "OR5x2", "AOI322xp5", "OAI322xp33",  # width 7
        "OAI31xp67", "XOR2x2",                                   # width 8
        "NOR3x1", "NAND3x1",                                     # width 9
    ],
    # cross-scale test set: one sequential and one combinational cell at folded TR 24
    "cross": ["DFFHQNx1", "FAx1"],

    # Dataset 2: ROUTE_ACCEPT 12, RELAXATION 0, split boundary folded TR 20.
    # The `indist_*` sets above use a different configuration; do not mix the
    # two, their success rates are not comparable.
    # Criteria: width >= 7 (the tile window), an R0 pool of at least 3 (so there
    # is something to rank), cell-disjoint, test widths inside train's, and every
    # function family present on both sides.
    # `DHL*` goes to train and `DLL*` to test: the two are the same circuit
    # differing only in clock phase, and splitting them this way puts unroutable
    # candidates on both sides.
    "r0_train": [
        "AOI21x1", "OAI21x1",                                    # width 7
        "NOR2x2", "XNOR2x2", "OAI31xp67",                        # width 8
        "AOI22x1", "XOR2x1", "DHLx1",                            # width 9
        "OA31x2", "NAND2x1p5", "DHLx2",                          # width 10
        "AOI221x1", "AO211x2", "OR2x6", "DHLx3",                 # width 11
        "AND5x2",                                                # width 14
    ],
    "r0_test": [
        "AO22x2",                                                # width 7
        "AOI211x1", "NAND3x1", "NOR3x1", "AND2x4", "OAI22x1",
        "XNOR2x1", "DLLx1",                                      # width 9
        "NOR2x1p5", "DLLx2",                                     # width 10
        "AND3x4", "AND2x6", "DLLx3",                             # width 11
        "OA221x2",                                               # width 13
    ],
    # cross-scale. `NOR3x2` is included because it is the one cell in this band
    # with a large pool that routes throughout, keeping both classes present.
    "r0_cross": ["NOR3x2", "FAx1", "DFFHQNx1", "SDFHx1", "DFFHQx4"],

    "mid": [
        "AO22x1", "AOI22x1", "XOR2x1", "XNOR2x1", "AO221x1", "AOI221x1",
        "AO32x1", "AOI32x1", "AO321x1", "AO331x1", "AOI333xp33",
        "AND4x2", "AND5x2", "OA33x1", "OAI33x1", "AO222x1", "AOI222xp33",
        "INVx11", "INVx13", "BUFx12", "AND5x1", "OR5x1",
    ],
    "wide": [
        "DFFHQNx1", "DFFHQNx2", "DFFHQx4", "DFFLQNx1", "SDFHx1", "SDFLx1",
        "ASYNC_DFFHx1", "DHLx1", "DLLx1", "ICGx1",
    ],
}

CELL_RE = re.compile(r"^\((\d+)\) Cell name : (.+)$", re.M)


def env(z3_ms: int) -> dict:
    return dict(os.environ,
                LD_LIBRARY_PATH=str(Z3LIB),
                XSYNTH_LABEL_ALL="1",
                XSYNTH_Z3_TIMEOUT_MS=str(z3_ms))


def label_one(cell: str, out: Path, style: Path, z3_ms: int, timeout: int) -> dict:
    """Label one cell in a temporary directory, then merge it into the destination."""
    from make_subset_netlist import parse_cells, resolve

    cells = parse_cells(NETLIST)
    full = resolve(cell, cells)
    if full is None:
        return {"cell": cell, "status": "not_in_netlist"}

    tmp = out / "_tmp" / cell
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "placement").mkdir(parents=True, exist_ok=True)
    sp = tmp / "cell.sp"
    sp.write_text(cells[full][0] + "\n")

    t0 = time.monotonic()
    timed_out = False
    try:
        subprocess.run([str(BIN), "-i", str(sp), "-d", str(style),
                        "-o", str(tmp / "placement")],
                       env=env(z3_ms), capture_output=True, text=True,
                       timeout=timeout or None)
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = time.monotonic() - t0

    # Merge: summary is appended to, everything else is moved file by file.
    n_cand = 0
    summary = tmp / "placement" / "summary.txt"
    if summary.is_file():
        text = summary.read_text(errors="replace")
        n_cand = len(re.findall(r"^Width \d+, Solution", text, re.M))
        dst = out / "placement" / "summary.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("a") as f:
            f.write(text if text.endswith("\n") else text + "\n")

    for sub in ("placement", "route", "IOnet"):
        src = tmp / "placement" if sub == "placement" else tmp / sub
        if not src.is_dir():
            continue
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name != "summary.txt" and f.name != "cell.sp":
                shutil.move(str(f), str(out / sub / f.name))

    shutil.rmtree(tmp, ignore_errors=True)
    return {"cell": cell, "subckt": full, "status": "timeout" if timed_out else "ok",
            "wall_s": round(wall, 1), "candidates": n_cand,
            "tr": cells[full][1]}


def load_manifest(out: Path) -> dict:
    f = out / "manifest.json"
    if f.is_file():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    return {"cells": {}, "config": {}}


def save_manifest(out: Path, m: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="resumable batch labelling")
    ap.add_argument("cells", nargs="*")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--style", type=Path, default=ROOT / "configs/label_r0.style")
    ap.add_argument("--z3-ms", type=int, default=60000)
    ap.add_argument("--per-cell-timeout", type=int, default=0,
                    help="per-cell wall-clock cap in seconds; 0 means none. "
                         "Cutting here truncates the pool silently; the "
                         "per-candidate cap is --z3-ms")
    ap.add_argument("--jobs", type=int, default=1,
                    help="cells in parallel; >1 distorts Routing time")
    ap.add_argument("--resume", action="store_true", help="skip finished cells")
    args = ap.parse_args()

    cells = list(args.cells) + (PRESETS.get(args.preset, []) if args.preset else [])
    if not cells:
        print("name some cells or use --preset", file=sys.stderr)
        return 1
    if not BIN.is_file():
        print(f"backend missing: {BIN}\n  ./scripts/build_backend.sh", file=sys.stderr)
        return 1

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(out)
    # One output directory can be filled by several invocations with different
    # styles (the cross-scale split caps two cells at 200 candidates), so the
    # style is recorded per cell and each style is kept under its own name.
    man["config"] = {"style": args.style.name, "z3_ms": args.z3_ms,
                     "jobs": args.jobs, "per_cell_timeout": args.per_cell_timeout}
    shutil.copy(args.style, out / f"used_{args.style.stem}.style")

    todo = [c for c in cells
            if not (args.resume and man["cells"].get(c, {}).get("status") == "ok")]
    done = len(cells) - len(todo)
    if done:
        print(f"skipping {done} finished cells (--resume)")
    if not todo:
        print("everything is already done")
        return 0

    print(f"labelling {len(todo)} cells -> {out}")
    print(f"  style {args.style.name} · Z3 {args.z3_ms}ms · "
          f"per-cell cap {args.per_cell_timeout or 'none'} · jobs {args.jobs}")
    if args.jobs > 1:
        print("  warning: parallel run distorts Routing time; labels only")
    print()

    t_start = time.monotonic()

    def run(cell: str) -> dict:
        r = label_one(cell, out, args.style, args.z3_ms, args.per_cell_timeout)
        r["style"] = args.style.name
        man["cells"][cell] = r
        save_manifest(out, man)          # saved per cell, so a lost session keeps it
        mark = "⏱" if r["status"] == "timeout" else ("✗" if r["status"] != "ok" else "✓")
        print(f"  {mark} {cell:<14} {r.get('wall_s','?'):>7}s  "
              f"candidates {r.get('candidates',0):>5}", flush=True)
        return r

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(run, todo))
    else:
        for c in todo:
            run(c)

    ok = sum(1 for v in man["cells"].values() if v.get("status") == "ok")
    total = sum(v.get("candidates", 0) for v in man["cells"].values())
    print(f"\ndone {ok}/{len(man['cells'])} cells · {total} candidates · "
          f"{(time.monotonic()-t_start)/60:.1f} min")
    print(f"result: {out}")
    print(f"\nnext: python3 scripts/run_xsynth.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
