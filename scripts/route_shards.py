#!/usr/bin/env python3
"""Route in parallel per candidate, so one cell does not occupy one core.

`label_batch.py` runs one process per cell, so a cell with a thousand candidates
stays on a single core to the end and the remaining cores idle once it is the
last one left. Here the unit of work is a `(cell, k)` pair instead, and any core
can pick up any cell.

    python3 scripts/route_shards.py SDFHx1 SDFLx3 --out work/run_r0_shard \\
        --style configs/label_r0.style --limit 200 --jobs 8 \\
        --skip-run work/run_r0_big

What this relies on
---------------
1. Placement is deterministic: the same cell under the same configuration
   produces a byte-identical placement file. So `k` names the same candidate
   across runs, and already-labelled `k` values can simply be skipped.
2. Patch P1 (`--xsynth-route ... --order`) routes an arbitrary subset of
   candidates and emits one JSON line per candidate as it finishes. Its `solver`
   field separates `unsat` from `unknown`, which the `summary.txt` path can only
   infer by comparing `Routing time` against the cap.

Why parallelism is safe here
-----------------------
| Resource | Conflict | Handling |
|---|---|---|
| result JSONL | several workers appending | only the parent writes; workers print to stdout and the parent appends under a lock |
| routing output | filename collisions | `<cell>_k<k>.txt` is unique per (cell, k) |
| working directory | the GDS jar writes `DEFAULT.gds` into the CWD | each worker gets its own temporary CWD |
| manifest | concurrent writes | only the parent writes |

WARNING: the Z3 `timeout` is wall-clock based, so raising the parallelism
   effectively tightens it. Raise `--z3-ms` along with `--jobs`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "third_party/autocellgen/MAKE/PLACE_ROUTE/csyn_fp/build/placement"
Z3LIB = ROOT / "third_party/z3-prebuilt/z3-4.8.11-x64-glibc-2.31/bin"
POOLS = ROOT / "work/place_pools/place_only"

SOLRE = re.compile(r"^-------- Solution \d+", re.M)
CELLRE = re.compile(r"^\(\d+\) Cell name : (\S+)")
ROWRE = re.compile(r"^Width \d+, Solution (\d+) \(Cost = \d+\): ")
FULLROW = re.compile(r"^Width (\d+), Solution (\d+) \(Cost = \d+\): (.+?), "
                     r"Routing time: (\d+)ms")


def placement_of(cell: str) -> tuple[Path, str, Path] | None:
    """(placement file, subckt name, netlist) as left behind by `place_pools`."""
    d = POOLS / cell / "placement"
    if not d.is_dir():
        return None
    files = sorted(d.glob(f"{cell}_*_w*.txt"))
    if not files:
        return None
    subckt = files[0].name.rsplit("_w", 1)[0]
    return files[0], subckt, POOLS / cell / "cell.sp"


def unknown_k(run: Path, unknown_ms: int) -> dict[str, set[tuple[int, int]]]:
    """Undecided (`unknown`) candidates as cell -> {(cellWidth, k)}.

    WARNING: the two sources use different units. `Width` in `summary.txt` is
    `cellWidth + 2`, and there `unknown` can only be inferred from a runtime
    sitting at the cap. `labels.jsonl` gives a definite value in `solver`.
    """
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    s = run / "placement" / "summary.txt"
    if s.is_file():
        cell = None
        for line in s.read_text(errors="replace").splitlines():
            if m := CELLRE.match(line):
                cell = m.group(1).replace("_ASAP7_75t_R", "")
            elif cell and (m := FULLROW.match(line)):
                if m.group(3).startswith("Unroutable") and int(m.group(4)) >= unknown_ms:
                    out[cell].add((int(m.group(1)) - 2, int(m.group(2))))
    j = run / "labels.jsonl"
    if j.is_file():
        for line in j.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r["routable"] and r.get("solver") == "unknown":
                    out[r["cell"]].add((int(r["width"]), int(r["k"])))
    return out


def labeled_k(run: Path) -> dict[str, set[int]]:
    """Already-labelled cell -> {k}, read from both summary.txt and labels.jsonl."""
    out: dict[str, set[int]] = defaultdict(set)
    s = run / "placement" / "summary.txt"
    if s.is_file():
        cell = None
        for line in s.read_text(errors="replace").splitlines():
            if m := CELLRE.match(line):
                cell = m.group(1)
            elif cell and (m := ROWRE.match(line)):
                out[cell].add(int(m.group(1)))
    j = run / "labels.jsonl"
    if j.is_file():
        for line in j.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r["cell"]].add(int(r["k"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="per-candidate parallel routing")
    ap.add_argument("cells", nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--style", type=Path, default=ROOT / "configs/label_r0.style")
    ap.add_argument("--z3-ms", type=int, default=300000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0,
                    help="per-cell candidate cap (k < limit); 0 means all")
    ap.add_argument("--chunk", type=int, default=4,
                    help="candidates routed per process launch; amortises startup "
                         "cost, but a large value loses more work on interruption")
    ap.add_argument("--skip-run", type=Path, action="append", default=[],
                    help="skip (cell, k) already present in these runs; repeatable")
    ap.add_argument("--recheck-run", type=Path, action="append", default=[],
                    help="re-judge only the unknown candidates of these runs; "
                         "--limit is ignored and existing labels are redone, so "
                         "raise --z3-ms when using it")
    ap.add_argument("--unknown-ms", type=int, default=285000,
                    help="runtime at or above which summary.txt counts as unknown")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BIN.is_file():
        print(f"backend missing: {BIN}", file=sys.stderr)
        return 1

    out = args.out
    (out / "route").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.style, out / "used.style")

    done: dict[str, set[int]] = defaultdict(set)
    for run in list(args.skip_run) + [out]:
        for cell, ks in labeled_k(Path(run)).items():
            done[cell.replace("_ASAP7_75t_R", "")] |= ks

    # Re-judge mode: the targets are the undecided candidates, not the remaining ones
    targets: dict[str, set[tuple[int, int]]] = defaultdict(set)
    if args.recheck_run:
        for run in args.recheck_run:
            for cell, wk in unknown_k(Path(run), args.unknown_ms).items():
                targets[cell] |= wk
        redone = labeled_k(out)          # already re-judged here, for resuming

    cells = args.cells or sorted(targets)
    if not cells:
        print("name some cells or pass --recheck-run", file=sys.stderr)
        return 1

    # The (cell, k) work list, interleaved across cells so no core piles onto one cell.
    per_cell: dict[str, list[int]] = {}
    meta: dict[str, tuple[Path, str, Path]] = {}
    for cell in cells:
        p = placement_of(cell)
        if p is None:
            print(f"  {cell}: no placement file ({POOLS/cell})", file=sys.stderr)
            continue
        meta[cell] = p
        n = len(SOLRE.findall(p[0].read_text(errors="replace")))
        cw = int(p[0].name.rsplit("_w", 1)[1].split(".")[0]) - 2   # filename holds cellWidth+2
        if args.recheck_run:
            # Only candidates at this file's width; other tiers are out of scope
            other = sorted({w for w, _ in targets[cell]} - {cw})
            todo = sorted(k for w, k in targets[cell]
                          if w == cw and k not in redone[cell])
            per_cell[cell] = todo
            print(f"  {cell:<14} unknown {len(targets[cell]):>4} · width {cw} targets "
                  f"{len(todo):>4}" + (f" · other widths skipped {other}" if other else ""))
        else:
            target = min(n, args.limit) if args.limit else n
            todo = [k for k in range(target) if k not in done[cell]]
            per_cell[cell] = todo
            print(f"  {cell:<14} candidates {n:>5} · target {target:>4} · "
                  f"done {len(done[cell]):>4} -> left {len(todo):>4}")

    # Cut into per-cell chunks, then interleave round-robin
    chunks: list[tuple[str, list[int]]] = []
    per_cell_chunks = {c: [ks[i:i + args.chunk] for i in range(0, len(ks), args.chunk)]
                       for c, ks in per_cell.items()}
    while any(per_cell_chunks.values()):
        for c in list(per_cell_chunks):
            if per_cell_chunks[c]:
                chunks.append((c, per_cell_chunks[c].pop(0)))
    total = sum(len(k) for _, k in chunks)
    print(f"\n{total} candidates · {len(chunks)} chunks · jobs {args.jobs} · "
          f"Z3 {args.z3_ms}ms · style {args.style.name}")
    if args.dry_run or not chunks:
        return 0

    q: Queue = Queue()
    for c in chunks:
        q.put(c)
    lock = threading.Lock()
    jsonl = out / "labels.jsonl"
    state = {"done": 0, "sat": 0, "unsat": 0, "unknown": 0, "t0": time.monotonic()}
    env = dict(os.environ, LD_LIBRARY_PATH=str(Z3LIB),
               XSYNTH_Z3_TIMEOUT_MS=str(args.z3_ms), XSYNTH_NO_EXIT="1")

    def worker(wid: int) -> None:
        cwd = out / "_w" / f"{wid}"          # a CWD per worker, isolating DEFAULT.gds
        cwd.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                cell, ks = q.get_nowait()
            except Exception:
                return
            place, subckt, sp = meta[cell]
            cmd = [str(BIN), "--xsynth-route", str(place), "--cell", subckt,
                   "-i", str(sp), "-d", str(args.style.resolve()),
                   "--order", ",".join(map(str, ks)),
                   "--out", str((out / "route").resolve())]
            try:
                p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
                for line in p.stdout:               # one line per finished candidate
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    r["cell"] = r["cell"].replace("_ASAP7_75t_R", "")
                    with lock:                      # only the parent writes files
                        with jsonl.open("a") as f:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        state["done"] += 1
                        state[r.get("solver", "unknown")] = \
                            state.get(r.get("solver", "unknown"), 0) + 1
                        d, el = state["done"], time.monotonic() - state["t0"]
                        eta = (total - d) * el / d / 3600 if d else 0
                        print(f"  [{d:>5}/{total}] {r['cell']:<13} k={r['k']:<4} "
                              f"{r.get('solver','?'):<7} {r['runtime_ms']:>7}ms  "
                              f"{eta:.1f}h left", flush=True)
                p.wait()
            except Exception as e:                  # one dead worker must not stop the rest
                print(f"  worker {wid} {cell}: {e}", file=sys.stderr, flush=True)
            finally:
                q.task_done()

    ts = [threading.Thread(target=worker, args=(i,), daemon=True)
          for i in range(args.jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    man = {"config": {"style": args.style.name, "z3_ms": args.z3_ms,
                      "jobs": args.jobs, "limit": args.limit, "chunk": args.chunk,
                      "mode": "candidate-sharded"},
           "cells": {c: len(k) for c, k in per_cell.items()},
           "result": {k: state[k] for k in ("done", "sat", "unsat", "unknown")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
    shutil.rmtree(out / "_w", ignore_errors=True)
    el = (time.monotonic() - state["t0"]) / 60
    print(f"\ndone {state['done']}/{total} · sat {state['sat']} · "
          f"unsat {state['unsat']} · unknown {state['unknown']} · {el:.1f} min")
    print(f"result: {jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
