#!/usr/bin/env python3
"""Run patch (tile) routing over every candidate of a labelling run.

    python3 scripts/tile_batch.py --run work/run_train --jobs 7

`--xsynth-tiles` slices a candidate with a window of `w=7` and stride `s=2` and
routes each tile twice, pin-aware and pin-unaware. That output is the tile hint
of paper §2.2.

It has to run separately from cell routing (`label_batch.py`) because the
backend does not do both modes at once. The work is 2-6x the candidate count (a
width-11 cell gives 3 tiles per candidate, routed twice), so resuming is
essential: JSONL is written as each placement file finishes and a rerun skips it.

    <run>/tiles/<cell>_w<N>.jsonl   per placement file
    <run>/tiles.jsonl               all of them merged (read by run_xsynth.py --tiles)

WARNING: a cell narrower than the window has no tiles (`xsynth_route.cpp:230`);
it emits a single `valid:false` line. That is why the datasets are cut at
width >= 7.
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
WIDTH_RE = re.compile(r"_w(\d+)\.txt$")


def n_candidates(pf: Path) -> int:
    return len(SOL_RE.findall(pf.read_text(errors="replace")))


def done_ks(jsonl: Path) -> set[int]:
    """Candidate indices that already have results, including shard files
    (`<tag>.s<i>.jsonl`). Tolerates a truncated final line.
    """
    ks: set[int] = set()
    for f in [jsonl] + sorted(jsonl.parent.glob(f"{jsonl.stem}.s*.jsonl")):
        if not f.is_file():
            continue
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    ks.add(json.loads(line)["k"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ks


def no_gds_style(style: Path, run: Path) -> Path:
    """A copy of the style with `GEN_GDS false`.

    Left on, tile routing can stop silently: `generate_ascii()` has paths that
    call `exit(0)` when boundary tracing fails; patch P8 turns those into an
    early return, but only under `XSYNTH_NO_EXIT`. A tile is a partial layout
    and needs no
    geometry, so it is simply turned off here; the `.ascii` from cell routing is
    unaffected.
    """
    out = run / "tiles" / f"{style.stem}_nogds.style"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_file():
        out.write_text("\n".join(
            "GEN_GDS false" if l.startswith("GEN_GDS") else l
            for l in style.read_text().splitlines()) + "\n")
    return out


def n_use(pf: Path, max_k: int) -> int:
    """How many candidates to actually run.

    `--max-k` is for cells whose labelling was capped, so that the placement pool
    is larger than the label count. Without the cap, tiles would be generated for
    candidates that have no label."""
    n = n_candidates(pf)
    return min(n, max_k) if max_k else n


def run_one(pf: Path, run: Path, style: Path, tw: int, ts: int,
            timeout: int, resume: bool, z3_ms: int,
            shard: tuple[int, list[int]] | None = None, max_k: int = 0) -> dict:
    from make_subset_netlist import parse_cells

    subckt = re.sub(r"_w\d+\.txt$", "", pf.name)
    cells = parse_cells(NETLIST)
    if subckt not in cells:
        return {"file": pf.name, "status": "not_in_netlist"}

    tag = pf.name[:-4]                        # <subckt>_w<N>
    tiles_dir = run / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    total = n_use(pf, max_k)

    if shard is not None:                     # running a slice of the candidates
        idx, order = shard
        jsonl = tiles_dir / f"{tag}.s{idx}.jsonl"
        if not order:
            return {"file": f"{pf.name} #{idx}", "status": "skipped", "n": total}
    else:
        idx, order = None, None
        jsonl = tiles_dir / f"{tag}.jsonl"
        if max_k:                             # a cap means the range must be explicit
            order = list(range(total))
        if resume:
            have = done_ks(jsonl)
            if len(have) >= total:
                return {"file": pf.name, "status": "skipped", "n": total}
            if have:                          # continue with the remaining candidates
                order = [k for k in range(total) if k not in have]

    sp = tiles_dir / f"{subckt}.sp"
    if not sp.is_file():
        sp.write_text(cells[subckt][0] + "\n")

    cmd = [str(BIN), "--xsynth-tiles", str(pf), "--cell", subckt,
           "-i", str(sp), "-d", str(no_gds_style(style, run)),
           "--tile-w", str(tw), "--tile-s", str(ts),
           "--out", str(tiles_dir / tag)]
    if order is not None:
        cmd += ["--order", ",".join(map(str, order))]

    # WARNING: the backend restarts `k` at 0 for every placement file. When one
    # cell emits several width tiers (`RELAXATION > 0`), `(cell, k)` collides
    # across tiers. The backend output carries no width, so `file_width` is added
    # here; without it hints attach to the wrong candidate, silently.
    m = WIDTH_RE.search(pf.name)
    file_width = int(m.group(1)) if m else None

    t0 = time.monotonic()
    status = "ok"
    n_lines = 0
    try:
        with jsonl.open("a") as out:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True,
                                    env={"LD_LIBRARY_PATH": str(Z3LIB),
                                         "XSYNTH_Z3_TIMEOUT_MS": str(z3_ms),
                                         "PATH": "/usr/bin:/bin"})
            assert proc.stdout is not None
            for line in proc.stdout:
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["file_width"] = file_width
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_lines += 1
            proc.wait(timeout=timeout or None)
        # Lines already written survive a mid-run crash of the backend. Report a
        # failure when this worker did not cover all of its own candidates,
        # otherwise truncated data reads as complete. For a shard, compare
        # against that shard's share rather than the total.
        got = done_ks(tiles_dir / f"{tag}.jsonl")
        want = set(order) if order is not None else set(range(total))
        if n_lines and want - got:
            status = "partial"
    except subprocess.TimeoutExpired:
        proc.kill()
        status = "timeout"

    return {"file": pf.name, "status": status, "n": total,
            "tiles": n_lines, "wall_s": round(time.monotonic() - t0, 1)}


def merge(run: Path) -> int:
    """Merge the per-placement-file JSONL into <run>/tiles.jsonl."""
    # WARNING: `*_w*.jsonl` also matches a shard file (`..._w21.s0.jsonl`),
    #    because `*` absorbs `.s0`. Concatenating two globs would include shards
    #    twice, so the paths are de-duplicated through a set.
    parts = sorted(set((run / "tiles").glob("*_w*.jsonl")))
    n = 0
    with (run / "tiles.jsonl").open("w") as out:
        for p in parts:
            for line in p.read_text(errors="replace").splitlines():
                if line.startswith("{"):
                    out.write(line + "\n")
                    n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="patch (tile) routing for every candidate")
    ap.add_argument("--run", type=Path, required=True, help="a label_batch output directory")
    ap.add_argument("--style", type=Path, default=ROOT / "configs/label_r0.style")
    ap.add_argument("--tile-w", type=int, default=7)
    ap.add_argument("--tile-s", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=0,
                    help="per-placement-file wall-clock cap in seconds; 0 means "
                         "none. The per-candidate cap is --z3-ms, so cutting here "
                         "truncates the pool")
    ap.add_argument("--z3-ms", type=int, default=120000,
                    help="Z3 cap per tile in ms; the upstream default is 60 min")
    ap.add_argument("--resume", action="store_true", help="skip finished candidates")
    ap.add_argument("--cells", nargs="*", help="filter by cell name")
    ap.add_argument("--max-k", type=int, default=0,
                    help="run only the first N candidates (0 = all); needed for "
                         "cells whose labelling was capped")
    ap.add_argument("--shards", type=int, default=1,
                    help="split one placement file's candidates into N slices run "
                         "in parallel; keeps the cores busy when one large cell is "
                         "all that is left")
    args = ap.parse_args()

    if not BIN.is_file():
        print(f"backend missing: {BIN}\n  ./scripts/build_backend.sh",
              file=sys.stderr)
        return 1
    pdir = args.run / "placement"
    if not pdir.is_dir():
        print(f"no placement output: {pdir}", file=sys.stderr)
        return 1

    files = sorted(f for f in pdir.glob("*_w*.txt") if f.name != "summary.txt")
    if args.cells:
        want = {c.lower() for c in args.cells}
        files = [f for f in files
                 if re.sub(r"_ASAP7.*$", "", f.name).lower() in want]
    if not files:
        print("no placement files to run", file=sys.stderr)
        return 1

    total_cand = sum(n_use(f, args.max_k) for f in files)
    print(f"patch routing {len(files)} placement files · {total_cand:,} candidates -> {args.run}/tiles/")
    print(f"  window w={args.tile_w} s={args.tile_s} · style {args.style.name} · "
          f"Z3 {args.z3_ms}ms · jobs {args.jobs}\n")

    t0 = time.monotonic()
    rows = []

    def go(job) -> dict:
        pf, shard = job
        r = run_one(pf, args.run, args.style, args.tile_w, args.tile_s,
                    args.timeout, args.resume, args.z3_ms, shard, args.max_k)
        rows.append(r)
        mark = {"ok": "✓", "skipped": "·", "timeout": "⏱",
                "partial": "⚠"}.get(r["status"], "✗")
        print(f"  {mark} {r['file'][:40]:40s} candidates {r.get('n',0):>5} "
              f"tile rows {r.get('tiles',0):>6} {r.get('wall_s','-')}s", flush=True)
        return r

    # The (placement file, shard) list. With `--shards N` the remaining
    # candidates are split into N slices.
    jobs = []
    for f in files:
        if args.shards <= 1:
            jobs.append((f, None))
            continue
        tag = f.name[:-4]
        have = done_ks(args.run / "tiles" / f"{tag}.jsonl") if args.resume else set()
        left = [k for k in range(n_use(f, args.max_k)) if k not in have]
        if not left:
            jobs.append((f, None))            # reported as skipped
            continue
        for i in range(args.shards):
            jobs.append((f, (i, left[i::args.shards])))

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(go, jobs))
    else:
        for j in jobs:
            go(j)

    n = merge(args.run)
    ok = sum(1 for r in rows if r["status"] in ("ok", "skipped"))
    print(f"\ndone {ok}/{len(rows)} · {n:,} tile records · "
          f"{(time.monotonic()-t0)/60:.1f} min")
    print(f"result: {args.run}/tiles.jsonl")
    print(f"\nnext: python3 scripts/run_xsynth.py {args.run} --tiles {args.run}/tiles.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
