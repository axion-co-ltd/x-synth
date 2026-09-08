#!/usr/bin/env python3
"""Compare a pipeline run against `data/expected.json`.

    python3 scripts/run_xsynth.py ... | tee run.log
    python3 scripts/verify_repro.py run.log

Exits non-zero if any check fails, so it can gate CI.

Sample and cell counts must match exactly, because they come from the data, not from
training. The oracle time is reported but not enforced. The t*/oracle ratios
are compared with the relative tolerance recorded in `expected.json`, because
training is only seed-stable on identical hardware and library versions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULT_RE = re.compile(r"^\s*RESULT (\{.*\})\s*$", re.M)


def main() -> int:
    ap = argparse.ArgumentParser(description="verify a reproduction run")
    ap.add_argument("log", type=Path, help="output of run_xsynth.py")
    ap.add_argument("--expected", type=Path, default=ROOT / "data/expected.json")
    args = ap.parse_args()

    exp = json.loads(args.expected.read_text())
    txt = args.log.read_text(errors="replace")

    got: dict[str, dict] = {}
    for line in RESULT_RE.findall(txt):
        r = json.loads(line)
        got[r["split"]] = r
    if not got:
        print("no RESULT lines in the log - is this run_xsynth.py output?",
              file=sys.stderr)
        return 1

    fails: list[str] = []

    print("deterministic - must match exactly")
    print(f"  {'split':6s} {'metric':9s} {'expected':>10s} {'got':>10s}  result")
    for name, e in exp["deterministic"].items():
        g = got.get(name)
        if g is None:
            fails.append(f"{name}: split missing from the log")
            print(f"  {name:6s} {'-':9s} {'':>10s} {'':>10s}  MISSING")
            continue
        for key, want in e.items():
            have = g.get(key)
            ok = have == want
            if not ok:
                fails.append(f"{name}.{key}: expected {want}, got {have}")
            print(f"  {name:6s} {key:9s} {want:>10,} {have or 0:>10,}  "
                  f"{'ok' if ok else 'FAIL'}")

    tol = exp["tolerance"]["ratio_rel"]
    print(f"\nt*/oracle - machine dependent, bounded at {tol:.0%}")
    print(f"  {'split':6s} {'metric':9s} {'reference':>10s} {'got':>10s}  result")
    for name, e in exp["reference"].items():
        g = got.get(name)
        if g is None:
            continue
        print(f"  {name:6s} {'oracle_ms':9s} {e['oracle_ms']:>10,} "
              f"{g['oracle_ms']:>10,}  (informational)")
        for key, want in e["ratio"].items():
            have = g[key]
            ok = abs(have - want) <= tol * want
            if not ok:
                fails.append(
                    f"{name}.{key}: reference {want:.2f}x +-{tol:.0%}, got {have:.2f}x")
            print(f"  {name:6s} {key:9s} {want:>9.2f}x {have:>9.2f}x  "
                  f"{'ok' if ok else 'FAIL'}")

    print()
    if fails:
        print(f"{len(fails)} check(s) failed:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
