#!/usr/bin/env python3
"""Extract a subset of cells from a CDL netlist.

The backend (`placement`) iterates over **every** cell in the netlist. It has no
CLI option to filter by cell name, only a commented-out filter inside main.cpp.
Rather than patching the C++, hand it a small netlist containing just the cells
of interest: no patch is needed and the file records exactly what was run.

    # list the cells and their transistor counts
    python3 scripts/make_subset_netlist.py --list

    # build a toy subset
    python3 scripts/make_subset_netlist.py -o work/t0.sp \\
        INVx1 NAND2x1 NOR2x1 AOI21x1

    # select by transistor count
    python3 scripts/make_subset_netlist.py -o work/small.sp --max-tr 8

Cell names may be given with or without the `_ASAP7_75t_R` suffix.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_NETLIST = (
    Path(__file__).resolve().parent.parent
    / "third_party/autocellgen/DATA/input/asap7sc7p5t.sp"
)
SUFFIX_RE = re.compile(r"_ASAP7_\d+t_R$")


def parse_cells(path: Path) -> dict[str, tuple[str, int]]:
    """{cell name: (body, transistor count)}"""
    cells: dict[str, tuple[str, int]] = {}
    name: str | None = None
    buf: list[str] = []
    ntr = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith(".subckt"):
            name = stripped.split()[1]
            buf, ntr = [line], 0
        elif low.startswith(".ends"):
            if name:
                buf.append(line)
                cells[name] = ("\n".join(buf), ntr)
            name = None
        elif name is not None:
            buf.append(line)
            if stripped[:1].upper() == "M":
                ntr += 1
    return cells


def resolve(requested: str, cells: dict) -> str | None:
    """Accept the name with or without the suffix."""
    if requested in cells:
        return requested
    matches = [n for n in cells if SUFFIX_RE.sub("", n) == requested]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"ambiguous name {requested!r}: {sorted(matches)}", file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="extract a CDL netlist subset")
    ap.add_argument("cells", nargs="*", help="cell names (suffix optional)")
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_NETLIST)
    ap.add_argument("-o", "--output", type=Path, help="output .sp path")
    ap.add_argument("--list", action="store_true", help="print cells and TR")
    ap.add_argument("--max-tr", type=int, help="select by netlist TR upper bound")
    ap.add_argument("--min-tr", type=int, help="select by netlist TR lower bound")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"netlist not found: {args.input}", file=sys.stderr)
        print("  git submodule update --init third_party/autocellgen", file=sys.stderr)
        return 1

    cells = parse_cells(args.input)

    if args.list:
        for n, (_, tr) in sorted(cells.items(), key=lambda kv: (kv[1][1], kv[0])):
            print(f"{tr:4d}  {SUFFIX_RE.sub('', n)}")
        print(f"\n{len(cells)} cells", file=sys.stderr)
        return 0

    selected: list[str] = []
    if args.max_tr is not None or args.min_tr is not None:
        lo = args.min_tr if args.min_tr is not None else 0
        hi = args.max_tr if args.max_tr is not None else 10**9
        selected = sorted(n for n, (_, tr) in cells.items() if lo <= tr <= hi)

    for c in args.cells:
        resolved = resolve(c, cells)
        if resolved is None:
            print(f"cell not found: {c}", file=sys.stderr)
            return 1
        if resolved not in selected:
            selected.append(resolved)

    if not selected:
        print("no cells selected; give names or use --min-tr/--max-tr", file=sys.stderr)
        return 1

    if not args.output:
        print("-o/--output is required", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(cells[n][0] for n in selected)
    args.output.write_text(body + "\n")

    print(f"{len(selected)} cells -> {args.output}")
    for n in selected:
        print(f"   {SUFFIX_RE.sub('', n):24s} TR={cells[n][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
