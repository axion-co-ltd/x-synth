"""AutoCellGen I/O parsers.

Reads the four things the backbone produces:
  placement file   per-candidate transistor placement (`<cell>_w<N>.txt`)
  summary.txt      labels from a place-and-route run
  IOnet file       the cell's external pin list
  route file       routing-result grids (H/V net density, M1/M2 grid)

WARNING on index conventions: placement files count from `Solution 1` while
   summary.txt counts from `Solution 0`. This module normalizes **everything to
   0-based**.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SUFFIX_RE = re.compile(r"_ASAP7_\d+t_R$")

_SIDE_RE = re.compile(
    r"(NMOS|PMOS)\s*:\s*(\S+?)\((\d+)\)\s*\[([^\]]*)\]"
)
_SUMMARY_RE = re.compile(
    r"^Width (\d+), Solution (\d+) \(Cost = ([\d.]+)\): "
    r"(M1 routable|M2 routable \(Usage ([\d.]+)\)|Unroutable), "
    r"Routing time: (\d+)ms$"
)
_FEAT_RE = re.compile(r"(\w+)=([-\d.eE]+)")
_CELL_RE = re.compile(r"^\((\d+)\) Cell name : (.+)$")


@dataclass(frozen=True)
class Device:
    """One side (N or P) of a column. `dummy` marks an empty slot."""

    name: str
    left: str
    gate: str
    right: str
    nfin: int

    @property
    def is_dummy(self) -> bool:
        return self.name == "dummy"

    def terminals(self) -> tuple[str, str, str]:
        return (self.left, self.gate, self.right)


@dataclass
class Placement:
    """One placement candidate: the c_i of paper §2.1."""

    cell: str
    index: int                      # 0-based (same convention as summary.txt)
    file_width: int                 # the _w<N> in the filename; cellWidth + 2
    nmos: list[Device] = field(default_factory=list)
    pmos: list[Device] = field(default_factory=list)
    # Rows above the first, as (nmos, pmos) pairs. Paper §2.2: "multi-height
    # cells stack a few such rows". Empty for a single-height cell, which is
    # everything this backbone produces.
    upper_rows: list[tuple[list[Device], list[Device]]] = field(default_factory=list)

    @property
    def cell_width(self) -> int:
        return len(self.nmos)

    @property
    def n_rows(self) -> int:
        """Cell rows stacked vertically. 1 for single-height."""
        return 1 + len(self.upper_rows)

    @property
    def rows(self) -> list[tuple[list[Device], list[Device]]]:
        """Every cell row bottom-up, as (nmos, pmos)."""
        return [(self.nmos, self.pmos)] + list(self.upper_rows)

    def nets(self, col0: int = 0, w: int | None = None,
             row: int | None = None) -> list[str]:
        """Nets appearing in the [col0, col0+w) window.

        The whole width if `w` is omitted, and every row if `row` is omitted.
        Nets are cell-global, so a net touching any row belongs to the cell.
        """
        hi = self.cell_width if w is None else min(col0 + w, self.cell_width)
        rows = self.rows if row is None else [self.rows[row]]
        seen: list[str] = []
        for n, p in rows:
            for i in range(col0, hi):
                for d in (n[i], p[i]):
                    for t in d.terminals():
                        if t and t != "dummy" and t not in seen:
                            seen.append(t)
        return seen


@dataclass(frozen=True)
class Label:
    """One summary.txt entry. The router's actual verdict on a candidate."""

    cell: str
    file_width: int
    index: int
    cost: float
    routable: bool
    m2: float
    runtime_ms: int
    m1: float = 0.0        # absent from summary.txt; filled from the route grid or labels.jsonl
    feats: dict[str, float] = field(default_factory=dict)

    @property
    def m1_routable(self) -> bool:
        """Routed without using a single M2 track."""
        return self.routable and self.m2 == 0.0


def strip_suffix(name: str) -> str:
    return SUFFIX_RE.sub("", name)


def parse_placements(path: Path) -> list[Placement]:
    """Read a placement file into a 0-based candidate list."""
    cell = strip_suffix_from_filename(path)
    fw = int(m.group(1)) if (m := re.search(r"_w(\d+)\.txt$", path.name)) else 0

    out: list[Placement] = []
    cur: Placement | None = None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("-------- Solution"):
            # The file is 1-based; we renumber to 0-based.
            cur = Placement(cell=cell, index=len(out), file_width=fw)
            out.append(cur)
            continue
        if cur is None or not line.startswith("NMOS"):
            continue
        sides = {m.group(1): m for m in _SIDE_RE.finditer(line)}
        if "NMOS" not in sides or "PMOS" not in sides:
            continue
        for tag, target in (("NMOS", cur.nmos), ("PMOS", cur.pmos)):
            m = sides[tag]
            terms = m.group(4).split()
            if len(terms) != 3:
                continue
            target.append(Device(name=m.group(2), nfin=int(m.group(3)),
                                 left=terms[0], gate=terms[1], right=terms[2]))
    return [p for p in out if p.nmos]


def strip_suffix_from_filename(path: Path) -> str:
    """`AOI22x1_ASAP7_75t_R_w11.txt` → `AOI22x1_ASAP7_75t_R`."""
    return re.sub(r"_w\d+\.txt$", "", path.name)


def parse_summary(path: Path) -> list[Label]:
    """Every label in summary.txt, keyed by cell, width and index."""
    out: list[Label] = []
    cell = ""
    pending: int | None = None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if m := _CELL_RE.match(line):
            cell = m.group(2)
            continue
        if m := _SUMMARY_RE.match(line):
            routable = not m.group(4).startswith("Unroutable")
            out.append(Label(
                cell=cell,
                file_width=int(m.group(1)),
                index=int(m.group(2)),
                cost=float(m.group(3)),
                routable=routable,
                m2=float(m.group(5)) if m.group(5) else 0.0,
                runtime_ms=int(m.group(6)),
            ))
            pending = len(out) - 1
            continue
        if line.startswith("[") and pending is not None:
            feats = {k: float(v) for k, v in _FEAT_RE.findall(line)}
            out[pending] = Label(**{**out[pending].__dict__, "feats": feats})
            pending = None
    return out


def parse_ionet(path: Path) -> list[str]:
    """IOnet file -> external pin list. The first line is the cell name, skipped."""
    lines = [l.strip() for l in path.read_text(errors="replace").splitlines()]
    lines = [l for l in lines if l]
    return lines[1:] if lines else []


def parse_jsonl(text: str) -> list[dict]:
    """Output of the backbone's `--xsynth-route` / `--xsynth-tiles`.

    Non-JSON diagnostic output is interleaved, so only lines starting with `{`
    are taken.
    """
    import json

    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def parse_labels_jsonl(path: Path) -> dict[tuple[str, int, int], Label]:
    """Read the `labels.jsonl` written by `route_shards.py` into `Label`s.

    WARNING: the width unit differs from `summary.txt`. There, `Width` is
    `cellWidth + 2`; here, `width` is `cellWidth`. Placement filenames (`_w<N>`)
    use `cellWidth + 2`, so 2 is added here to line them up. Without that the
    same candidate ends up under two different keys and is dropped silently.
    """
    import json as _json

    out: dict[tuple[str, int, int], Label] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            r = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        cell = r["cell"]
        if not cell.endswith("_ASAP7_75t_R"):
            cell += "_ASAP7_75t_R"
        fw = int(r["width"]) + 2
        out[(cell, fw, int(r["k"]))] = Label(
            cell=cell, file_width=fw, index=int(r["k"]),
            cost=float(r.get("cost") or 0), routable=bool(r["routable"]),
            m2=float(r.get("m2") or 0), runtime_ms=int(r.get("runtime_ms") or 0),
        )
    return out


_GRID_TAGS = {"M1", "V1", "M2", "V2"}


def parse_route_grid(path: Path, tag: str = "M1") -> float | None:
    """Horizontal-track usage from the **last** `tag` grid in
    `route/<cell>_w<N>_<k>.txt`.

    Same formula as `Router.cpp:2077`: horizontal connections divided by
    ((columns - 1) * rows).

    **Why this is needed.** `summary.txt` carries no numeric M1 usage, only the
    `M1 routable` classification. That leaves `Label.m1` empty and kills one of
    the paper's three Pareto axes. The grid is already written to disk during
    routing, so the value can be **recovered without re-running anything**.

    Format: each line is fixed-width as `v0 sep0 v1 sep1 ... v(n-1)`, so the odd
    positions hold the horizontal connections (`-`). Vertical-connector rows
    (all spaces) sit between the metal rows, so **only lines starting with a
    digit** are counted; stopping at a blank line would read just the first row.

    Multiple blocks appear because M1/M2 minimisation calls the solver several
    times; the backbone reports the **last** call. The same computation applied
    to M2 reproduces the `Usage` field of `summary.txt`.
    """
    rows = _last_grid(path, tag)
    if not rows:
        return None
    cols = (len(rows[0]) + 1) // 2
    used = sum(1 for r in rows for x in range(1, len(r), 2) if r[x] == "-")
    res = (cols - 1) * len(rows)
    return used / res if res else 0.0


def _last_grid(path: Path, tag: str) -> list[str] | None:
    """Metal rows of the last `tag` block, or None."""
    if not path.is_file():
        return None
    lines = path.read_text(errors="replace").splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip() == tag]
    if not starts:
        return None
    rows = []
    for l in lines[starts[-1] + 1:]:
        if l.strip() in _GRID_TAGS:
            break
        if l.strip() and l.lstrip()[0].isdigit():
            rows.append(l)
    return rows or None


def parse_route_profile(path: Path, tag: str = "M1") -> list[float] | None:
    """Per-column horizontal-track usage from the last `tag` grid.

    `parse_route_grid` collapses the grid to one number. This keeps the column
    axis: entry `x` is the fraction of rows using the horizontal connection
    between columns `x` and `x+1`, so the result has `cols - 1` entries.

    This is the label for the paper's auxiliary column-wise M1 profile head
    (§2.4), which exists to preserve the local structure that the cell-level M1
    total washes out.
    """
    rows = _last_grid(path, tag)
    if not rows:
        return None
    cols = (len(rows[0]) + 1) // 2
    out = []
    for x in range(cols - 1):
        pos = 2 * x + 1
        hit = sum(1 for r in rows if pos < len(r) and r[pos] == "-")
        out.append(hit / len(rows))
    return out
