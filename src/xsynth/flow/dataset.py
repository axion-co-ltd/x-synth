"""Labelling-run output -> training samples.

One sample = (target cell, one placement candidate), the same unit as paper §3.1.

    work/run_<name>/
      placement/summary.txt          labels
      placement/<cell>_w<N>.txt      candidate placements
      IOnet/<cell>_IOnet.txt         external pins
      route/<cell>_w<N>_<k>.txt      routing grid (for the pin-access check)

The split is **cell-disjoint**: candidates of the same cell must never straddle
two splits (paper §3.1). The TR criterion is the post-folding device count, so it
is read from the placement output rather than from the netlist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..backend import pin_access
from ..backend.formats import (
    Label, Placement, parse_ionet, parse_labels_jsonl, parse_placements,
    parse_route_grid, parse_route_profile, parse_summary,
)
from ..graph.build import build
from ..tiles.decompose import DEFAULT_S, DEFAULT_W, Hint, TiledCandidate


@dataclass
class Sample:
    cell: str
    index: int
    file_width: int
    placement: Placement
    label: Label
    io_nets: list[str]
    hints: list[Hint] = field(default_factory=list)
    y_route: bool = False        # pin-access-aware (after the checker)
    y_naive: bool = False        # the router's own success flag
    # Per-column M1 usage from the routing grid, the label for the paper's
    # auxiliary column-wise profile head (§2.4). Empty when the candidate did
    # not route, since there is no grid then.
    m1_profile: list[float] = field(default_factory=list)

    @property
    def cell_width(self) -> int:
        return self.placement.cell_width

    def graph(self, w: int = DEFAULT_W, s: int = DEFAULT_S):
        return build(self.placement, self.io_nets, self.hints, w=w, s=s)

    def hc_target(self, w: int = DEFAULT_W, s: int = DEFAULT_S) -> list[float]:
        """The column-wise M1 profile folded onto tiles (paper §2.4, aux head).

        The head emits one value per tile, so each tile takes the mean of the
        profile entries its window spans. Empty when the candidate did not route.
        """
        if not self.m1_profile:
            return []
        prof = self.m1_profile
        out = []
        for t in build(self.placement, self.io_nets, self.hints, w=w, s=s).tiles:
            seg = prof[t.col0:t.col0 + t.width - 1]
            out.append(sum(seg) / len(seg) if seg else 0.0)
        return out


def _cell_of(path: Path) -> str:
    return re.sub(r"_w\d+\.txt$", "", path.name)


def load_run(run_dir: Path, k_access: int = 1,
             tiles: dict[tuple[str, int, int], list[Hint]] | None = None,
             extra_labels: list[Path] | None = None,
             require_free_m2: bool = True) -> list[Sample]:
    """Read one labelling run into a list of samples.

    `tiles` maps {(cell, file_width, candidate): hints}. Without it the samples
    are built hint-free, which is the "w/o hints" configuration of paper Table 2
    row 3.

    `extra_labels` are `labels.jsonl` files **from other runs**. `route_shards.py`
    reuses an existing placement and writes only labels, so those runs have no
    `placement/summary.txt`. Pass the run that owns the placement as `run_dir`
    and its `labels.jsonl` here, or those candidates are dropped silently.
    """
    pdir, rdir, idir = run_dir / "placement", run_dir / "route", run_dir / "IOnet"
    summary = pdir / "summary.txt"
    if not summary.is_file():
        return []

    labels: dict[tuple[str, int, int], Label] = {
        (l.cell, l.file_width, l.index): l for l in parse_summary(summary)
    }
    for extra in extra_labels or []:
        labels.update(parse_labels_jsonl(extra))

    io_cache: dict[str, list[str]] = {}
    out: list[Sample] = []

    for pf in sorted(pdir.glob("*_w*.txt")):
        cell = _cell_of(pf)
        if cell not in io_cache:
            f = idir / f"{cell}_IOnet.txt"
            io_cache[cell] = parse_ionet(f) if f.is_file() else []

        for p in parse_placements(pf):
            key = (cell, p.file_width, p.index)
            lab = labels.get(key)
            if lab is None:
                continue

            y_naive = lab.routable
            y_route = y_naive
            rf = rdir / f"{cell}_w{p.file_width}_{p.index}.txt"

            # Recover the M1 usage that `summary.txt` omits from the routing
            # grid. Nothing is re-routed; the files are already there
            # (see `parse_route_grid`).
            if not lab.m1:
                m1 = parse_route_grid(rf, "M1")
                if m1 is not None:
                    lab = Label(**{**lab.__dict__, "m1": m1})
            profile = parse_route_profile(rf, "M1") or []
            if y_naive and rf.is_file() and io_cache[cell]:
                y_route, y_naive, _ = pin_access.label(
                    rf, io_cache[cell], routable=lab.routable, k=k_access,
                    require_free_m2=require_free_m2)

            out.append(Sample(
                cell=cell, index=p.index, file_width=p.file_width,
                placement=p, label=lab, io_nets=io_cache[cell],
                hints=(tiles or {}).get(key, []),
                y_route=y_route, y_naive=y_naive, m1_profile=profile,
            ))
    return out


def split_by_cell(samples: list[Sample], tr_threshold: int = 16
                  ) -> tuple[list[Sample], list[Sample]]:
    """Cell-disjoint split (paper §3.1).

    Uses cell width as a proxy for size, because this backbone only reveals the
    device count in the placement output and width is monotone in it. This is
    **not** the paper's TR criterion: the two disagree wherever devices fold.
    The preset lists exist to avoid that.

    The folded-TR boundary is not monotone in width, so prefer the explicit
    preset lists (`run_xsynth.py --preset`) when the split matters.

    Returns: (in-distribution, cross-scale)
    """
    by_cell: dict[str, int] = {}
    for s in samples:
        by_cell[s.cell] = max(by_cell.get(s.cell, 0), s.cell_width)

    small = {c for c, w in by_cell.items() if w < tr_threshold}
    ind = [s for s in samples if s.cell in small]
    cross = [s for s in samples if s.cell not in small]
    return ind, cross


def stats(samples: list[Sample]) -> dict:
    if not samples:
        return {"n": 0}
    n = len(samples)
    cells = {s.cell for s in samples}
    routable = sum(s.y_route for s in samples)
    naive = sum(s.y_naive for s in samples)
    return {
        "n": n,
        "cells": len(cells),
        "y_route": routable,
        "y_route_rate": routable / n,
        "y_naive": naive,
        "contrastive": naive - routable,   # candidates where the two labels differ
        "widths": sorted({s.cell_width for s in samples}),
        "with_hints": sum(bool(s.hints) for s in samples),
    }
