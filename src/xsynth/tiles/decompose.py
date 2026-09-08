"""Tile decomposition and hint vectors (paper §2.2).

A fixed-width window slides across the cell to produce an overlapping tile
sequence t_1, t_2, ... Because the window has the same shape regardless of cell
size, **a cell of any size becomes a tile sequence of the same shape**, which is
what lets one predictor handle small and large cells together.

Each tile carries the local routing outcome as a hint:

    h_i = (ỹ_route, ỹ_naive, m̃1, m̃2, v)

Generating the hints is the C++ backbone's job (`--xsynth-tiles`, patches
P1/P6). This module owns the window placement rule and the parsing/alignment of
the hints.

Hint generation cost scales with the tile count, so it only pays off when
`cellWidth >> w`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The values used in the paper's experiments
DEFAULT_W = 7
DEFAULT_S = 2


@dataclass(frozen=True)
class Tile:
    """One window. The `(row, col)` 2D indexing is kept even for single-height."""

    row: int                # cell row; always 0 for single-height
    col: int                # tile column index (0, 1, 2, ...)
    col0: int               # first device position covered
    width: int              # window width w

    @property
    def cols(self) -> range:
        return range(self.col0, self.col0 + self.width)

    def covers(self, pos: int) -> bool:
        return self.col0 <= pos < self.col0 + self.width


@dataclass(frozen=True)
class Hint:
    """A tile's local routing outcome. The paper's h_i."""

    y_route: bool           # pin-access-aware local success
    y_naive: bool           # pin-unaware local success
    m1: float               # M1 usage inside the window
    m2: float               # M2 usage inside the window
    valid: bool             # did this come from a usable result (the paper's v)
    runtime_ms: int = 0

    @classmethod
    def missing(cls) -> "Hint":
        """A tile with no hint: flagged v=0, values left neutral."""
        return cls(y_route=False, y_naive=False, m1=0.0, m2=0.0, valid=False)

    def features(self) -> list[float]:
        """The model-input vector. When valid=False the profile is gated to 0.

        This is the "valid-gated M2 profile" of paper Figure 5: numbers from an
        invalid hint must not blend in as if they were real.
        """
        g = 1.0 if self.valid else 0.0
        return [
            g * float(self.y_route),
            g * float(self.y_naive),
            g * self.m1,
            g * self.m2,
            float(self.valid),
        ]


FEATURE_DIM = 5


def tile_positions(cell_width: int, w: int = DEFAULT_W, s: int = DEFAULT_S,
                   n_rows: int = 1) -> list[Tile]:
    """Place the windows, row-major.

    Paper §2.2: adjacent tiles overlap, every device is covered, and **no window
    crosses the cell boundary.** A cell narrower than the window yields no tiles
    at all, and such candidates have to proceed without hints.

    A tile covers device positions in **one** cell row, so a multi-height cell
    repeats the same tile row for each of its stacked rows ("multi-height cells
    stack a few such rows"). The result is ordered row-major, which is the order
    `tiles_per_row` and the graph builder rely on.
    """
    if cell_width < w or w <= 0 or s <= 0 or n_rows <= 0:
        return []
    cols = list(range(0, cell_width - w + 1, s))
    return [Tile(row=r, col=i, col0=c, width=w)
            for r in range(n_rows)
            for i, c in enumerate(cols)]


def tiles_per_row(cell_width: int, w: int = DEFAULT_W, s: int = DEFAULT_S) -> int:
    """How many tile columns one cell row holds."""
    if cell_width < w or w <= 0 or s <= 0:
        return 0
    return len(range(0, cell_width - w + 1, s))


def coverage(cell_width: int, tiles: list[Tile]) -> list[int]:
    """How many tiles cover each device position. 0 means no hint reaches it."""
    cov = [0] * cell_width
    for t in tiles:
        for p in t.cols:
            if 0 <= p < cell_width:
                cov[p] += 1
    return cov


def uncovered(cell_width: int, tiles: list[Tile]) -> list[int]:
    """Positions no tile covers.

    With `w=7, s=2` a window may not cross the boundary, so **the right end of
    the cell can be left over** (e.g. width 10 gives tiles [0,7) and [2,9),
    leaving position 9 uncovered). The paper says "every device is covered",
    which only holds when `(cell_width - w) % s == 0`.
    """
    return [i for i, c in enumerate(coverage(cell_width, tiles)) if c == 0]


@dataclass
class TiledCandidate:
    """The tile sequence and hints for one candidate."""

    cell: str
    index: int
    cell_width: int
    tiles: list[Tile] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)

    @property
    def has_hints(self) -> bool:
        return bool(self.tiles) and any(h.valid for h in self.hints)

    def feature_matrix(self) -> list[list[float]]:
        return [h.features() for h in self.hints]


def from_jsonl(records: list[dict], cell_width: int,
               w: int = DEFAULT_W, s: int = DEFAULT_S) -> dict[int, TiledCandidate]:
    """Group `--xsynth-tiles` output by candidate.

    Candidates for which the backbone emitted no tile (`valid:false`,
    `tile:-1`) are kept as empty entries; dropping them silently would make
    candidates disappear from the dataset.
    """
    out: dict[int, TiledCandidate] = {}

    for r in records:
        k = r.get("k")
        if k is None:
            continue
        tc = out.setdefault(k, TiledCandidate(
            cell=r.get("cell", ""), index=k, cell_width=cell_width,
            tiles=tile_positions(cell_width, w, s)))

        ti = r.get("tile", -1)
        if ti < 0:                      # no tile, e.g. cellWidth < w
            continue
        while len(tc.hints) <= ti:
            tc.hints.append(Hint.missing())
        tc.hints[ti] = Hint(
            y_route=bool(r.get("y_route", False)),
            y_naive=bool(r.get("y_naive", False)),
            m1=float(r.get("m1", 0.0)),
            m2=float(r.get("m2", 0.0)),
            valid=bool(r.get("valid", False)),
            runtime_ms=int(r.get("runtime_ms", 0)),
        )

    # Align the hint count with the tile count.
    for tc in out.values():
        while len(tc.hints) < len(tc.tiles):
            tc.hints.append(Hint.missing())
    return out
