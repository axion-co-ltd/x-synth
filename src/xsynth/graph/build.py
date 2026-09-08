"""Tile-hinted heterogeneous graph (paper §2.3).

One placement candidate is represented by three node types:

  device  a G/D terminal slot in the lower device-access region, one per
          placement column x {N, P} x {L, G, R}
  net     a cell-global net, preserving the circuit identity of nets that span
          several regions
  tile    a window from §2.2, carrying a hint vector and a (row, col) 2D
          positional encoding

Edges:
  tile->device   terminal slots the window covers
  tile->net      cell-global nets with a terminal inside the window
  tile<->tile    left/right (column direction only for SH; MH would add up/down)
  device<->net   the net a terminal belongs to

Built with plain Python data structures, no torch. Tensor conversion happens on
the model side so that graph construction is not tied to a framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backend.formats import Placement
from ..tiles.decompose import DEFAULT_S, DEFAULT_W, Hint, Tile, tile_positions

# device node feature dimension (see device_features below)
DEVICE_DIM = 9
NET_DIM = 5

POWER_NETS = frozenset({"VDD", "VSS", "VPP", "VBB", "GND"})

# terminal kinds
TERM_L, TERM_G, TERM_R = 0, 1, 2


@dataclass(frozen=True)
class DeviceNode:
    """One G/D terminal slot."""

    idx: int
    col: int                # placement column
    is_pmos: bool           # True on the P row (P/N row role)
    term: int               # TERM_L / TERM_G / TERM_R
    net: str
    nfin: int
    stack: int              # order within the column (stack index)
    is_dummy: bool
    is_io: bool             # I/O access mask - slot carrying an external cell pin
    row: int = 0            # cell row; 0 unless the cell is multi-height

    def features(self) -> list[float]:
        return [
            float(self.term == TERM_G),      # is it a gate
            float(self.term == TERM_L),
            float(self.term == TERM_R),
            float(self.is_pmos),             # P/N row role
            float(self.nfin),                # proxy for contact demand
            float(self.stack),               # stack index
            float(self.is_dummy),
            float(self.is_io),               # I/O access mask
            float(self.net in POWER_NETS),
        ]


@dataclass(frozen=True)
class NetNode:
    idx: int
    name: str
    degree: int             # number of terminals on this net
    span: int               # rightmost column - leftmost column (+1)
    is_io: bool
    is_power: bool

    def features(self) -> list[float]:
        return [
            float(self.degree),
            float(self.span),
            float(self.is_io),
            float(self.is_power),
            float(self.degree > 2),          # multi-fanout
        ]


@dataclass
class HeteroGraph:
    """The graph for one candidate, held as adjacency lists rather than tensors."""

    cell: str
    index: int
    cell_width: int

    devices: list[DeviceNode] = field(default_factory=list)
    nets: list[NetNode] = field(default_factory=list)
    tiles: list[Tile] = field(default_factory=list)
    hints: list[Hint] = field(default_factory=list)

    # (src, dst) index pairs
    e_tile_device: list[tuple[int, int]] = field(default_factory=list)
    e_tile_net: list[tuple[int, int]] = field(default_factory=list)
    e_tile_tile: list[tuple[int, int]] = field(default_factory=list)
    e_device_net: list[tuple[int, int]] = field(default_factory=list)

    @property
    def tile_shape(self) -> tuple[int, int]:
        """(cell rows, tile columns) of the tile grid, for the axis pools."""
        if not self.tiles:
            return (0, 0)
        return (max(t.row for t in self.tiles) + 1,
                max(t.col for t in self.tiles) + 1)

    def tile_pe(self) -> list[list[float]]:
        """2D positional encoding (row, col) for tiles, paper §2.3.

        Both axes are normalized by their extent so that cells of different
        sizes and heights share a scale. A single-height cell has one tile row,
        so the row component is 0 throughout.
        """
        ncol = max((t.col for t in self.tiles), default=0) + 1
        nrow = max((t.row for t in self.tiles), default=0) + 1
        return [[t.row / nrow if nrow > 1 else 0.0, t.col / ncol]
                for t in self.tiles]

    def summary(self) -> dict[str, int]:
        return {
            "device": len(self.devices), "net": len(self.nets),
            "tile": len(self.tiles),
            "tile-device": len(self.e_tile_device),
            "tile-net": len(self.e_tile_net),
            "tile-tile": len(self.e_tile_tile),
            "device-net": len(self.e_device_net),
        }


def build(placement: Placement, io_nets: list[str],
          hints: list[Hint] | None = None,
          w: int = DEFAULT_W, s: int = DEFAULT_S) -> HeteroGraph:
    """Turn one placement candidate into a tile-hinted heterogeneous graph."""
    io = set(io_nets)
    cw = placement.cell_width
    n_rows = placement.n_rows

    g = HeteroGraph(cell=placement.cell, index=placement.index, cell_width=cw)

    # --- device nodes: cell row x column x {N,P} x {L,G,R} ---
    net_terms: dict[str, list[int]] = {}
    net_cols: dict[str, list[int]] = {}

    for row, (nmos, pmos) in enumerate(placement.rows):
        for col in range(cw):
            for is_p, dev in ((False, nmos[col]), (True, pmos[col])):
                for stack, (term, netname) in enumerate(
                        zip((TERM_L, TERM_G, TERM_R), dev.terminals())):
                    node = DeviceNode(
                        idx=len(g.devices), col=col, row=row, is_pmos=is_p, term=term,
                        net=netname, nfin=dev.nfin, stack=stack,
                        is_dummy=dev.is_dummy, is_io=netname in io,
                    )
                    g.devices.append(node)
                    if netname and netname != "dummy":
                        net_terms.setdefault(netname, []).append(node.idx)
                        net_cols.setdefault(netname, []).append(col)

    # --- net nodes: cell-global ---
    net_index: dict[str, int] = {}
    for name in sorted(net_terms):
        cols = net_cols[name]
        node = NetNode(
            idx=len(g.nets), name=name, degree=len(net_terms[name]),
            span=max(cols) - min(cols) + 1,
            is_io=name in io, is_power=name in POWER_NETS,
        )
        net_index[name] = node.idx
        g.nets.append(node)

    for name, terms in net_terms.items():
        ni = net_index[name]
        for t in terms:
            g.e_device_net.append((t, ni))

    # --- tile nodes ---
    g.tiles = tile_positions(cw, w=w, s=s, n_rows=n_rows)
    g.hints = list(hints) if hints else [Hint.missing() for _ in g.tiles]
    while len(g.hints) < len(g.tiles):
        g.hints.append(Hint.missing())
    g.hints = g.hints[: len(g.tiles)]

    # A tile covers device slots in its own cell row only.
    for ti, tile in enumerate(g.tiles):
        seen_nets: set[int] = set()
        for d in g.devices:
            if d.row == tile.row and tile.covers(d.col):
                g.e_tile_device.append((ti, d.idx))
                if d.net in net_index:
                    seen_nets.add(net_index[d.net])
        for ni in sorted(seen_nets):
            g.e_tile_net.append((ti, ni))

    # tile<->tile, both ways: left/right within a cell row, and up/down between
    # the same tile column of vertically adjacent rows (paper §2.3). A
    # single-height cell has one tile row, so only the column direction exists.
    by_pos = {(t.row, t.col): i for i, t in enumerate(g.tiles)}
    for (r, c), a in by_pos.items():
        for nb in ((r, c + 1), (r + 1, c)):
            b = by_pos.get(nb)
            if b is not None:
                g.e_tile_tile.append((a, b))
                g.e_tile_tile.append((b, a))

    return g


def feature_matrices(g: HeteroGraph) -> dict[str, list[list[float]]]:
    """Per-type input matrices, consumed by the model's type-specific projections."""
    return {
        "device": [d.features() for d in g.devices],
        "net": [n.features() for n in g.nets],
        "tile": [h.features() + pe for h, pe in zip(g.hints, g.tile_pe())],
    }
