"""Tile-hinted heterogeneous graph, single- and multi-height (paper §2.3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xsynth.backend.formats import Device, Placement          # noqa: E402
from xsynth.graph.build import build                          # noqa: E402


def _row(width: int, tag: str) -> tuple[list[Device], list[Device]]:
    """One cell row of `width` columns, with per-row net names."""
    n = [Device(name=f"MN{i}", left=f"{tag}n{i}", gate=f"{tag}g{i}",
                right=f"{tag}n{i + 1}", nfin=1) for i in range(width)]
    p = [Device(name=f"MP{i}", left=f"{tag}p{i}", gate=f"{tag}g{i}",
                right=f"{tag}p{i + 1}", nfin=1) for i in range(width)]
    return n, p


def _placement(width: int, rows: int) -> Placement:
    first = _row(width, "r0")
    upper = [_row(width, f"r{i}") for i in range(1, rows)]
    return Placement(cell="X", index=0, file_width=width + 2,
                     nmos=first[0], pmos=first[1], upper_rows=upper)


def test_single_height_unchanged():
    g = build(_placement(9, 1), io_nets=["r0g0"])
    assert g.tile_shape == (1, 2)                  # (9-7)//2 + 1 = 2 columns
    assert all(t.row == 0 for t in g.tiles)
    assert all(d.row == 0 for d in g.devices)
    # one row means no up/down relations, so every edge is left/right
    cols = {t.col for t in g.tiles}
    assert len(g.e_tile_tile) == 2 * (len(cols) - 1)
    assert all(pe[0] == 0.0 for pe in g.tile_pe())


def test_multi_height_stacks_rows():
    g1 = build(_placement(9, 1), io_nets=[])
    g2 = build(_placement(9, 2), io_nets=[])

    assert g2.tile_shape == (2, 2)
    assert len(g2.tiles) == 2 * len(g1.tiles)
    assert len(g2.devices) == 2 * len(g1.devices)
    # nets are cell-global, and the two rows use different names here
    assert len(g2.nets) == 2 * len(g1.nets)


def test_tiles_cover_only_their_own_row():
    g = build(_placement(9, 2), io_nets=[])
    rows = {d.idx: d.row for d in g.devices}
    for ti, di in g.e_tile_device:
        assert rows[di] == g.tiles[ti].row


def test_vertical_tile_relations_appear():
    g1 = build(_placement(9, 1), io_nets=[])
    g2 = build(_placement(9, 2), io_nets=[])
    pos = {i: (t.row, t.col) for i, t in enumerate(g2.tiles)}

    vertical = [(a, b) for a, b in g2.e_tile_tile
                if pos[a][1] == pos[b][1] and pos[a][0] != pos[b][0]]
    horizontal = [(a, b) for a, b in g2.e_tile_tile
                  if pos[a][0] == pos[b][0] and pos[a][1] != pos[b][1]]

    assert vertical, "multi-height must add up/down tile relations"
    assert len(vertical) == 2 * 2                    # 2 columns, both directions
    assert len(horizontal) == 2 * len(g1.e_tile_tile)  # one row's worth per row
    # every relation is a nearest neighbour on the grid
    for a, b in g2.e_tile_tile:
        dr = abs(pos[a][0] - pos[b][0])
        dc = abs(pos[a][1] - pos[b][1])
        assert dr + dc == 1


def test_row_positional_encoding_is_used():
    pe1 = build(_placement(9, 1), io_nets=[]).tile_pe()
    pe2 = build(_placement(9, 2), io_nets=[]).tile_pe()
    assert {p[0] for p in pe1} == {0.0}             # single height: flat
    assert len({p[0] for p in pe2}) == 2            # multi height: two levels


def test_narrow_cell_has_no_tiles_in_any_row():
    g = build(_placement(6, 2), io_nets=[])
    assert g.tiles == []
    assert g.tile_shape == (0, 0)
    assert g.e_tile_tile == []


def _fns():
    return [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


def test_m2_blocking_only_counts_other_nets():
    """An access point is blocked by another net's M2, not by the pin's own."""
    from xsynth.backend.pin_access import _m2_owners

    grids = {
        "A": {"M1": [[1, 0]], "M2": [[1, 0]]},   # A covers its own point
        "B": {"M1": [[0, 1]], "M2": [[1, 1]]},   # B also sits on A's point
    }
    owners = _m2_owners(grids)
    assert owners[(0, 0)] == {"A", "B"}
    assert owners[(0, 1)] == {"B"}
    # A's point is blocked by B; B's own M2 does not block B
    assert owners[(0, 0)] - {"A"} == {"B"}
    assert owners[(0, 1)] - {"B"} == set()


if __name__ == "__main__":
    fns = _fns()
    for f in fns:
        f()
    print(f"\n{len(fns)} passed")
