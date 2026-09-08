"""Tile decomposition tests (paper §2.2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsynth.tiles.decompose import (
    FEATURE_DIM, Hint, coverage, from_jsonl, tile_positions, uncovered,
)


def test_paper_defaults_on_a_wide_cell():
    # Paper §2.2 example: slide the window with w=7, s=2
    tiles = tile_positions(13, w=7, s=2)
    assert [t.col0 for t in tiles] == [0, 2, 4, 6]
    assert all(t.width == 7 for t in tiles)


def test_no_window_crosses_the_boundary():
    for cw in range(7, 30):
        for t in tile_positions(cw, w=7, s=2):
            assert t.col0 + t.width <= cw


def test_cell_narrower_than_window_has_no_tiles():
    assert tile_positions(6, w=7, s=2) == []
    assert tile_positions(0, w=7, s=2) == []


def test_cell_exactly_window_width_has_one_tile():
    tiles = tile_positions(7, w=7, s=2)
    assert len(tiles) == 1 and tiles[0].col0 == 0


def test_adjacent_tiles_overlap():
    tiles = tile_positions(13, w=7, s=2)
    for a, b in zip(tiles, tiles[1:]):
        assert b.col0 < a.col0 + a.width      # they overlap


def test_coverage_counts_overlap():
    tiles = tile_positions(11, w=7, s=2)
    cov = coverage(11, tiles)
    assert len(cov) == 11
    assert min(cov) >= 1                       # (11-7)%2==0, so everything is covered
    assert max(cov) > 1                        # some positions are covered twice


def test_right_edge_can_be_uncovered():
    # The paper says "every device is covered", but a tail is left when (cw-w)%s != 0
    assert uncovered(10, tile_positions(10, w=7, s=2)) == [9]
    assert uncovered(11, tile_positions(11, w=7, s=2)) == []


def test_tile_covers():
    t = tile_positions(13, w=7, s=2)[1]        # col0=2
    assert t.covers(2) and t.covers(8) and not t.covers(9) and not t.covers(1)


def test_invalid_hint_is_gated_to_zero():
    h = Hint(y_route=True, y_naive=True, m1=0.9, m2=0.8, valid=False)
    f = h.features()
    assert len(f) == FEATURE_DIM
    assert f[:4] == [0.0, 0.0, 0.0, 0.0]       # the profile is gated off
    assert f[4] == 0.0                          # the v flag itself is preserved


def test_valid_hint_passes_through():
    h = Hint(y_route=True, y_naive=False, m1=0.5, m2=0.25, valid=True)
    assert h.features() == [1.0, 0.0, 0.5, 0.25, 1.0]


def test_missing_hint_is_invalid():
    assert Hint.missing().valid is False


def test_from_jsonl_groups_by_candidate():
    recs = [
        {"cell": "C", "k": 0, "tile": 0, "y_route": True, "y_naive": True,
         "m1": 0.3, "m2": 0.0, "valid": True, "runtime_ms": 100},
        {"cell": "C", "k": 0, "tile": 1, "y_route": False, "y_naive": True,
         "m1": 0.4, "m2": 0.1, "valid": True, "runtime_ms": 120},
        {"cell": "C", "k": 1, "tile": 0, "y_route": True, "y_naive": True,
         "m1": 0.2, "m2": 0.0, "valid": True, "runtime_ms": 90},
    ]
    out = from_jsonl(recs, cell_width=11)
    assert set(out) == {0, 1}
    # width 11, w=7, s=2 -> 3 tiles (col0 = 0,2,4). Only 2 hints arrived, so the last is padded.
    assert len(out[0].hints) == len(out[0].tiles) == 3
    assert out[0].hints[1].y_route is False
    assert out[0].hints[2].valid is False
    assert out[0].has_hints


def test_from_jsonl_keeps_candidates_without_tiles():
    # Candidates with cellWidth < w must not disappear either
    recs = [{"cell": "C", "k": 3, "tile": -1, "valid": False}]
    out = from_jsonl(recs, cell_width=5)
    assert 3 in out and not out[3].has_hints


def test_hint_count_matches_tile_count():
    recs = [{"cell": "C", "k": 0, "tile": 0, "valid": True}]
    out = from_jsonl(recs, cell_width=13)      # 4 tiles but only 1 hint arrived
    assert len(out[0].hints) == len(out[0].tiles) == 4
    assert all(not h.valid for h in out[0].hints[1:])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(fns)} passed")
