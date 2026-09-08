"""HeteroGraph -> torch tensors.

The only seam between graph construction (pure Python) and the model (torch).
"""

from __future__ import annotations

import torch

from ..graph.build import HeteroGraph, feature_matrices
from .tile_transformer import RELATIONS


def _edges(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def encode(g: HeteroGraph, device: str | torch.device = "cpu"):
    """A (feats, edges) tensor pair. Empty node sets keep their shape."""
    fm = feature_matrices(g)

    from ..graph.build import DEVICE_DIM, NET_DIM
    from .tile_transformer import TILE_DIM

    dims = {"device": DEVICE_DIM, "net": NET_DIM, "tile": TILE_DIM}
    feats = {}
    for t, rows in fm.items():
        feats[t] = (torch.tensor(rows, dtype=torch.float32) if rows
                    else torch.zeros((0, dims[t]), dtype=torch.float32)).to(device)

    edges = {
        "tile_device": _edges(g.e_tile_device),
        "tile_net": _edges(g.e_tile_net),
        "tile_tile": _edges(g.e_tile_tile),
        "device_net": _edges(g.e_device_net),
    }
    edges = {k: v.to(device) for k, v in edges.items()}
    assert set(edges) == set(RELATIONS)
    return feats, edges


def targets(*, y_route: bool, m1: float, m2: float,
            y_naive: bool | None = None, hc: list[float] | None = None,
            device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Training targets. Auxiliary heads are included only when available."""
    t = {
        "y_route": torch.tensor(float(y_route), device=device),
        "m1": torch.tensor(float(m1), device=device),
        "m2": torch.tensor(float(m2), device=device),
    }
    if y_naive is not None:
        t["y_naive"] = torch.tensor(float(y_naive), device=device)
    if hc:
        t["hc"] = torch.tensor(hc, dtype=torch.float32, device=device)
    return t
