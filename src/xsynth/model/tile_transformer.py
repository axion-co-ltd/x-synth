"""Tile-Transformer predictor (paper §2.4).

    per-type input projection -> heterogeneous graph transformer (shared backbone)
      -> prediction heads (y_route, m1, m2) + train-only aux heads (y_naive, y_hc)

Follows the architecture of paper Figure 5. Written in plain torch without PyG:
the heterogeneous graph is small (three node types, four relations), and every
dependency dropped makes the reproduction easier to run.

Two design points matter:

1. **Neighbour-restricted attention.** Rather than full graph-transformer
   attention, each node attends only to its neighbours, so the physical
   structure of the standard-cell graph acts as prior knowledge. Each relation
   gets its own projection, letting the model learn per-relation attention
   strength.

2. **Tile-hint residual.** So that a risk signal concentrated in a few tiles is
   not diluted by graph pooling, a summary of the hints bypasses the backbone
   and feeds the classifier directly (the dashed path in Figure 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph.build import DEVICE_DIM, NET_DIM
from ..tiles.decompose import FEATURE_DIM as HINT_DIM

TILE_DIM = HINT_DIM + 2          # 5 hint dims + 2D positional encoding

# Relation names - each relation owns its attention projection
RELATIONS = ("tile_device", "tile_net", "tile_tile", "device_net")


@dataclass
class ModelConfig:
    hidden: int = 64
    layers: int = 3              # the paper's L
    heads: int = 4
    dropout: float = 0.1


class RelationAttention(nn.Module):
    """Neighbour-restricted multi-head attention for one relation.

    Source nodes send messages to destination nodes. Pairs without an edge are
    never computed, which avoids the O(N^2) of full attention and makes the
    graph structure itself the mask.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """edges: (2, E) as [src_idx; dst_idx]. Returns messages gathered at dst."""
        out = torch.zeros_like(x_dst)
        if edges.numel() == 0:
            return out

        src, dst = edges[0], edges[1]
        q = self.q(x_dst)[dst].view(-1, self.h, self.dk)      # (E, H, dk)
        k = self.k(x_src)[src].view(-1, self.h, self.dk)
        v = self.v(x_src)[src].view(-1, self.h, self.dk)

        score = (q * k).sum(-1) / math.sqrt(self.dk)          # (E, H)

        # softmax per dst - edge-wise scatter softmax
        score = score - score.max()
        e = score.exp()
        denom = torch.zeros(x_dst.size(0), self.h, device=x_dst.device)
        denom.index_add_(0, dst, e)
        alpha = e / (denom[dst] + 1e-9)

        msg = (alpha.unsqueeze(-1) * v).reshape(-1, self.h * self.dk)
        out.index_add_(0, dst, msg)
        return out


class HeteroLayer(nn.Module):
    """One heterogeneous graph transformer layer, updating from per-relation attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden
        self.attn = nn.ModuleDict({r: RelationAttention(d, cfg.heads) for r in RELATIONS})
        self.norm = nn.ModuleDict({t: nn.LayerNorm(d) for t in ("device", "net", "tile")})
        self.ff = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                             nn.Dropout(cfg.dropout), nn.Linear(2 * d, d))
            for t in ("device", "net", "tile")
        })

    def forward(self, z: dict[str, torch.Tensor],
                edges: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        agg = {t: torch.zeros_like(v) for t, v in z.items()}

        # tile -> device / net / tile, device -> net (the reverse also flows)
        agg["device"] = agg["device"] + self.attn["tile_device"](
            z["tile"], z["device"], edges["tile_device"])
        agg["net"] = agg["net"] + self.attn["tile_net"](
            z["tile"], z["net"], edges["tile_net"])
        agg["tile"] = agg["tile"] + self.attn["tile_tile"](
            z["tile"], z["tile"], edges["tile_tile"])
        agg["net"] = agg["net"] + self.attn["device_net"](
            z["device"], z["net"], edges["device_net"])

        # reverse: net -> device, device -> tile, net -> tile
        agg["device"] = agg["device"] + self.attn["device_net"](
            z["net"], z["device"], edges["device_net"].flip(0))
        agg["tile"] = agg["tile"] + self.attn["tile_device"](
            z["device"], z["tile"], edges["tile_device"].flip(0))
        agg["tile"] = agg["tile"] + self.attn["tile_net"](
            z["net"], z["tile"], edges["tile_net"].flip(0))

        out = {}
        for t in z:
            h = self.norm[t](z[t] + agg[t])
            out[t] = h + self.ff[t](h)
        return out


class AxisAttentionPool(nn.Module):
    """Pool tile embeddings along one axis of the tile grid with a learned query.

    The RowAttn / ColAttn of paper §2.4: the row (stack) axis summarises #M1 and
    the column axis summarises #M2.

    Tiles arrive flat in row-major order. Given the grid shape, this attends
    within each line of `axis` and then averages the line summaries, so the two
    heads see genuinely different reductions. A single-height cell has one tile
    row, which makes the row axis a single line, so that reduction is the same as
    attending over every tile at once.
    """

    def __init__(self, dim: int, axis: str):
        super().__init__()
        assert axis in ("row", "col")
        self.axis = axis
        self.score = nn.Linear(dim, 1)

    def _attend(self, z: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.score(z).squeeze(-1), dim=0)
        return (a.unsqueeze(-1) * z).sum(0)

    def forward(self, z_tile: torch.Tensor,
                shape: tuple[int, int] | None = None) -> torch.Tensor:
        if z_tile.numel() == 0:
            return torch.zeros(z_tile.size(-1), device=z_tile.device)
        n_rows, n_cols = shape or (1, z_tile.size(0))
        if n_rows * n_cols != z_tile.size(0) or n_rows <= 1:
            return self._attend(z_tile)          # single-height, or shape unknown
        grid = z_tile.view(n_rows, n_cols, -1)
        # "row" pools across the stacked cell rows at each tile column;
        # "col" pools along the columns within each cell row.
        lines = grid.transpose(0, 1) if self.axis == "row" else grid
        return torch.stack([self._attend(line) for line in lines]).mean(0)


class TileTransformer(nn.Module):
    """The full architecture of paper Figure 5."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        d = self.cfg.hidden

        # per-type input projections - different meanings and dimensions
        self.proj = nn.ModuleDict({
            "device": nn.Linear(DEVICE_DIM, d),
            "net": nn.Linear(NET_DIM, d),
            "tile": nn.Linear(TILE_DIM, d),
        })

        self.backbone = nn.ModuleList([HeteroLayer(self.cfg) for _ in range(self.cfg.layers)])

        # CLS-style pooling: a learned query attends over all node embeddings
        self.cls = nn.Parameter(torch.randn(d) * 0.02)
        self.cls_attn = nn.Linear(d, d)

        # tile-hint residual - bypasses the backbone into the classifier (Figure 5, dashed)
        self.hint_res = nn.Sequential(nn.Linear(TILE_DIM, d), nn.GELU(), nn.Linear(d, d))

        self.row_pool = AxisAttentionPool(d, "row")     # #M1
        self.col_pool = AxisAttentionPool(d, "col")     # #M2

        # prediction heads
        self.g_route = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.g_m1 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.g_m2 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        # train-only auxiliary heads - discarded at inference
        self.g_naive = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.g_hc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, feats: dict[str, torch.Tensor],
                edges: dict[str, torch.Tensor],
                tile_shape: tuple[int, int] | None = None) -> dict[str, torch.Tensor]:
        """`tile_shape` is the (cell rows, tile columns) of the tile grid.

        Omitted, the tiles are treated as one row, which is what a single-height
        cell is.
        """
        z = {t: self.proj[t](feats[t]) for t in ("device", "net", "tile")}
        for layer in self.backbone:
            z = layer(z, edges)

        # cell-level CLS vector
        allz = torch.cat([z["device"], z["net"], z["tile"]], dim=0)
        score = (self.cls_attn(allz) @ self.cls) / math.sqrt(allz.size(-1))
        z_cls = (torch.softmax(score, 0).unsqueeze(-1) * allz).sum(0)

        # hint residual: summarise the raw hints without passing the backbone
        r_tile = (self.hint_res(feats["tile"]).mean(0)
                  if feats["tile"].numel() else torch.zeros_like(z_cls))

        head_in = torch.cat([z_cls, r_tile], dim=-1)
        z_row = self.row_pool(z["tile"], tile_shape)
        z_col = self.col_pool(z["tile"], tile_shape)

        return {
            "y_route": self.g_route(head_in).squeeze(-1),
            "m1": self.g_m1(z_row).squeeze(-1),
            "m2": self.g_m2(z_col).squeeze(-1),
            # auxiliary (training only)
            "y_naive": self.g_naive(head_in).squeeze(-1),
            "hc": self.g_hc(z["tile"]).squeeze(-1) if z["tile"].numel()
                  else torch.zeros(0, device=allz.device),
        }

    @torch.no_grad()
    def predict(self, feats, edges, tile_shape=None) -> tuple[float, float, float]:
        """Inference output (y_route, m1, m2). Aux heads are dropped - paper §2.4."""
        was_training = self.training
        self.eval()
        try:
            o = self.forward(feats, edges, tile_shape)
        finally:
            self.train(was_training)
        return (torch.sigmoid(o["y_route"]).item(), o["m1"].item(), o["m2"].item())


def multitask_loss(out: dict[str, torch.Tensor], target: dict[str, torch.Tensor],
                   w_route: float = 1.0, w_m1: float = 0.5, w_m2: float = 0.5,
                   w_naive: float = 0.3, w_hc: float = 0.2) -> torch.Tensor:
    """Weighted-sum multi-task loss (paper §2.4).

    Main heads: BCE for y_route, MSE for m1/m2.
    Aux heads:  BCE for y_naive (contrastive supervision on pin-access failure),
                MSE for hc, the per-column M1 profile that preserves local
                structure the cell-level total would wash out.

    The paper specifies the composition but not the weight values; the defaults
    below are this repository's choice.
    """
    loss = w_route * F.binary_cross_entropy_with_logits(
        out["y_route"].reshape(()), target["y_route"].reshape(()))
    loss = loss + w_m1 * F.mse_loss(out["m1"].reshape(()), target["m1"].reshape(()))
    loss = loss + w_m2 * F.mse_loss(out["m2"].reshape(()), target["m2"].reshape(()))

    if "y_naive" in target:
        loss = loss + w_naive * F.binary_cross_entropy_with_logits(
            out["y_naive"].reshape(()), target["y_naive"].reshape(()))
    if "hc" in target and out["hc"].numel() and target["hc"].numel():
        n = min(out["hc"].numel(), target["hc"].numel())
        loss = loss + w_hc * F.mse_loss(out["hc"][:n], target["hc"][:n])
    return loss
