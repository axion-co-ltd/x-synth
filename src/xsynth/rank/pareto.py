"""Pareto-layered candidate ranker (paper §2.5).

Takes the predictor's (y_route, m1, m2) estimates, sorts the candidates by
non-dominated sorting into layers (rank 0, 1, 2, ...), and emits the queue the
router should consume in that order.

No dependencies. Not even numpy. A few thousand candidates per cell is well
within pure Python, and this module in particular has to run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Prediction:
    """Output of the predictor f_theta(c_i): the (y_route, m) of paper §2.1."""

    cid: int
    y_route: float          # higher is better (pin-access-aware routability)
    m1: float               # lower is better
    m2: float               # lower is better
    width: int = 0          # lower is better (the width term of §2.5)


@dataclass
class Ranked:
    pred: Prediction
    layer: int              # Pareto rank (0 is best)
    order: int              # final position in the queue (0-based)
    _key: tuple = field(default=(), repr=False)


def dominates(a: Prediction, b: Prediction) -> bool:
    """Paper §2.5: does `a` dominate `b`?

    No worse on all three axes and strictly better on at least one
    (routability up, M1 down, M2 down).
    """
    not_worse = (a.y_route >= b.y_route) and (a.m1 <= b.m1) and (a.m2 <= b.m2)
    if not not_worse:
        return False
    return (a.y_route > b.y_route) or (a.m1 < b.m1) or (a.m2 < b.m2)


def pareto_layers(preds: list[Prediction]) -> list[list[Prediction]]:
    """Peel non-dominated frontiers one at a time into a list of layers.

    The non-dominated set is rank 0; removing it exposes the next frontier as
    rank 1, and so on. This is O(L*N^2), which is fine at a few thousand
    candidates per cell.
    """
    remaining = list(preds)
    layers: list[list[Prediction]] = []

    while remaining:
        front = [p for p in remaining
                 if not any(dominates(q, p) for q in remaining if q is not p)]
        if not front:
            # Unreachable in theory (dominance is acyclic), but do not drop
            # whatever is left if it ever happens.
            layers.append(remaining)
            break
        layers.append(front)
        front_ids = {id(p) for p in front}
        remaining = [p for p in remaining if id(p) not in front_ids]

    return layers


def rank(preds: list[Prediction], *, width_weight: bool = True) -> list[Ranked]:
    """Sort candidates into a priority queue (paper §2.5).

    1) Pareto rank ascending
    2) Within a rank: y_route descending, then m2 ascending, then m1 ascending
    3) Width is the final tie-break: "do not buy routability with cell area"

    WARNING: the paper only says the ranker "also accounts for" width and never
    says where it enters. Here it is used **only as a tie-break, never in the
    dominance test**. Adding it as a fourth dominance axis widens the frontier
    so much that nearly everything becomes non-dominated and the ranking stops
    meaning anything.
    """
    layers = pareto_layers(preds)

    out: list[Ranked] = []
    for depth, layer in enumerate(layers):
        key = lambda p: (-p.y_route, p.m2, p.m1, p.width if width_weight else 0)
        for p in sorted(layer, key=key):
            out.append(Ranked(pred=p, layer=depth, order=len(out), _key=key(p)))
    return out


def queue(preds: list[Prediction], k: int | None = None) -> list[int]:
    """Candidate ids in the order the router should try them.

    With `k`, only the first k are returned (the paper's K_Phi).
    """
    ids = [r.pred.cid for r in rank(preds)]
    return ids if k is None else ids[:k]
