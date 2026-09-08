"""Baseline orderings that need no model.

The paper compares against NVCell 2 and LatticeGNN+PDA; neither is
reimplemented here. Instead we compare against orderings that follow **from the
labels alone**. One of them, `cost`, is the placer's own congestion
estimate. It is not a weak opponent.

Every function takes the candidate list and returns the order (a list of ids)
in which the router should try them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """The minimum needed to evaluate. Only `oracle` looks at the labels."""

    cid: int
    cost: float
    width: int
    routable: bool
    m1: float
    m2: float
    runtime_ms: int

    @property
    def quality(self) -> tuple[int, float, float]:
        """Smaller is better: the lexicographic W > M2 > M1 of paper Table 1."""
        return (self.width, self.m2, self.m1)


def by_cost(c: list[Candidate]) -> list[int]:
    """Ascending placer cost. The ordering the backbone gives away for free."""
    return [x.cid for x in sorted(c, key=lambda x: (x.cost, x.cid))]


def by_width(c: list[Candidate]) -> list[int]:
    return [x.cid for x in sorted(c, key=lambda x: (x.width, x.cid))]


def by_random(c: list[Candidate], seed: int = 0) -> list[int]:
    ids = [x.cid for x in c]
    random.Random(seed).shuffle(ids)
    return ids


def by_oracle(c: list[Candidate]) -> list[int]:
    """The best order given the labels. This is the reachable lower bound."""
    return [x.cid for x in sorted(
        c, key=lambda x: (not x.routable, x.quality, x.runtime_ms, x.cid))]


def reach_times(c: list[Candidate], order: list[int]) -> tuple[float | None, float | None]:
    """(t1, t*) for a given order, as cumulative routing time in ms.

    t1  time until the first routable candidate is reached
    t*  time until the lexicographic-best candidate is reached
    """
    by_id = {x.cid: x for x in c}
    routables = [x for x in c if x.routable]
    if not routables:
        return None, None
    best = min(x.quality for x in routables)

    cum, t1, tstar = 0.0, None, None
    for cid in order:
        x = by_id.get(cid)
        if x is None:
            continue
        cum += x.runtime_ms
        if x.routable:
            if t1 is None:
                t1 = cum
            if tstar is None and x.quality == best:
                tstar = cum
    return t1, tstar


def under_budget(c: list[Candidate], order: list[int],
                 budget_ms: float) -> dict[str, object]:
    """What the flow returns under a cumulative wall-clock budget (paper §2.5).

    "With a wall-clock budget, the router proceeds from the front of the queue
    until the cumulative routing time is exhausted. After each router
    invocation, the framework updates the best feasible layout found so far and
    the best observed #M1/#M2 usage among feasible layouts."

    A candidate is routed only if it fits entirely within what is left, so the
    budget is never overspent. Reports whether anything routable was found, and
    whether the lexicographic-best candidate was among them.
    """
    by_id = {x.cid: x for x in c}
    routables = [x for x in c if x.routable]
    best = min((x.quality for x in routables), default=None)

    spent = 0.0
    n = 0
    found: Candidate | None = None
    for cid in order:
        x = by_id.get(cid)
        if x is None:
            continue
        if spent + x.runtime_ms > budget_ms:
            break
        spent += x.runtime_ms
        n += 1
        if x.routable and (found is None or x.quality < found.quality):
            found = x

    return {
        "n_routed": n,
        "spent_ms": spent,
        "routable": found is not None,
        "best": found is not None and best is not None and found.quality == best,
        "quality": found.quality if found else None,
    }


ORDERINGS = {
    "oracle": by_oracle,
    "cost": by_cost,
    "width": by_width,
    "random": by_random,
}


def evaluate(c: list[Candidate], predicted: list[int] | None = None,
             *, trials: int = 200,
             budgets_ms: tuple[float, ...] = ()) -> dict[str, dict]:
    """t1/t* for every baseline and, when given, for the predicted order.

    `budgets_ms` additionally reports, per ordering, what the flow would return
    under each cumulative wall-clock budget Phi (paper §2.5, §3.2).
    """
    out: dict[str, dict[str, float | None]] = {}

    for name, fn in ORDERINGS.items():
        if name == "random":
            t1s, tss = [], []
            for s in range(trials):
                a, b = reach_times(c, by_random(c, seed=s))
                if a is not None:
                    t1s.append(a)
                if b is not None:
                    tss.append(b)
            out[name] = {"t1": sum(t1s) / len(t1s) if t1s else None,
                         "tstar": sum(tss) / len(tss) if tss else None}
        else:
            a, b = reach_times(c, fn(c))
            out[name] = {"t1": a, "tstar": b}

    if predicted is not None:
        a, b = reach_times(c, predicted)
        out["xsynth"] = {"t1": a, "tstar": b}

    # The budget rows use a single shuffle (seed 0), not the mean over `trials`.
    # Averaging would be more consistent with t1/t*; it has not been done.
    for name in out:
        order = (predicted if name == "xsynth"
                 else by_random(c, seed=0) if name == "random"
                 else ORDERINGS[name](c))
        out[name]["budget"] = {
            int(b): under_budget(c, order, b) for b in budgets_ms
        }
    return out
