"""Pareto ranker tests (paper §2.5)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsynth.rank.pareto import Prediction, dominates, pareto_layers, queue, rank


def P(cid, y, m1, m2, w=0):
    return Prediction(cid=cid, y_route=y, m1=m1, m2=m2, width=w)


def test_dominates_all_axes_better():
    assert dominates(P(0, 0.9, 1, 1), P(1, 0.5, 2, 2))
    assert not dominates(P(1, 0.5, 2, 2), P(0, 0.9, 1, 1))


def test_dominates_needs_strict_improvement():
    a, b = P(0, 0.5, 1, 1), P(1, 0.5, 1, 1)
    assert not dominates(a, b) and not dominates(b, a)


def test_dominates_is_partial_on_tradeoff():
    # high routability + high metal  vs  low routability + low metal
    a, b = P(0, 0.9, 5, 5), P(1, 0.3, 1, 1)
    assert not dominates(a, b) and not dominates(b, a)


def test_layers_peel_frontiers():
    preds = [P(0, 0.9, 1, 1), P(1, 0.8, 2, 2), P(2, 0.7, 3, 3)]
    layers = pareto_layers(preds)
    assert [[p.cid for p in L] for L in layers] == [[0], [1], [2]]


def test_layers_keep_every_candidate():
    preds = [P(i, i % 3 / 3, i % 4, i % 5) for i in range(40)]
    layers = pareto_layers(preds)
    flat = [p.cid for L in layers for p in L]
    assert sorted(flat) == list(range(40))


def test_tie_break_order_is_route_then_m2_then_m1():
    # Built so that none dominates another, putting them all on one frontier
    preds = [P(0, 0.5, 9, 1), P(1, 0.5, 1, 9), P(2, 0.9, 5, 5)]
    ids = queue(preds)
    assert ids[0] == 2                    # highest y_route
    assert ids.index(0) < ids.index(1)    # equal y_route -> lower m2 first


def test_width_breaks_remaining_ties():
    preds = [P(0, 0.5, 1, 1, w=9), P(1, 0.5, 1, 1, w=3)]
    assert queue(preds)[0] == 1           # narrower cell first


def test_queue_prefix_is_budget():
    preds = [P(i, 1 - i / 10, i, i) for i in range(10)]
    assert queue(preds, k=3) == queue(preds)[:3]
    assert len(queue(preds, k=3)) == 3


def test_ranked_layers_are_monotonic():
    preds = [P(i, (i * 7 % 10) / 10, i % 4, i % 3) for i in range(30)]
    layers = [r.layer for r in rank(preds)]
    assert layers == sorted(layers)


def test_single_candidate():
    assert queue([P(0, 0.5, 1, 1)]) == [0]


def test_empty():
    assert queue([]) == []


def test_budget_stops_when_the_wall_clock_is_exhausted():
    """Paper §2.5: route from the front of the queue until Phi is exhausted."""
    from xsynth.baselines.orderings import Candidate, under_budget

    c = [Candidate(cid=0, cost=0, width=8, routable=False, m1=0, m2=0, runtime_ms=400),
         Candidate(cid=1, cost=1, width=8, routable=True, m1=0, m2=0.5, runtime_ms=400),
         Candidate(cid=2, cost=2, width=8, routable=True, m1=0, m2=0.1, runtime_ms=400)]
    order = [0, 1, 2]

    # nothing fits
    r = under_budget(c, order, 100)
    assert r["n_routed"] == 0 and not r["routable"]

    # the first two fit: routable, but not the best (cid 2 has lower m2)
    r = under_budget(c, order, 900)
    assert r["n_routed"] == 2 and r["routable"] and not r["best"]

    # everything fits
    r = under_budget(c, order, 5000)
    assert r["n_routed"] == 3 and r["best"]

    # a candidate is never part-routed, so the budget is not overspent
    assert under_budget(c, order, 799)["spent_ms"] <= 799


def test_budget_never_exceeds_phi():
    from xsynth.baselines.orderings import Candidate, under_budget

    c = [Candidate(cid=i, cost=i, width=8, routable=True, m1=0, m2=0,
                   runtime_ms=300 + 100 * i) for i in range(5)]
    for phi in (0, 250, 700, 1500, 10_000):
        assert under_budget(c, [x.cid for x in c], phi)["spent_ms"] <= phi


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\n{len(fns)} passed")
