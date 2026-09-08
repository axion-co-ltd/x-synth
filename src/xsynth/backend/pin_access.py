"""Pin-access checker: the pin-access-aware routability label of paper §2.4.

The paper's criterion is tied to an industrial rule deck and cannot be carried
over as-is, so it is **defined here** on top of an open PDK. That is a required
port, not a change of method.

Definition
----------
A cell's external pins (IOnet) are only useful if upper-level routing can reach
them from outside the cell. In the backbone's configuration M2 is horizontal
(`M2_DIR HOR`), so wiring entering from outside approaches per **row**:

    access point   = a grid position (y, x) where that net has M1 metal
    access track   = a distinct row y among those positions
    pin reachable  = number of access tracks >= k

`k` is the difficulty knob (k=1 loose, k=2 strict; `run_xsynth.py --k-access`).

Labels
------
    y_naive = did the router succeed          (the backbone's is_routable)
    y_route = y_naive AND every external pin is reachable

This mirrors the paper's statement that the naive head supplies contrastive
supervision on pin-access failure: the candidates where the two labels disagree
are exactly that signal.

Power rails (VDD/VSS) are excluded by default. That is the usual convention for
signal-pin accessibility checks, and rails cross the cell boundary anyway, so
they are always reachable.

Choosing the criterion
----------------------
On an open grid an input pin contacts a single gate poly, so one M1 stub is
enough and the router has no reason to create more; input pins therefore tend to
have exactly one access track. Counting tracks alone then makes `k=1` pass
everything and `k=2` fail every input pin.

`require_free_m2` adds the one piece of geometry the lattice does carry: an
access point is only usable if upper-level routing can land on it, so a position
already occupied by a **different** net's M2 does not count. The paper's own
criterion comes from an industrial deck (via enclosure, spacing, end-of-line),
which this abstraction does not have; blocking is the closest available stand-in.

Without it, `y_route` reduces to plain routing success, which is a single class
in-distribution on this backbone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

POWER_NETS = frozenset({"VDD", "VSS", "VPP", "VBB", "GND"})

_NET_HDR_RE = re.compile(r"^=+\s*(\S+)\s*=+$")
_LAYER_RE = re.compile(r"^(V0|M1|V1|M2)$")


@dataclass(frozen=True)
class PinAccess:
    net: str
    n_points: int           # number of M1 metal positions
    n_tracks: int           # number of distinct rows (= access tracks)
    n_blocked: int = 0      # access points another net's M2 already covers

    def ok(self, k: int) -> bool:
        return self.n_tracks >= k


@dataclass(frozen=True)
class AccessReport:
    per_pin: dict[str, PinAccess]
    k: int

    @property
    def ok(self) -> bool:
        """Does every external signal pin meet the criterion?"""
        return all(p.ok(self.k) for p in self.per_pin.values())

    @property
    def failed(self) -> list[str]:
        return sorted(n for n, p in self.per_pin.items() if not p.ok(self.k))


def _metal_rows(lines: list[str]) -> list[list[int]]:
    """Extract only the grid rows from a metal block.

    The backbone alternates metal rows with vertical-connector rows (`|` and
    spaces). Only the lines containing digits are grid rows.
    """
    rows = []
    for ln in lines:
        digits = re.findall(r"\d", ln)
        if digits:
            rows.append([int(d) for d in digits])
    return rows


def parse_route_nets(path: Path) -> dict[str, dict[str, list[list[int]]]]:
    """Per-net layer grids from a route file.

    Returns {net_name: {"M1": [[..]], "M2": [[..]], ...}}, reading only the
    `====== <net> ======` sections after `-------- Routing Results ---`.
    """
    text = path.read_text(errors="replace")
    if "Routing Results" in text:
        text = text.split("Routing Results", 1)[1]

    out: dict[str, dict[str, list[list[int]]]] = {}
    net: str | None = None
    layer: str | None = None
    buf: list[str] = []

    def flush():
        if net and layer and buf:
            out.setdefault(net, {})[layer] = _metal_rows(buf)

    for raw in text.splitlines():
        line = raw.rstrip()
        if m := _NET_HDR_RE.match(line.strip()):
            flush()
            net, layer, buf = m.group(1), None, []
            continue
        if _LAYER_RE.match(line.strip()):
            flush()
            layer, buf = line.strip(), []
            continue
        if layer:
            buf.append(line)
    flush()
    return out


def _m2_owners(grids: dict[str, dict[str, list[list[int]]]]
               ) -> dict[tuple[int, int], set[str]]:
    """Which nets occupy M2 at each lattice position."""
    owners: dict[tuple[int, int], set[str]] = {}
    for net, layers in grids.items():
        for y, row in enumerate(layers.get("M2", [])):
            for x, v in enumerate(row):
                if v:
                    owners.setdefault((y, x), set()).add(net)
    return owners


def check(
    route_file: Path,
    io_nets: list[str],
    k: int = 1,
    *,
    include_power: bool = False,
    layer: str = "M1",
    require_free_m2: bool = True,
) -> AccessReport:
    """Decide external-pin accessibility from a routing result.

    `require_free_m2` (on by default) counts an access point only when the M2
    position directly above it is not already taken by a **different** net. An
    access point that upper-level routing cannot land on is not usable in
    practice, and this is the only part of the deck's geometry the lattice
    exposes. The pin's own M2 does not block it.
    """
    grids = parse_route_nets(route_file)
    owners = _m2_owners(grids) if require_free_m2 else {}

    pins = [n for n in io_nets if include_power or n not in POWER_NETS]
    per: dict[str, PinAccess] = {}

    for n in pins:
        rows = grids.get(n, {}).get(layer, [])
        pts = [(y, x) for y, row in enumerate(rows) for x, v in enumerate(row) if v]
        if require_free_m2:
            free = [(y, x) for (y, x) in pts if not (owners.get((y, x), set()) - {n})]
        else:
            free = pts
        per[n] = PinAccess(net=n, n_points=len(pts),
                           n_tracks=len({y for y, _ in free}),
                           n_blocked=len(pts) - len(free))

    return AccessReport(per_pin=per, k=k)


def label(
    route_file: Path,
    io_nets: list[str],
    routable: bool,
    k: int = 1,
    *,
    require_free_m2: bool = True,
) -> tuple[bool, bool, AccessReport]:
    """Produce both of the paper's labels.

    Returns: (y_route, y_naive, report)
    """
    y_naive = routable
    report = check(route_file, io_nets, k=k, require_free_m2=require_free_m2)
    y_route = y_naive and report.ok
    return y_route, y_naive, report
