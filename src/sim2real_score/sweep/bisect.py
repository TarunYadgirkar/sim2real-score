"""Bisection: locate the failure boundary (breaking point) per parameter.

Holds every other parameter at nominal and moves one parameter from nominal
toward each bound, bisecting to the value where `success_rate` crosses 0.5.
For integer parameters the returned value is the first *failing* integer; for
continuous ones it is the midpoint of the final bracket."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BISECT_INDEX_OFFSET = 200_000
MAX_ITERS = 40
SUCCESS_CUTOFF = 0.5


@dataclass
class BreakingPoint:
    param: str
    low: Optional[float] = None
    high: Optional[float] = None
    fails_at_nominal: bool = False

    def to_dict(self) -> dict:
        return {"param": self.param, "low": self.low, "high": self.high,
                "fails_at_nominal": self.fails_at_nominal}


class _Counter:
    """Deterministic index source: evaluation order is fixed by the algorithm,
    so a plain counter yields reproducible rollout seeds."""

    def __init__(self, start: int):
        self.value = start

    def next(self) -> int:
        self.value += 1
        return self.value


def _passes(evaluate, param: str, value: float, counter: _Counter) -> bool:
    return evaluate({param: value}, counter.next()).success_rate >= SUCCESS_CUTOFF


def _bisect_side(evaluate, spec, bound: float, counter: _Counter,
                 tol: float) -> Optional[float]:
    """Return the boundary between nominal (passing) and `bound`, or None if the
    policy still passes at `bound`."""
    if _passes(evaluate, spec.name, bound, counter):
        return None
    pass_val, fail_val = float(spec.nominal), float(bound)
    for _ in range(MAX_ITERS):
        if spec.is_int:
            mid = float(round((pass_val + fail_val) / 2.0))
            if mid == pass_val or mid == fail_val:
                break
        else:
            if abs(fail_val - pass_val) <= tol:
                break
            mid = (pass_val + fail_val) / 2.0
        if _passes(evaluate, spec.name, mid, counter):
            pass_val = mid
        else:
            fail_val = mid
    return fail_val if spec.is_int else (pass_val + fail_val) / 2.0


def bisect_boundary(space, param: str, evaluate, tol: float = 0.02,
                    index_offset: int = BISECT_INDEX_OFFSET) -> BreakingPoint:
    spec = space.param(param)
    counter = _Counter(index_offset)
    bp = BreakingPoint(param=param)

    if not _passes(evaluate, param, spec.nominal, counter):
        bp.fails_at_nominal = True
        bp.low = bp.high = float(spec.nominal)
        return bp

    if spec.low < spec.nominal:
        bp.low = _bisect_side(evaluate, spec, spec.low, counter, tol)
    if spec.high > spec.nominal:
        bp.high = _bisect_side(evaluate, spec, spec.high, counter, tol)
    return bp


def all_breaking_points(space, evaluate, tol: float = 0.02) -> dict:
    out = {}
    for i, spec in enumerate(space.active_params):
        # disjoint index blocks per parameter -> stable, collision-free seeds
        offset = BISECT_INDEX_OFFSET + i * 1000
        out[spec.name] = bisect_boundary(space, spec.name, evaluate, tol=tol,
                                         index_offset=offset)
    return out
