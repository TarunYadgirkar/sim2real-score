"""Coarse grid sweep: find the regions of the randomization space where the
policy fails, before bisection refines the boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..rollout.executor import run_batch

GRID_CAP = 4096          # max points evaluated; beyond this the factorial is subsampled
GRID_INDEX_OFFSET = 100_000   # keeps grid seeds disjoint from other stages


@dataclass
class GridResult:
    names: List[str]
    points: List[Dict[str, float]]
    success_rate: np.ndarray
    truncated: bool = False
    n_full: int = 0
    per_param: Dict[str, dict] = field(default_factory=dict)

    @property
    def failure_fraction(self) -> float:
        return float(np.mean(self.success_rate < 0.5)) if len(self.success_rate) else 0.0


def _axis(spec, resolution: int) -> List[float]:
    if spec.is_int:
        lo, hi = int(round(spec.low)), int(round(spec.high))
        vals = sorted({int(round(v)) for v in np.linspace(lo, hi, resolution)})
        return [float(v) for v in vals]
    return [float(v) for v in np.linspace(spec.low, spec.high, resolution)]


def coarse_grid(space, evaluate, resolution: int = 3, n_jobs: int = 1,
                serial: bool = False) -> GridResult:
    specs = space.active_params
    if not specs:
        return GridResult(names=[], points=[], success_rate=np.zeros(0))

    axes = [_axis(s, resolution) for s in specs]
    names = [s.name for s in specs]
    n_full = int(np.prod([len(a) for a in axes]))

    idx = np.arange(n_full)
    truncated = n_full > GRID_CAP
    if truncated:
        # deterministic uniform subsample of the full factorial
        idx = np.unique(np.linspace(0, n_full - 1, GRID_CAP).astype(int))

    points = []
    for flat in idx:
        combo = np.unravel_index(int(flat), tuple(len(a) for a in axes))
        points.append({n: axes[d][combo[d]] for d, n in enumerate(names)})

    jobs = [(p, GRID_INDEX_OFFSET + int(i)) for i, p in zip(idx, points)]
    results = run_batch(evaluate, jobs, n_jobs=n_jobs, serial=serial)
    success = np.array([r.success_rate for r in results], dtype=np.float64)

    per_param = {}
    for name in names:
        vals = np.array([p[name] for p in points], dtype=np.float64)
        uniq = np.unique(vals)
        per_param[name] = {
            "values": [float(v) for v in uniq],
            "success_rate": [float(success[vals == v].mean()) for v in uniq],
        }

    return GridResult(names=names, points=points, success_rate=success,
                      truncated=truncated, n_full=n_full, per_param=per_param)
