"""Variance-based (Sobol) sensitivity over the randomization space.

Saltelli-sampled, analysing the *failure indicator* `1 - success_rate`, so the
variance decomposed is variance in what breaks the policy. Parameters are ranked
by total-order index `ST`, which includes every interaction the parameter takes
part in -- one-at-a-time deltas would miss exactly those."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..rollout.executor import run_batch

SOBOL_INDEX_OFFSET = 0
ZERO_VARIANCE_TOL = 1e-12


@dataclass
class SensitivityResult:
    names: List[str]
    S1: np.ndarray
    ST: np.ndarray
    S2: np.ndarray
    mean_success: float
    degenerate: bool = False

    def ranking(self) -> List[str]:
        """Parameter names, most influential first (total-order, ties broken by
        first-order then name for reproducibility)."""
        order = sorted(range(len(self.names)),
                       key=lambda i: (-self.ST[i], -self.S1[i], self.names[i]))
        return [self.names[i] for i in order]

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "S1": [float(v) for v in self.S1],
            "ST": [float(v) for v in self.ST],
            "S2": [[None if np.isnan(v) else float(v) for v in row] for row in self.S2],
            "ranking": self.ranking(),
            "mean_success": float(self.mean_success),
            "degenerate": bool(self.degenerate),
        }


def _problem(specs):
    return {
        "num_vars": len(specs),
        "names": [s.name for s in specs],
        "bounds": [[float(min(s.low, s.high)), float(max(s.low, s.high))] for s in specs],
    }


def sample_space(space, n_base: int, seed: int = 0) -> tuple:
    """Saltelli design over the active parameters. Returns (specs, X).

    `seed` is mandatory in practice: SALib scrambles the Sobol' sequence with a
    fresh random seed unless one is supplied, which would make every run produce
    a different design (and different indices)."""
    from SALib.sample import sobol as sobol_sample

    specs = space.active_params
    if not specs:
        return specs, np.zeros((0, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SALib warns when n_base is not 2^k
        X = sobol_sample.sample(_problem(specs), n_base, calc_second_order=True,
                                seed=int(seed))
    return specs, X


def sobol_analysis(space, evaluate, n_base: int = 32, n_jobs: int = 1,
                   serial: bool = False,
                   samples: Optional[tuple] = None) -> SensitivityResult:
    from SALib.analyze import sobol as sobol_analyze

    seed = int(getattr(space, "seed", 0))
    specs, X = samples if samples is not None else sample_space(space, n_base, seed)
    names = [s.name for s in specs]
    if not specs:
        empty = np.zeros(0)
        return SensitivityResult(names=[], S1=empty, ST=empty,
                                 S2=np.zeros((0, 0)), mean_success=1.0,
                                 degenerate=True)

    jobs = []
    for i, row in enumerate(X):
        values = {s.name: s.clip(float(v)) for s, v in zip(specs, row)}
        jobs.append((values, SOBOL_INDEX_OFFSET + i))
    results = run_batch(evaluate, jobs, n_jobs=n_jobs, serial=serial)

    success = np.array([r.success_rate for r in results], dtype=np.float64)
    Y = 1.0 - success                      # failure indicator
    n = len(names)

    if float(np.var(Y)) <= ZERO_VARIANCE_TOL:
        # Uniformly robust (or uniformly broken): no variance to attribute.
        return SensitivityResult(names=names, S1=np.zeros(n), ST=np.zeros(n),
                                 S2=np.full((n, n), np.nan),
                                 mean_success=float(success.mean()), degenerate=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Si = sobol_analyze.analyze(_problem(specs), Y, calc_second_order=True,
                                   print_to_console=False, seed=seed)

    S1 = np.nan_to_num(np.asarray(Si["S1"], dtype=np.float64))
    ST = np.nan_to_num(np.asarray(Si["ST"], dtype=np.float64))
    S2 = np.asarray(Si["S2"], dtype=np.float64)
    return SensitivityResult(names=names, S1=S1, ST=ST, S2=S2,
                             mean_success=float(success.mean()))


def interaction_matrix(result: SensitivityResult) -> np.ndarray:
    """Symmetric |S2| with the total-order indices on the diagonal; NaNs zeroed
    so it renders as a heatmap."""
    n = len(result.names)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        M[i, i] = result.ST[i]
        for j in range(i + 1, n):
            v = result.S2[i, j] if result.S2.size else np.nan
            v = 0.0 if np.isnan(v) else abs(float(v))
            M[i, j] = M[j, i] = v
    return M
