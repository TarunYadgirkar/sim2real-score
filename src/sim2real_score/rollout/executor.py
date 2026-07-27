"""Run a batch of (values, index) evaluations serially or across CPU cores.

Determinism contract: results depend only on each job's `index` (which seeds the
rollout), never on scheduling or worker identity, and are returned in input
order. Hence serial and parallel outputs are identical."""
from __future__ import annotations

import warnings
from typing import Callable, List, Sequence, Tuple


def run_batch(evaluate: Callable, jobs: Sequence[Tuple[dict, int]],
              n_jobs: int = 1, serial: bool = False) -> List:
    if serial or n_jobs == 1 or len(jobs) <= 1:
        return [evaluate(values, index) for values, index in jobs]
    try:
        from joblib import Parallel, delayed
        return list(Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate)(values, index) for values, index in jobs))
    except Exception as exc:  # unpicklable policy (e.g. live ONNX session), no fork, ...
        warnings.warn(
            f"parallel execution unavailable ({type(exc).__name__}: {exc}); "
            "falling back to serial. Results are unchanged.", RuntimeWarning)
        return [evaluate(values, index) for values, index in jobs]
