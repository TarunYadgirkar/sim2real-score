"""Top-level orchestration: sweep + sensitivity + score -> AnalysisResult, plus
the suggested domain-randomization config derived from where the policy is weak."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .randomization.space import RandomizationSpace
from .rollout.runner import make_evaluator, nominal_return
from .sensitivity.sobol import (SensitivityResult, interaction_matrix,
                                sobol_analysis)
from .sweep.bisect import BreakingPoint, all_breaking_points
from .sweep.grid import GridResult, coarse_grid

MIN_MULTIPLICATIVE = 1e-3   # keep suggested multiplicative ranges physical (> 0)
BASE_EXPANSION = 0.5        # how far past a breaking point to train, at ST = 0
MAX_EXPANSION = 1.5         # ... and at ST = 1
# A suggested range must stay on the same scale as the breaking point it covers.
# Linear expansion by the nominal-to-breaking-point gap can overshoot through
# zero (e.g. breaks at 0.38, gap 0.62 -> negative), which clamps to a degenerate
# floor and suggests training over a physically meaningless range.
MIN_FRACTION_OF_BP = 0.4
MAX_FACTOR_OF_BP = 2.5


@dataclass
class AnalysisResult:
    env_id: str
    score: float
    sensitivity: SensitivityResult
    breaking_points: Dict[str, BreakingPoint]
    grid: GridResult
    space: RandomizationSpace
    nominal_return: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def interaction_matrix(self) -> np.ndarray:
        return interaction_matrix(self.sensitivity)

    def to_dict(self) -> dict:
        return {
            "env": self.env_id,
            "score": self.score,
            "nominal_return": self.nominal_return,
            "sensitivity": self.sensitivity.to_dict(),
            "breaking_points": {k: v.to_dict()
                                for k, v in sorted(self.breaking_points.items())},
            "grid": {
                "names": list(self.grid.names),
                "per_param": self.grid.per_param,
                "failure_fraction": self.grid.failure_fraction,
                "truncated": self.grid.truncated,
                "n_full": self.grid.n_full,
                "n_evaluated": int(len(self.grid.success_rate)),
            },
            "space": self.space.to_dict(),
            "suggested_dr_config": suggest_dr_config(self),
            "meta": dict(self.meta),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=float)


def run_analysis(policy, env_id: str, space: Optional[RandomizationSpace] = None,
                 seed: int = 0, jobs: int = 1, serial: bool = False,
                 sobol_base: int = 32, grid_res: int = 3,
                 bisect_tol: float = 0.02) -> AnalysisResult:
    if space is None:
        from .envs.registry import default_space
        space = default_space(env_id)
    space = RandomizationSpace.from_dict({**space.to_dict(), "seed": int(seed)})

    nom = None
    if space.failure.get("threshold_kind") == "fraction_of_nominal":
        nom = nominal_return(policy, env_id, space)

    evaluate = make_evaluator(policy, env_id, space, nom_return=nom)

    sens = sobol_analysis(space, evaluate, n_base=sobol_base, n_jobs=jobs,
                          serial=serial)
    grid = coarse_grid(space, evaluate, resolution=grid_res, n_jobs=jobs,
                       serial=serial)
    bps = all_breaking_points(space, evaluate, tol=bisect_tol)

    return AnalysisResult(
        env_id=env_id,
        score=float(100.0 * sens.mean_success),
        sensitivity=sens,
        breaking_points=bps,
        grid=grid,
        space=space,
        nominal_return=nom,
        meta={"seed": int(seed), "sobol_base": int(sobol_base),
              "grid_resolution": int(grid_res), "episodes": space.rollout["episodes"],
              "max_steps": space.rollout["max_steps"]},
    )


def _expansion(st: float) -> float:
    st = 0.0 if not np.isfinite(st) else float(min(max(st, 0.0), 1.0))
    return BASE_EXPANSION + (MAX_EXPANSION - BASE_EXPANSION) * st


def suggest_dr_config(result: AnalysisResult) -> dict:
    """Ranges to train over next. Where the policy breaks, push the training
    range *past* the breaking point -- further for parameters with a high
    total-order index, since those are what actually drive failure. Parameters
    with no located breaking point keep their swept range."""
    space = result.space
    st_by_name = dict(zip(result.sensitivity.names,
                          [float(v) for v in result.sensitivity.ST]))
    params = {}
    for name, spec in space.params.items():
        lo, hi = float(spec.low), float(spec.high)
        bp = result.breaking_points.get(name)
        grow = _expansion(st_by_name.get(name, 0.0))
        if bp is not None:
            if bp.low is not None:
                lo = bp.low - grow * abs(spec.nominal - bp.low)
                if bp.low > 0:
                    lo = max(lo, MIN_FRACTION_OF_BP * bp.low)
            if bp.high is not None:
                hi = bp.high + grow * abs(bp.high - spec.nominal)
                if bp.high > 0:
                    hi = min(hi, MAX_FACTOR_OF_BP * bp.high)
        if spec.kind in ("multiplicative", "probability", "additive_std"):
            lo = max(lo, MIN_MULTIPLICATIVE if spec.kind == "multiplicative" else 0.0)
        if spec.kind == "probability":
            hi = min(hi, 1.0)
        if spec.is_int:
            lo, hi = float(int(round(lo))), float(int(round(hi)))
        params[name] = {"nominal": float(spec.nominal), "low": float(min(lo, hi)),
                        "high": float(max(lo, hi)), "kind": spec.kind}
        if spec.is_int:
            params[name]["is_int"] = True
    return {
        "env": space.env,
        "seed": space.seed,
        "rollout": dict(space.rollout),
        "failure": dict(space.failure),
        "params": params,
    }


def dump_dr_config(result: AnalysisResult, path: str) -> str:
    import yaml
    cfg = suggest_dr_config(result)
    ranking = result.sensitivity.ranking()
    ranked = (", ".join(ranking) if ranking and not result.sensitivity.degenerate
              else "n/a (no variance to attribute)")
    header = ("# Suggested domain-randomization ranges, derived from sim2real-score.\n"
              f"# Robustness score: {result.score:.1f}/100. "
              f"Ranked by total-order Sobol index: {ranked}\n")
    stuck = sorted(n for n, bp in result.breaking_points.items() if bp.fails_at_nominal)
    if stuck:
        header += ("# WARNING: the policy already fails at nominal for "
                   f"{', '.join(stuck)}, so no boundary could be located and\n"
                   "# those ranges collapse to the nominal value. Get the policy "
                   "working at nominal before training on this config.\n")
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)
    return path
