"""sim2real-score: find where a control policy breaks under domain shift."""

__version__ = "0.1.0"

from .policies.base import ConstantPolicy, LinearPolicy, Policy
from .policies.loader import load_policy
from .randomization.space import ParamSpec, RandomizationSpace
from .envs.registry import default_space, make_env
from .rollout.runner import RolloutResult, evaluate_point, make_evaluator, nominal_return

__all__ = [
    "__version__",
    "Policy", "LinearPolicy", "ConstantPolicy", "load_policy",
    "RandomizationSpace", "ParamSpec",
    "make_env", "default_space",
    "RolloutResult", "evaluate_point", "make_evaluator", "nominal_return",
]

from .sensitivity.sobol import SensitivityResult, sobol_analysis
from .sweep.bisect import BreakingPoint, bisect_boundary
from .sweep.grid import GridResult, coarse_grid
from .analysis import (AnalysisResult, dump_dr_config, run_analysis,
                       suggest_dr_config)

__all__ += [
    "SensitivityResult", "sobol_analysis",
    "BreakingPoint", "bisect_boundary", "GridResult", "coarse_grid",
    "AnalysisResult", "run_analysis", "suggest_dr_config", "dump_dr_config",
]
