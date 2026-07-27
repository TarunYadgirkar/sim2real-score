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

# Optional (added once implemented): analysis + report symbols.
try:  # pragma: no cover
    from .analysis import AnalysisResult, run_analysis  # noqa: F401
    __all__ += ["AnalysisResult", "run_analysis"]
except Exception:
    pass

try:  # pragma: no cover
    from .sensitivity.sobol import SensitivityResult  # noqa: F401
    __all__ += ["SensitivityResult"]
except Exception:
    pass
