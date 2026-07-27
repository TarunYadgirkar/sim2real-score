"""Env factory + per-env default randomization spaces."""
from __future__ import annotations

import copy

from ..randomization.space import RandomizationSpace
from .linear import LinearEnv

_LINEAR_PARAMS = {
    "friction":       {"nominal": 1.0, "low": 0.3, "high": 2.0, "kind": "multiplicative"},
    "mass":           {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "damping":        {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "actuator_gain":  {"nominal": 1.0, "low": 0.5, "high": 1.5, "kind": "multiplicative"},
    "obs_noise":      {"nominal": 0.0, "low": 0.0, "high": 0.2, "kind": "additive_std"},
    "action_latency": {"nominal": 0,   "low": 0,   "high": 6,   "kind": "int"},
    "sensor_dropout": {"nominal": 0.0, "low": 0.0, "high": 0.6, "kind": "probability"},
}

_MUJOCO_PARAMS = {
    "friction":       {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "mass":           {"nominal": 1.0, "low": 0.6, "high": 1.6, "kind": "multiplicative"},
    "damping":        {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "actuator_gain":  {"nominal": 1.0, "low": 0.7, "high": 1.3, "kind": "multiplicative"},
    "obs_noise":      {"nominal": 0.0, "low": 0.0, "high": 0.05, "kind": "additive_std"},
    "action_latency": {"nominal": 0,   "low": 0,   "high": 3,   "kind": "int"},
    "sensor_dropout": {"nominal": 0.0, "low": 0.0, "high": 0.2, "kind": "probability"},
}

_DEFAULTS = {
    "linear": {
        "env": "linear", "seed": 0,
        "rollout": {"episodes": 6, "max_steps": 200},
        "failure": {"metric": "return", "threshold": 0.45, "threshold_kind": "mean_reward"},
        "params": _LINEAR_PARAMS,
    },
    "Reacher-v5": {
        "env": "Reacher-v5", "seed": 0,
        "rollout": {"episodes": 3, "max_steps": 50},
        "failure": {"metric": "return", "threshold": 0.6, "threshold_kind": "fraction_of_nominal"},
        "params": _MUJOCO_PARAMS,
    },
    "Hopper-v5": {
        "env": "Hopper-v5", "seed": 0,
        "rollout": {"episodes": 3, "max_steps": 300},
        "failure": {"metric": "return", "threshold": 0.6, "threshold_kind": "fraction_of_nominal"},
        "params": _MUJOCO_PARAMS,
    },
    "Walker2d-v5": {
        "env": "Walker2d-v5", "seed": 0,
        "rollout": {"episodes": 3, "max_steps": 300},
        "failure": {"metric": "return", "threshold": 0.6, "threshold_kind": "fraction_of_nominal"},
        "params": _MUJOCO_PARAMS,
    },
}

# gymnasium/MuJoCo aliases -> canonical
_ALIASES = {"Reacher": "Reacher-v5", "Hopper": "Hopper-v5", "Walker2d": "Walker2d-v5"}


def _canonical(env_id: str) -> str:
    return _ALIASES.get(env_id, env_id)


def make_env(env_id: str, seed=None):
    env_id = _canonical(env_id)
    if env_id == "linear":
        return LinearEnv(seed=seed)
    from .mujoco import MujocoControlEnv  # lazy; needs gymnasium+mujoco
    return MujocoControlEnv(env_id, seed=seed)


def default_space(env_id: str) -> RandomizationSpace:
    env_id = _canonical(env_id)
    if env_id in _DEFAULTS:
        return RandomizationSpace.from_dict(copy.deepcopy(_DEFAULTS[env_id]))
    # Unknown gymnasium id: fall back to generic MuJoCo-style defaults.
    d = copy.deepcopy(_DEFAULTS["Hopper-v5"])
    d["env"] = env_id
    return RandomizationSpace.from_dict(d)
