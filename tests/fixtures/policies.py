"""Ground-truth policy fixtures on the built-in `linear` env (regulated damped
point mass, obs=[pos,vel], action=[u], u = weight @ obs + bias).

Gains are tuned so each fixture has a *directly measurable* sensitivity profile
(see DECISIONS D4). Do not tune these to make the tool pass — tune them so the
fixture genuinely has the property, then let the tool discover it.
"""
import numpy as np
from sim2real_score import LinearPolicy, ConstantPolicy, RandomizationSpace

OBS_DIM = 2
ACT_DIM = 1

# Low derivative gain -> leans on the environment's own friction for closed-loop
# damping. Fragile when friction drops (under-damped -> diverges). Low overall
# gain -> large delay margin -> robust to action latency.
FRICTION_OVERFIT_GAINS = (5.0, 0.2)   # (Kp, Kd)

# Strong self-damping -> robust across the friction range. But high derivative
# gain on a delayed signal -> instability -> fragile to action latency.
LATENCY_FRAGILE_GAINS = (7.0, 7.0)    # (Kp, Kd)

RANDOM_GAIN = 12.0


def _feedback(kp, kd):
    return LinearPolicy(weight=np.array([[-kp, -kd]], dtype=np.float64),
                        bias=np.zeros(ACT_DIM))


def friction_overfit_policy():
    return _feedback(*FRICTION_OVERFIT_GAINS)


def latency_fragile_policy():
    return _feedback(*LATENCY_FRAGILE_GAINS)


def zero_policy():
    return ConstantPolicy(np.zeros(ACT_DIM))


def random_policy(seed=0):
    """A fixed, large random linear map: erratic and destabilizing, yet a pure
    obs->action function so determinism/parallel guarantees still hold."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=RANDOM_GAIN, size=(ACT_DIM, OBS_DIM))
    return LinearPolicy(weight=w, bias=np.zeros(ACT_DIM))


_PARAM_DEFAULTS = {
    "friction":       {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "mass":           {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "damping":        {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "actuator_gain":  {"nominal": 1.0, "low": 0.5, "high": 1.5, "kind": "multiplicative"},
    "obs_noise":      {"nominal": 0.0, "low": 0.0, "high": 0.2, "kind": "additive_std"},
    "action_latency": {"nominal": 0,   "low": 0,   "high": 6,   "kind": "int"},
    "sensor_dropout": {"nominal": 0.0, "low": 0.0, "high": 0.6, "kind": "probability"},
}


def linear_space(active=("friction", "mass", "action_latency"), episodes=5,
                 max_steps=150, threshold=0.5, threshold_kind="mean_reward",
                 seed=0):
    """Build a RandomizationSpace on the linear env. Params not in `active` are
    pinned to nominal (low == high == nominal), so they are inert in the sweep."""
    params = {name: dict(spec) for name, spec in _PARAM_DEFAULTS.items()}
    for name, spec in params.items():
        if name not in active:
            spec["low"] = spec["high"] = spec["nominal"]
    return RandomizationSpace.from_dict({
        "env": "linear",
        "seed": seed,
        "rollout": {"episodes": episodes, "max_steps": max_steps},
        "failure": {"metric": "return", "threshold": threshold,
                    "threshold_kind": threshold_kind},
        "params": params,
    })
