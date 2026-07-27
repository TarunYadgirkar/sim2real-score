"""Ground-truth policy fixtures on the built-in `linear` env (damped spring-mass,
obs=[pos,vel], action=[u], u = weight @ obs).

Each fixture's fragility comes from a *distinct physical mechanism*, so the two
axes decouple (see DECISIONS D4/D8):

- friction-overfit: `u = -Kp*pos + c*vel` with `c` sized to cancel friction at
  nominal. Net closed-loop damping is `(b - c)`, so when friction drops the loop
  loses its damping and destabilizes. Loop gain stays low, so delay margin is
  large -> latency-robust.
- latency-fragile: `u = -Kp*pos - Kd*vel` with strong Kd. Its own damping swamps
  any friction variation -> friction-robust; but high derivative gain acting on a
  delayed signal loses phase margin -> latency-fragile.

Do not tune these to make the tool pass. Tune them so the fixture genuinely has
the property (verified directly by the `test_ground_truth_*` tests), then let the
tool discover it.
"""
import numpy as np
from sim2real_score import LinearPolicy, ConstantPolicy, RandomizationSpace

OBS_DIM = 2
ACT_DIM = 1

# (Kp, c) -- c > 0 is positive velocity feedback that cancels nominal friction.
FRICTION_OVERFIT_GAINS = (2.0, 1.1)
# (Kp, Kd) -- strong self-damping.
LATENCY_FRAGILE_GAINS = (10.0, 18.0)

RANDOM_GAIN = 12.0

FRICTION_LOW = 0.3
LATENCY_HIGH = 6
EPISODES = 6
MAX_STEPS = 200
THRESHOLD = 0.45


def _policy(pos_gain, vel_coeff):
    """u = pos_gain*pos + vel_coeff*vel (callers pass signed coefficients)."""
    return LinearPolicy(weight=np.array([[pos_gain, vel_coeff]], dtype=np.float64),
                        bias=np.zeros(ACT_DIM))


def friction_overfit_policy():
    kp, c = FRICTION_OVERFIT_GAINS
    return _policy(-kp, +c)


def latency_fragile_policy():
    kp, kd = LATENCY_FRAGILE_GAINS
    return _policy(-kp, -kd)


def zero_policy():
    return ConstantPolicy(np.zeros(ACT_DIM))


def random_policy(seed=0):
    """A fixed, large random linear map: erratic and destabilizing, yet a pure
    obs->action function so determinism/parallel guarantees still hold."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=RANDOM_GAIN, size=(ACT_DIM, OBS_DIM))
    return LinearPolicy(weight=w, bias=np.zeros(ACT_DIM))


_PARAM_DEFAULTS = {
    "friction":       {"nominal": 1.0, "low": FRICTION_LOW, "high": 2.0, "kind": "multiplicative"},
    # Mass is deliberately a *narrow* range: wide enough to be a real control
    # parameter, narrow enough that neither fixture breaks on it, so each
    # policy's fragility stays isolated to its intended axis.
    "mass":           {"nominal": 1.0, "low": 0.8, "high": 1.3, "kind": "multiplicative"},
    "damping":        {"nominal": 1.0, "low": 0.5, "high": 2.0, "kind": "multiplicative"},
    "actuator_gain":  {"nominal": 1.0, "low": 0.5, "high": 1.5, "kind": "multiplicative"},
    "obs_noise":      {"nominal": 0.0, "low": 0.0, "high": 0.2, "kind": "additive_std"},
    "action_latency": {"nominal": 0,   "low": 0,   "high": LATENCY_HIGH, "kind": "int"},
    "sensor_dropout": {"nominal": 0.0, "low": 0.0, "high": 0.6, "kind": "probability"},
}


def linear_space(active=("friction", "mass", "action_latency"), episodes=EPISODES,
                 max_steps=MAX_STEPS, threshold=THRESHOLD,
                 threshold_kind="mean_reward", seed=0):
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
