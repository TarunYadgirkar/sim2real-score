"""Apply domain-shift parameters to a rollout.

Dynamics params (friction/mass/damping) go to the env's `set_domain_params`.
Universal params are applied by `DomainShiftWrapper`, which works on any env:
  - actuator_gain  : scale the action before it reaches the env
  - action_latency : integer step delay on applied actions (zero-filled buffer)
  - obs_noise      : additive Gaussian std on observations
  - sensor_dropout : per-step probability the observation is held stale

All randomness is drawn from a caller-provided Generator so a rollout is a pure
function of its seed."""
from __future__ import annotations

from collections import deque

import numpy as np

from .space import DYNAMICS_PARAMS, WRAPPER_PARAMS, base_param


def split_params(params: dict):
    """Route parameters to the simulator model vs the universal wrapper. Names may
    carry a target suffix (`friction.foot`), which only dynamics params accept."""
    dynamics, wrapper = {}, {}
    for name, value in params.items():
        base = base_param(name)
        if base in DYNAMICS_PARAMS:
            dynamics[name] = value
        elif base in WRAPPER_PARAMS:
            if name != base:
                raise ValueError(
                    f"{name!r}: {base} applies to the whole policy/env and takes "
                    "no target suffix")
            wrapper[name] = value
    return dynamics, wrapper


class DomainShiftWrapper:
    def __init__(self, env, wrapper_params: dict, rng: np.random.Generator):
        self.env = env
        self.rng = rng
        self.actuator_gain = float(wrapper_params.get("actuator_gain", 1.0))
        self.obs_noise = float(wrapper_params.get("obs_noise", 0.0))
        self.action_latency = int(round(wrapper_params.get("action_latency", 0)))
        self.sensor_dropout = float(wrapper_params.get("sensor_dropout", 0.0))
        self._buf = deque()
        self._held = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, *, seed=None):
        obs, info = self.env.reset(seed=seed)
        self._buf = deque(np.zeros_like(self.env.action_space.low)
                          for _ in range(self.action_latency))
        self._held = np.asarray(obs, dtype=np.float64)
        return self._corrupt(np.asarray(obs, dtype=np.float64)), info

    def step(self, action):
        gained = self.actuator_gain * np.asarray(action, dtype=np.float64)
        if self.action_latency > 0:
            self._buf.append(gained)
            applied = self._buf.popleft()
        else:
            applied = gained
        obs, reward, terminated, truncated, info = self.env.step(applied)
        return (self._corrupt(np.asarray(obs, dtype=np.float64)), reward,
                terminated, truncated, info)

    def _corrupt(self, fresh: np.ndarray) -> np.ndarray:
        if self.sensor_dropout > 0.0 and self.rng.random() < self.sensor_dropout:
            clean = self._held
        else:
            clean = fresh
        self._held = clean
        if self.obs_noise > 0.0:
            return clean + self.rng.normal(0.0, self.obs_noise, size=clean.shape)
        return clean.copy()
