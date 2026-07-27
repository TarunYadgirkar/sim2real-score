"""Built-in synthetic env: a regulated damped point mass.

obs = [pos, vel], action = [u]. Discrete dynamics (explicit Euler):
    b     = FRICTION0*friction + DAMPING0*damping     (viscous drag)
    m     = MASS0*mass
    vel  += DT * (u - b*vel) / m
    pos  += DT * vel
    reward= exp(-(pos^2 + CTRL_COST*u^2))              in (0, 1]

Each episode starts from a small random displacement (seeded), vel=0, so a
regulating controller must actively stabilize — exposing gain/latency/friction
behavior — while a do-nothing policy stays near its start (vel stays 0). This
makes the friction-vs-latency ground truth analytic and deterministic
(see DECISIONS D4)."""
from __future__ import annotations

import numpy as np

from .base import Box

DT = 0.1
MASS0 = 1.0
FRICTION0 = 1.0
DAMPING0 = 0.5
CTRL_COST = 1e-3
POS0_LOW = 0.2
POS0_HIGH = 0.4
DIVERGE = 1e3


class LinearEnv:
    supported_domain_params = {"friction", "mass", "damping"}

    def __init__(self, seed=None):
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))
        self._friction = 1.0
        self._mass = 1.0
        self._damping = 1.0
        self.rng = np.random.default_rng(seed)
        self.pos = 0.0
        self.vel = 0.0

    def set_domain_params(self, params: dict) -> None:
        self._friction = float(params.get("friction", self._friction))
        self._mass = float(params.get("mass", self._mass))
        self._damping = float(params.get("damping", self._damping))

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = float(self.rng.uniform(POS0_LOW, POS0_HIGH))
        self.vel = 0.0
        return self._obs(), {}

    def _obs(self):
        return np.array([self.pos, self.vel], dtype=np.float64)

    def step(self, action):
        u = float(np.asarray(action).reshape(-1)[0])
        b = FRICTION0 * self._friction + DAMPING0 * self._damping
        m = MASS0 * self._mass
        self.vel = self.vel + DT * (u - b * self.vel) / m
        self.pos = self.pos + DT * self.vel
        terminated = (not np.isfinite(self.pos) or not np.isfinite(self.vel)
                      or abs(self.pos) > DIVERGE)
        if terminated:
            reward = 0.0
        else:
            reward = float(np.exp(-(self.pos ** 2 + CTRL_COST * u ** 2)))
        return self._obs(), reward, terminated, False, {}
