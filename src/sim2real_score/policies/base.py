"""Policy protocol and pure in-memory policies. A policy maps observations to
deterministic actions: `predict` accepts a single obs `(obs_dim,)` -> `(action_dim,)`
or a batch `(N, obs_dim)` -> `(N, action_dim)`."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    obs_dim: Optional[int]
    action_dim: int

    def predict(self, obs: np.ndarray) -> np.ndarray:
        ...


class LinearPolicy:
    """action = weight @ obs + bias. `weight` is (action_dim, obs_dim)."""

    def __init__(self, weight, bias=None):
        self.weight = np.asarray(weight, dtype=np.float64)
        if self.weight.ndim != 2:
            raise ValueError("weight must be 2D (action_dim, obs_dim)")
        self.action_dim, self.obs_dim = self.weight.shape
        if bias is None:
            self.bias = np.zeros(self.action_dim, dtype=np.float64)
        else:
            self.bias = np.asarray(bias, dtype=np.float64).reshape(self.action_dim)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        single = obs.ndim == 1
        x = obs.reshape(1, -1) if single else obs
        y = x @ self.weight.T + self.bias
        return y[0] if single else y


class ConstantPolicy:
    """Ignores observations; always returns the same action. `obs_dim` is None
    (accepts any observation dimension)."""

    obs_dim = None

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64).reshape(-1)
        self.action_dim = self.action.shape[0]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if obs.ndim == 1:
            return self.action.copy()
        return np.tile(self.action, (obs.shape[0], 1))
