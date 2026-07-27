"""Environment protocol + a minimal Box space (so the built-in linear env needs
no gymnasium/MuJoCo)."""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class Box:
    def __init__(self, low, high, shape):
        self.shape = tuple(shape)
        self.low = np.broadcast_to(np.asarray(low, dtype=np.float64), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=np.float64), self.shape).copy()


class ControlEnv(Protocol):
    observation_space: Box
    action_space: Box
    supported_domain_params: set

    def reset(self, *, seed: Optional[int] = None):
        ...

    def step(self, action):
        ...

    def set_domain_params(self, params: dict) -> None:
        ...
