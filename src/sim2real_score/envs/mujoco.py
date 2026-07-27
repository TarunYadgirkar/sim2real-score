"""gymnasium/MuJoCo environments (Hopper, Walker2d, Reacher, ...) exposed through
the ControlEnv interface.

Dynamics parameters are applied as multipliers on the *nominal* model values,
which are cached at construction so repeated `set_domain_params` calls never
compound. Runs headless: no renderer is created."""
from __future__ import annotations

import numpy as np


class MujocoControlEnv:
    supported_domain_params = {"friction", "mass", "damping"}

    def __init__(self, env_id: str, seed=None):
        import gymnasium as gym

        self.env_id = env_id
        self.env = gym.make(env_id)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._seed = seed

        model = self.env.unwrapped.model
        self._nominal = {
            "body_mass": np.array(model.body_mass, copy=True),
            "dof_damping": np.array(model.dof_damping, copy=True),
            "geom_friction": np.array(model.geom_friction, copy=True),
        }

    def set_domain_params(self, params: dict) -> None:
        model = self.env.unwrapped.model
        if "mass" in params:
            model.body_mass[:] = self._nominal["body_mass"] * float(params["mass"])
        if "damping" in params:
            model.dof_damping[:] = self._nominal["dof_damping"] * float(params["damping"])
        if "friction" in params:
            model.geom_friction[:] = self._nominal["geom_friction"] * float(params["friction"])

    def reset(self, *, seed=None):
        obs, info = self.env.reset(seed=self._seed if seed is None else seed)
        return np.asarray(obs, dtype=np.float64).reshape(-1), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (np.asarray(obs, dtype=np.float64).reshape(-1), float(reward),
                bool(terminated), bool(truncated), info)

    def close(self):
        self.env.close()
