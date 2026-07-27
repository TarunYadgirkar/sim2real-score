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

    def _resolve(self, base: str, target: str):
        """Rows of the model array that a target name refers to."""
        import mujoco

        model = self.env.unwrapped.model
        obj_type = {"friction": mujoco.mjtObj.mjOBJ_GEOM,
                    "mass": mujoco.mjtObj.mjOBJ_BODY,
                    "damping": mujoco.mjtObj.mjOBJ_JOINT}[base]
        index = mujoco.mj_name2id(model, obj_type, target)
        if index < 0:
            kind = {"friction": "geom", "mass": "body", "damping": "joint"}[base]
            count = {"friction": model.ngeom, "mass": model.nbody,
                     "damping": model.njnt}[base]
            available = [mujoco.mj_id2name(model, obj_type, i) for i in range(count)]
            raise ValueError(
                f"{base}.{target}: no {kind} named {target!r} in {self.env_id}. "
                f"Available {kind}s: {', '.join(n for n in available if n)}")
        if base != "damping":
            return [index]
        # a joint owns a contiguous block of DOFs
        start = int(model.jnt_dofadr[index])
        end = (int(model.jnt_dofadr[index + 1]) if index + 1 < model.njnt
               else int(model.nv))
        return list(range(start, end))

    def set_domain_params(self, params: dict) -> None:
        """Multipliers are always interpreted against the *nominal* model, so
        repeated calls never compound and a sweep point never inherits the
        previous one. Targeted and global factors compose multiplicatively."""
        from ..randomization.space import split_target

        model = self.env.unwrapped.model
        field = {"friction": "geom_friction", "mass": "body_mass",
                 "damping": "dof_damping"}
        pending = {}
        for name, value in params.items():
            base, target = split_target(name)
            if base not in field:
                continue
            pending.setdefault(base, []).append((target, float(value)))

        for base, entries in pending.items():
            values = self._nominal[field[base]].copy()
            for target, factor in entries:
                if target is None:
                    values *= factor
                else:
                    values[self._resolve(base, target)] *= factor
            getattr(model, field[base])[:] = values

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
