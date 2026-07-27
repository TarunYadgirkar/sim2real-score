"""Load policies from disk: stable-baselines3 `.zip`, TorchScript `.pt`, or ONNX.
Heavy backends are imported lazily so the core tool needs none of them."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base import Policy


def load_policy(path: str, kind: str = "auto", obs_dim: Optional[int] = None,
                action_dim: Optional[int] = None,
                vecnormalize: Optional[str] = None) -> Policy:
    """Load a policy. `vecnormalize` is the path to a saved SB3 VecNormalize
    (`.pkl`); its observation statistics are applied before every prediction.
    MuJoCo policies are almost always trained under VecNormalize, and feeding
    such a policy raw observations silently evaluates a different policy than the
    one that was trained."""
    kind = kind.lower()
    if kind == "auto":
        kind = _infer_kind(path)
    if vecnormalize is not None and kind != "sb3":
        raise ValueError("vecnormalize applies to SB3 checkpoints only")
    if kind == "torch":
        return TorchScriptPolicy(path, obs_dim=obs_dim, action_dim=action_dim)
    if kind == "onnx":
        return OnnxPolicy(path, obs_dim=obs_dim, action_dim=action_dim)
    if kind == "sb3":
        return Sb3Policy(path, vecnormalize=vecnormalize)
    raise ValueError(f"unknown policy kind: {kind!r}")


def _infer_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        return "onnx"
    if ext == ".zip":
        return "sb3"
    if ext in (".pt", ".pth", ".ts"):
        return "torch"
    raise ValueError(f"cannot infer policy kind from extension {ext!r}; pass kind=")


def _as_batch(obs):
    obs = np.asarray(obs, dtype=np.float32)
    single = obs.ndim == 1
    return (obs.reshape(1, -1) if single else obs), single


class _ReloadablePolicy:
    """Backend handles (TorchScript modules, ONNX sessions) are not picklable, so
    parallel workers cannot receive them directly. Pickle the *load spec* instead
    and rebuild the handle in the worker -- otherwise every analysis using a
    policy file would silently fall back to serial execution."""

    def __getstate__(self):
        return {"path": self.path, "obs_dim": self.obs_dim,
                "action_dim": self.action_dim}

    def __setstate__(self, state):
        self.__init__(state["path"], obs_dim=state["obs_dim"],
                      action_dim=state["action_dim"])


class TorchScriptPolicy(_ReloadablePolicy):
    """A TorchScript module taking float32 `[N, obs_dim]` -> `[N, action_dim]`."""

    def __init__(self, path: str, obs_dim=None, action_dim=None):
        import torch  # lazy
        self._torch = torch
        self.path = path
        self.module = torch.jit.load(path, map_location="cpu")
        self.module.eval()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x, single = _as_batch(obs)
        torch = self._torch
        with torch.no_grad():
            y = self.module(torch.from_numpy(x)).cpu().numpy()
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(x.shape[0], -1)
        return y[0] if single else y


class OnnxPolicy(_ReloadablePolicy):
    """A single-input/single-output ONNX model float32 `[N, obs_dim]` -> `[N, action_dim]`."""

    def __init__(self, path: str, obs_dim=None, action_dim=None):
        import onnxruntime as ort  # lazy
        self.path = path
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x, single = _as_batch(obs)
        y = self.session.run(None, {self.input_name: x})[0]
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(x.shape[0], -1)
        return y[0] if single else y


class Sb3Policy:
    """A stable-baselines3 checkpoint. The concrete algorithm is auto-detected by
    trying the standard SB3 algorithms."""

    _ALGOS = ("PPO", "SAC", "TD3", "A2C", "DDPG", "DQN")

    def __getstate__(self):
        return {"path": self.path, "vecnormalize": self.vecnormalize_path}

    def __setstate__(self, state):
        self.__init__(state["path"], vecnormalize=state.get("vecnormalize"))

    def __init__(self, path: str, vecnormalize: Optional[str] = None):
        import stable_baselines3 as sb3  # lazy
        self.path = path
        self.vecnormalize_path = vecnormalize
        self._obs_mean, self._obs_var, self._clip_obs, self._epsilon = (
            None, None, None, None)
        if vecnormalize is not None:
            self._load_obs_stats(vecnormalize)
        last_err = None
        model = None
        for name in self._ALGOS:
            cls = getattr(sb3, name, None)
            if cls is None:
                continue
            try:
                model = cls.load(path, device="cpu")
                break
            except Exception as e:  # wrong algorithm class -> try next
                last_err = e
        if model is None:
            raise RuntimeError(f"could not load SB3 checkpoint {path!r}: {last_err}")
        self.model = model
        space = model.observation_space
        self.obs_dim = int(np.prod(space.shape)) if space.shape else None
        self.action_dim = int(np.prod(model.action_space.shape))

    def _load_obs_stats(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            vec_normalize = pickle.load(f)
        if not getattr(vec_normalize, "norm_obs", False):
            return
        rms = vec_normalize.obs_rms
        self._obs_mean = np.asarray(rms.mean, dtype=np.float64)
        self._obs_var = np.asarray(rms.var, dtype=np.float64)
        self._clip_obs = float(vec_normalize.clip_obs)
        self._epsilon = float(vec_normalize.epsilon)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self._obs_mean is None:
            return x
        z = (x - self._obs_mean) / np.sqrt(self._obs_var + self._epsilon)
        return np.clip(z, -self._clip_obs, self._clip_obs).astype(np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        x = obs.reshape(1, -1) if single else obs
        x = self._normalize(x)
        action, _ = self.model.predict(x, deterministic=True)
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            action = action.reshape(x.shape[0], -1)
        return action[0] if single else action
