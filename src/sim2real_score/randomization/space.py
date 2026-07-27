"""Randomization space: the set of domain-shift parameters, their nominal values
and sweep bounds, plus rollout/failure config. Loaded from YAML or a dict."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

DYNAMICS_PARAMS = ("friction", "mass", "damping")
WRAPPER_PARAMS = ("actuator_gain", "obs_noise", "action_latency", "sensor_dropout")


@dataclass
class ParamSpec:
    name: str
    nominal: float
    low: float
    high: float
    kind: str = "multiplicative"
    is_int: bool = False

    @property
    def active(self) -> bool:
        return self.low != self.high

    def clip(self, value: float) -> float:
        lo, hi = min(self.low, self.high), max(self.low, self.high)
        value = float(min(max(value, lo), hi))
        return round(value) if self.is_int else value


class RandomizationSpace:
    def __init__(self, env: str, params: Dict[str, ParamSpec], rollout: dict,
                 failure: dict, seed: int = 0):
        self.env = env
        self.params = params
        self.rollout = dict(rollout)
        self.failure = dict(failure)
        self.seed = seed

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "RandomizationSpace":
        params = {}
        for name, spec in d.get("params", {}).items():
            kind = spec.get("kind", "multiplicative")
            params[name] = ParamSpec(
                name=name,
                nominal=float(spec["nominal"]),
                low=float(spec["low"]),
                high=float(spec["high"]),
                kind=kind,
                is_int=bool(spec.get("is_int", kind == "int")),
            )
        rollout = d.get("rollout", {"episodes": 5, "max_steps": 200})
        failure = d.get("failure", {"metric": "return", "threshold": 0.5,
                                    "threshold_kind": "mean_reward"})
        return cls(env=d.get("env", "linear"), params=params, rollout=rollout,
                   failure=failure, seed=int(d.get("seed", 0)))

    @classmethod
    def from_yaml(cls, path: str) -> "RandomizationSpace":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    # ---- accessors ----------------------------------------------------
    @property
    def active_params(self) -> List[ParamSpec]:
        return [p for p in self.params.values() if p.active]

    @property
    def active_names(self) -> List[str]:
        return [p.name for p in self.active_params]

    @property
    def nominal_params(self) -> Dict[str, float]:
        return {name: p.nominal for name, p in self.params.items()}

    def param(self, name: str) -> ParamSpec:
        return self.params[name]

    def nominal(self, name: str) -> float:
        return self.params[name].nominal

    def low(self, name: str) -> float:
        return self.params[name].low

    def high(self, name: str) -> float:
        return self.params[name].high

    # ---- serialization ------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "env": self.env,
            "seed": self.seed,
            "rollout": dict(self.rollout),
            "failure": dict(self.failure),
            "params": {
                name: {"nominal": p.nominal, "low": p.low, "high": p.high,
                       "kind": p.kind, **({"is_int": True} if p.is_int else {})}
                for name, p in self.params.items()
            },
        }
