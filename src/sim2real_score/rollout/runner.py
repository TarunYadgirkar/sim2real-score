"""Rollouts + the deterministic seeding contract.

Every rollout is a pure function of (policy, params, base_seed, index, episodes,
max_steps): the seed for each episode derives only from (base_seed, index,
episode) via numpy SeedSequence, never from wall clock or worker identity. Hence
serial and parallel execution are identical."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..envs.registry import make_env
from ..randomization.apply import DomainShiftWrapper, split_params


@dataclass
class RolloutResult:
    mean_return: float
    min_return: float
    success_rate: float
    n_episodes: int


def episode_seeds(base_seed: int, index: int, episode: int):
    ss = np.random.SeedSequence([int(base_seed), int(index), int(episode)])
    env_ss, wrap_ss = ss.spawn(2)
    return int(env_ss.generate_state(1)[0]), int(wrap_ss.generate_state(1)[0])


def _episode_return(policy, wrapped, env_seed, max_steps) -> float:
    obs, _ = wrapped.reset(seed=env_seed)
    total = 0.0
    for _ in range(max_steps):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, _ = wrapped.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total


def _episode_success(total, max_steps, failure, nominal_return) -> bool:
    kind = failure.get("threshold_kind", "mean_reward")
    thr = float(failure.get("threshold", 0.5))
    if kind == "mean_reward":
        return (total / max_steps) >= thr
    if kind == "absolute":
        return total >= thr
    if kind == "fraction_of_nominal":
        if nominal_return is None or nominal_return <= 0:
            return total >= thr  # degenerate nominal: treat threshold as absolute
        return total >= thr * nominal_return
    raise ValueError(f"unknown threshold_kind: {kind!r}")


def run_rollout(policy, env_id: str, params: Dict[str, float], base_seed: int,
                index: int, episodes: int, max_steps: int, failure: dict,
                nominal_return: Optional[float] = None) -> RolloutResult:
    dynamics, wrapper_params = split_params(params)
    env = make_env(env_id)
    env.set_domain_params(dynamics)
    totals = np.empty(episodes, dtype=np.float64)
    successes = 0
    for ep in range(episodes):
        env_seed, wrap_seed = episode_seeds(base_seed, index, ep)
        wrapped = DomainShiftWrapper(env, wrapper_params, np.random.default_rng(wrap_seed))
        total = _episode_return(policy, wrapped, env_seed, max_steps)
        totals[ep] = total
        if _episode_success(total, max_steps, failure, nominal_return):
            successes += 1
    return RolloutResult(mean_return=float(totals.mean()),
                         min_return=float(totals.min()),
                         success_rate=successes / episodes,
                         n_episodes=episodes)


def nominal_return(policy, env_id: str, space, base_seed=None) -> float:
    """Mean return at nominal params (used by fraction_of_nominal thresholding)."""
    seed = space.seed if base_seed is None else base_seed
    res = run_rollout(policy, env_id, space.nominal_params, seed, 0,
                      space.rollout["episodes"], space.rollout["max_steps"],
                      space.failure, nominal_return=None)
    return res.mean_return


def evaluate_point(policy, env_id: str, space, overrides: Optional[dict] = None,
                   seed: int = 0, nom_return: Optional[float] = None) -> RolloutResult:
    """Run a single point: nominal params overridden by `overrides`. Test-facing."""
    params = dict(space.nominal_params)
    if overrides:
        params.update(overrides)
    if nom_return is None and space.failure.get("threshold_kind") == "fraction_of_nominal":
        nom_return = nominal_return(policy, env_id, space, base_seed=seed)
    return run_rollout(policy, env_id, params, seed, 0,
                       space.rollout["episodes"], space.rollout["max_steps"],
                       space.failure, nominal_return=nom_return)


def make_evaluator(policy, env_id: str, space, nom_return: Optional[float] = None):
    """Return `evaluate(values: dict[name->float], index: int) -> RolloutResult`
    where `values` overrides active params on top of nominal. `index` seeds the
    rollout deterministically."""
    base = dict(space.nominal_params)
    episodes = space.rollout["episodes"]
    max_steps = space.rollout["max_steps"]
    failure = space.failure
    seed = space.seed

    def evaluate(values: dict, index: int) -> RolloutResult:
        params = dict(base)
        params.update(values)
        return run_rollout(policy, env_id, params, seed, index, episodes,
                           max_steps, failure, nominal_return=nom_return)

    return evaluate
