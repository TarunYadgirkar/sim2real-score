"""Validation experiment: does sim2real-score rank friction correctly for a
*trained* policy, not just for the analytic fixtures?

Trains two PPO policies on the same MuJoCo env:

  nominal  -- trained only at the simulator's default friction
  dr       -- trained with geom friction resampled every episode

Hypothesis: the tool assigns friction a higher total-order Sobol index for the
nominal-trained policy, and locates a friction breaking point closer to nominal
for it than for the DR-trained one. If domain randomization does what it claims,
the tool must be able to see it.

    python experiments/train_dr_policies.py --steps 600000 --out experiments/policies

Writes `<name>.zip` (SB3) plus a `manifest.json` recording the training config.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np

ENV_ID = "Hopper-v5"
FRICTION_RANGE = (0.5, 2.0)


class FrictionRandomizer(gym.Wrapper):
    """Resamples a multiplicative geom-friction factor at every reset."""

    def __init__(self, env, low, high):
        super().__init__(env)
        self.low, self.high = low, high
        self._nominal = np.array(env.unwrapped.model.geom_friction, copy=True)

    def reset(self, **kwargs):
        factor = self.np_random.uniform(self.low, self.high)
        self.env.unwrapped.model.geom_friction[:] = self._nominal * factor
        return self.env.reset(**kwargs)


def make_env_fn(randomize: bool):
    def _fn():
        env = gym.make(ENV_ID)
        if randomize:
            env = FrictionRandomizer(env, *FRICTION_RANGE)
        return env
    return _fn


def train(name: str, randomize: bool, steps: int, out_dir: str, n_envs: int,
          seed: int) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    venv = SubprocVecEnv([make_env_fn(randomize) for _ in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                n_steps=512, batch_size=1024, gae_lambda=0.95, gamma=0.99,
                n_epochs=10, ent_coef=0.0, learning_rate=3e-4, clip_range=0.2)
    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=False)
    elapsed = time.time() - t0

    path = os.path.join(out_dir, f"{name}.zip")
    model.save(path)
    venv.save(os.path.join(out_dir, f"{name}_vecnormalize.pkl"))
    venv.close()
    print(f"  {name}: {steps} steps in {elapsed/60:.1f} min -> {path}")
    return path


def evaluate(path: str, vecnorm_path: str, friction: float, episodes: int = 10,
             seed: int = 0) -> float:
    """Mean return at a given friction multiplier, using the saved obs
    normalization (frozen)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([make_env_fn(False)])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(path, device="cpu")

    inner = venv.venv.envs[0].unwrapped
    nominal = np.array(inner.model.geom_friction, copy=True)
    inner.model.geom_friction[:] = nominal * friction

    venv.seed(seed)
    totals = []
    for _ in range(episodes):
        obs = venv.reset()
        inner.model.geom_friction[:] = nominal * friction
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            total += float(infos[0].get("episode", {}).get("r", reward[0])
                           if False else reward[0])
            done = bool(dones[0])
        totals.append(total)
    venv.close()
    return float(np.mean(totals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--out", default="experiments/policies")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true",
                    help="time a short run and exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.benchmark:
        t0 = time.time()
        train("_benchmark", False, 20_000, args.out, args.n_envs, args.seed)
        rate = 20_000 / (time.time() - t0)
        print(f"{rate:.0f} steps/s -> {args.steps/rate/60:.1f} min per policy")
        return

    print(f"training on {ENV_ID}, {args.steps} steps each")
    paths = {
        "nominal": train("nominal", False, args.steps, args.out, args.n_envs, args.seed),
        "dr": train("dr", True, args.steps, args.out, args.n_envs, args.seed),
    }

    print("\nsanity check (mean return over 10 episodes):")
    report = {}
    for name, path in paths.items():
        vn = os.path.join(args.out, f"{name}_vecnormalize.pkl")
        row = {f"friction_{f}": evaluate(path, vn, f) for f in (1.0, 0.7, 0.5)}
        report[name] = row
        print(f"  {name}: " + "  ".join(f"{k}={v:.0f}" for k, v in row.items()))

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"env": ENV_ID, "steps": args.steps, "seed": args.seed,
                   "friction_range": list(FRICTION_RANGE), "returns": report},
                  f, indent=2)


if __name__ == "__main__":
    main()
