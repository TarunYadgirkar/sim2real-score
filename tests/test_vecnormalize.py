"""SB3 policies for MuJoCo are almost always trained under VecNormalize. Feeding
such a policy raw observations silently produces a different (worse) policy than
the one that was trained, which would make any robustness verdict meaningless."""
import numpy as np
import pytest

from sim2real_score import load_policy

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A tiny VecNormalize-wrapped model with non-trivial obs statistics."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    out = tmp_path_factory.mktemp("sb3")
    venv = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
    model = PPO("MlpPolicy", venv, seed=0, n_steps=64, batch_size=64, device="cpu")
    model.learn(total_timesteps=256)
    zip_path = out / "model.zip"
    vn_path = out / "vecnormalize.pkl"
    model.save(str(zip_path))
    venv.save(str(vn_path))
    venv.close()
    return str(zip_path), str(vn_path)


def test_vecnormalize_stats_are_applied(trained):
    """Loading with the stats must reproduce predict-on-normalized-obs, and must
    differ from the un-normalized result (otherwise the test proves nothing)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym

    zip_path, vn_path = trained
    raw = np.array([[0.6, -0.5, 3.0]], dtype=np.float32)

    venv = VecNormalize.load(vn_path, DummyVecEnv([lambda: gym.make("Pendulum-v1")]))
    venv.training = False
    expected_obs = venv.normalize_obs(raw)
    model = PPO.load(zip_path, device="cpu")
    expected_action, _ = model.predict(expected_obs, deterministic=True)
    unnormalized_action, _ = model.predict(raw, deterministic=True)
    venv.close()

    policy = load_policy(zip_path, kind="sb3", vecnormalize=vn_path)
    got = policy.predict(raw)

    assert np.allclose(got, expected_action, atol=1e-5)
    assert not np.allclose(expected_action, unnormalized_action, atol=1e-6), (
        "obs statistics are too close to identity for this test to be meaningful")


def test_without_stats_observations_are_untouched(trained):
    from stable_baselines3 import PPO

    zip_path, _ = trained
    raw = np.array([[0.6, -0.5, 3.0]], dtype=np.float32)
    expected, _ = PPO.load(zip_path, device="cpu").predict(raw, deterministic=True)
    policy = load_policy(zip_path, kind="sb3")
    assert np.allclose(policy.predict(raw), expected, atol=1e-5)


def test_normalized_policy_survives_pickling(trained):
    """Parallel workers receive the policy by pickle; the stats must ride along."""
    import pickle

    zip_path, vn_path = trained
    raw = np.array([[0.6, -0.5, 3.0]], dtype=np.float32)
    policy = load_policy(zip_path, kind="sb3", vecnormalize=vn_path)
    revived = pickle.loads(pickle.dumps(policy))
    assert np.allclose(revived.predict(raw), policy.predict(raw), atol=1e-6)
