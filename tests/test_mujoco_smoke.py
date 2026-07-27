"""Acceptance 7: the full pipeline runs headless on a MuJoCo env and emits a
report. Skipped if MuJoCo/gymnasium is absent."""
import os
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from sim2real_score import run_analysis, make_env, ConstantPolicy, default_space
from sim2real_score.report import build_report


ENV_ID = "Reacher-v5"


def test_reacher_env_constructs():
    env = make_env(ENV_ID, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    a = np.zeros(env.action_space.shape, dtype=np.float64)
    obs2, r, term, trunc, info = env.step(a)
    assert obs2.shape == obs.shape


def test_pipeline_runs_and_reports(tmp_path):
    env = make_env(ENV_ID, seed=0)
    act_dim = int(np.prod(env.action_space.shape))
    space = default_space(ENV_ID)
    # keep it tiny for a smoke test
    space.rollout["episodes"] = 1
    space.rollout["max_steps"] = 30
    res = run_analysis(ConstantPolicy(np.zeros(act_dim)), ENV_ID, space,
                       seed=0, jobs=1, sobol_base=4, grid_res=3)
    assert 0.0 <= res.score <= 100.0
    html = build_report(res, str(tmp_path))
    assert os.path.exists(html)
    assert os.path.getsize(html) > 1000
