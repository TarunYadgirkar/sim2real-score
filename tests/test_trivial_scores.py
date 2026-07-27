"""Acceptance 3: trivially-robust policy scores high, random scores low."""
from sim2real_score import run_analysis
from fixtures.policies import zero_policy, random_policy, linear_space
from fixtures.checks import SCORE_HIGH, SCORE_LOW


def test_zero_action_scores_high():
    space = linear_space(active=("friction", "mass", "damping", "actuator_gain",
                                 "obs_noise", "action_latency", "sensor_dropout"))
    res = run_analysis(zero_policy(), "linear", space, seed=0, jobs=1, sobol_base=16)
    assert res.score >= SCORE_HIGH


def test_random_action_scores_low():
    space = linear_space(active=("friction", "mass", "damping", "actuator_gain",
                                 "obs_noise", "action_latency", "sensor_dropout"))
    res = run_analysis(random_policy(), "linear", space, seed=0, jobs=1, sobol_base=16)
    assert res.score <= SCORE_LOW


def test_robust_beats_random_with_margin():
    space = linear_space(active=("friction", "mass", "action_latency"))
    s_zero = run_analysis(zero_policy(), "linear", space, seed=0, jobs=1, sobol_base=16).score
    s_rand = run_analysis(random_policy(), "linear", space, seed=0, jobs=1, sobol_base=16).score
    assert s_zero > s_rand + 30.0
