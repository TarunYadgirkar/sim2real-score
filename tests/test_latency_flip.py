"""Acceptance 2 (core): a policy robust to friction but fragile to latency ->
tool ranks latency top, and the friction-vs-latency ranking flips vs the
friction-overfit policy."""
from sim2real_score import run_analysis, evaluate_point
from fixtures.policies import (friction_overfit_policy, latency_fragile_policy,
                               linear_space)
from fixtures.checks import brackets_boundary

ACTIVE = ("friction", "mass", "action_latency")


def test_ground_truth_latency_fragile_friction_robust():
    pol = latency_fragile_policy()
    space = linear_space(active=ACTIVE)
    nom = evaluate_point(pol, "linear", space, {}, seed=0)
    high_lat = evaluate_point(pol, "linear", space, {"action_latency": space.high("action_latency")}, seed=0)
    low_fric = evaluate_point(pol, "linear", space, {"friction": space.low("friction")}, seed=0)
    assert nom.success_rate > 0.5
    assert high_lat.success_rate < 0.5     # latency breaks it
    assert low_fric.success_rate > 0.5     # friction drop does not


def test_tool_ranks_latency_top():
    pol = latency_fragile_policy()
    space = linear_space(active=ACTIVE)
    res = run_analysis(pol, "linear", space, seed=0, jobs=1, sobol_base=32)
    assert res.sensitivity.ranking()[0] == "action_latency"


def test_tool_locates_latency_breaking_point():
    pol = latency_fragile_policy()
    space = linear_space(active=ACTIVE)
    res = run_analysis(pol, "linear", space, seed=0, jobs=1, sobol_base=16)
    bp = res.breaking_points["action_latency"]
    assert bp.high is not None
    assert space.nominal("action_latency") < bp.high <= space.high("action_latency")
    assert brackets_boundary(evaluate_point, pol, "linear", space, "action_latency",
                             bp.high, "high", margin=1.0)


def test_ranking_flips_between_the_two_policies():
    space = linear_space(active=ACTIVE)
    rank_fric = run_analysis(friction_overfit_policy(), "linear", space, seed=0, jobs=1, sobol_base=32).sensitivity.ranking()
    rank_lat = run_analysis(latency_fragile_policy(), "linear", space, seed=0, jobs=1, sobol_base=32).sensitivity.ranking()
    assert rank_fric[0] == "friction"
    assert rank_lat[0] == "action_latency"
