"""Acceptance 1: a policy overfit to nominal friction -> tool ranks friction top
and locates its breaking point."""
from sim2real_score import run_analysis, evaluate_point
from fixtures.policies import friction_overfit_policy, linear_space
from fixtures.checks import brackets_boundary

ACTIVE = ("friction", "mass", "action_latency")


def test_ground_truth_friction_fragile_latency_robust():
    pol = friction_overfit_policy()
    space = linear_space(active=ACTIVE)
    nom = evaluate_point(pol, "linear", space, {}, seed=0)
    low_fric = evaluate_point(pol, "linear", space, {"friction": space.low("friction")}, seed=0)
    high_lat = evaluate_point(pol, "linear", space, {"action_latency": space.high("action_latency")}, seed=0)
    assert nom.success_rate > 0.5
    assert low_fric.success_rate < 0.5     # friction drop breaks it
    assert high_lat.success_rate > 0.5     # action latency does not


def test_tool_ranks_friction_top():
    pol = friction_overfit_policy()
    space = linear_space(active=ACTIVE)
    res = run_analysis(pol, "linear", space, seed=0, jobs=1, sobol_base=32)
    assert res.sensitivity.ranking()[0] == "friction"


def test_tool_locates_friction_breaking_point():
    pol = friction_overfit_policy()
    space = linear_space(active=ACTIVE)
    res = run_analysis(pol, "linear", space, seed=0, jobs=1, sobol_base=16)
    bp = res.breaking_points["friction"]
    assert bp.low is not None
    assert space.low("friction") < bp.low < space.nominal("friction")
    assert brackets_boundary(evaluate_point, pol, "linear", space, "friction", bp.low, "low")
