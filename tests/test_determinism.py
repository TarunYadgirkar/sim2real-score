"""Acceptance 4: fixed seed -> identical score, sensitivity, breaking points,
and serialized result."""
import numpy as np
from sim2real_score import run_analysis
from fixtures.policies import friction_overfit_policy, linear_space

ACTIVE = ("friction", "mass", "action_latency")


def _run():
    return run_analysis(friction_overfit_policy(), "linear", linear_space(active=ACTIVE),
                        seed=0, jobs=1, sobol_base=16)


def test_score_and_sensitivity_reproducible():
    a, b = _run(), _run()
    assert a.score == b.score
    assert np.array_equal(a.sensitivity.ST, b.sensitivity.ST)
    assert a.sensitivity.ranking() == b.sensitivity.ranking()


def test_breaking_points_reproducible():
    a, b = _run(), _run()
    assert a.breaking_points.keys() == b.breaking_points.keys()
    for k in a.breaking_points:
        assert a.breaking_points[k].low == b.breaking_points[k].low
        assert a.breaking_points[k].high == b.breaking_points[k].high


def test_serialized_report_reproducible():
    a, b = _run(), _run()
    assert a.to_json() == b.to_json()
