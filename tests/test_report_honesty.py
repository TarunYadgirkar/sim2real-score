"""The report must not present a ranking it cannot support.

Two cases where Sobol indices carry no information but a bar chart would still
render, implying one parameter matters more than another:
  - degenerate: the policy behaved identically everywhere, so there is no
    variance to attribute (all indices zero);
  - single active parameter: the one parameter explains all of the variance by
    construction, so ST ~ 1 says nothing about how fragile the policy is.
"""
import numpy as np
import pytest

from sim2real_score import ConstantPolicy, LinearPolicy, run_analysis
from sim2real_score.report import build_report
from fixtures.policies import linear_space


def test_uniformly_robust_policy_is_flagged_degenerate(tmp_path):
    res = run_analysis(ConstantPolicy(np.zeros(1)), "linear",
                       linear_space(active=("friction", "mass")), seed=0,
                       jobs=1, sobol_base=8)
    assert res.sensitivity.degenerate
    assert np.allclose(res.sensitivity.ST, 0.0)

    html = open(build_report(res, str(tmp_path))).read()
    assert "no variance" in html.lower(), (
        "report must say why the indices are zero rather than charting zeros")


def test_single_parameter_analysis_is_flagged(tmp_path):
    """With one active parameter the index is ~1 by construction."""
    res = run_analysis(LinearPolicy(np.array([[-2.0, 1.1]])), "linear",
                       linear_space(active=("friction",)), seed=0, jobs=1,
                       sobol_base=8)
    assert len(res.space.active_params) == 1
    html = open(build_report(res, str(tmp_path))).read()
    assert "only one" in html.lower() or "single" in html.lower(), (
        "report must note that a lone parameter trivially explains all variance")


def test_ranking_is_still_reported_when_meaningful(tmp_path):
    res = run_analysis(LinearPolicy(np.array([[-2.0, 1.1]])), "linear",
                       linear_space(active=("friction", "mass", "action_latency")),
                       seed=0, jobs=1, sobol_base=16)
    assert not res.sensitivity.degenerate
    html = open(build_report(res, str(tmp_path))).read()
    assert "no variance" not in html.lower()
