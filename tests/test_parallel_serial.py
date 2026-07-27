"""Acceptance 5: parallel and serial execution produce identical results."""
import numpy as np
from sim2real_score import run_analysis
from fixtures.policies import friction_overfit_policy, linear_space

ACTIVE = ("friction", "mass", "action_latency")


def test_parallel_equals_serial():
    space = linear_space(active=ACTIVE)
    serial = run_analysis(friction_overfit_policy(), "linear", space, seed=0,
                          jobs=1, serial=True, sobol_base=16)
    parallel = run_analysis(friction_overfit_policy(), "linear", space, seed=0,
                            jobs=4, serial=False, sobol_base=16)
    assert serial.score == parallel.score
    assert np.array_equal(serial.sensitivity.ST, parallel.sensitivity.ST)
    assert serial.to_json() == parallel.to_json()
