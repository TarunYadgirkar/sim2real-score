"""Acceptance 5: parallel and serial execution produce identical results."""
import os
import warnings

import numpy as np
from sim2real_score import run_analysis
from sim2real_score.rollout.executor import run_batch
from fixtures.policies import friction_overfit_policy, linear_space

ACTIVE = ("friction", "mass", "action_latency")


def _worker_pid(values, index):
    return os.getpid()


def test_parallel_path_actually_engages():
    """Guards the equality test below: if the executor silently fell back to
    serial, `test_parallel_equals_serial` would pass vacuously."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pids = run_batch(_worker_pid, [({}, i) for i in range(12)], n_jobs=4)
    fallbacks = [str(w.message) for w in caught
                 if "parallel execution unavailable" in str(w.message)]
    assert not fallbacks, f"executor fell back to serial: {fallbacks}"
    assert set(pids) != {os.getpid()}, "work ran in the parent process"


def test_parallel_equals_serial():
    space = linear_space(active=ACTIVE)
    serial = run_analysis(friction_overfit_policy(), "linear", space, seed=0,
                          jobs=1, serial=True, sobol_base=16)
    parallel = run_analysis(friction_overfit_policy(), "linear", space, seed=0,
                            jobs=4, serial=False, sobol_base=16)
    assert serial.score == parallel.score
    assert np.array_equal(serial.sensitivity.ST, parallel.sensitivity.ST)
    assert serial.to_json() == parallel.to_json()
