"""Validation against *trained* policies, not analytic fixtures.

Two PPO policies are trained on Hopper-v5 (see experiments/train_dr_policies.py):
one only at nominal friction, one with friction randomized every episode. If
domain randomization does what it claims, the tool must be able to see the
difference. Skipped unless the trained policies are present, since training takes
tens of minutes:

    python experiments/train_dr_policies.py --steps 500000

These assertions are about *relative* ordering between the two policies, not
absolute index values, because a trained network's exact sensitivity depends on
the training run.
"""
import json
import os

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco")

from sim2real_score import default_space, load_policy, run_analysis

POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "policies")
ENV_ID = "Hopper-v5"
SOBOL_BASE = 16
EPISODES = 2
MAX_STEPS = 300

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(POLICY_DIR, "manifest.json")),
    reason="trained policies absent; run experiments/train_dr_policies.py")


def _load(name):
    return load_policy(os.path.join(POLICY_DIR, f"{name}.zip"), kind="sb3",
                       vecnormalize=os.path.join(POLICY_DIR, f"{name}_vecnormalize.pkl"))


def _space():
    space = default_space(ENV_ID)
    space.rollout["episodes"] = EPISODES
    space.rollout["max_steps"] = MAX_STEPS
    # Isolate the friction axis: everything else pinned to nominal.
    for name, spec in space.params.items():
        if name != "friction":
            spec.low = spec.high = spec.nominal
    return space


@pytest.fixture(scope="module")
def analyses():
    return {name: run_analysis(_load(name), ENV_ID, _space(), seed=0, jobs=4,
                               sobol_base=SOBOL_BASE)
            for name in ("nominal", "dr")}


def test_training_actually_produced_the_intended_gap():
    """Ground truth, measured during training and independent of this tool: the
    nominal-trained policy must degrade more at low friction than the DR-trained
    one. Without this the comparison below would be meaningless."""
    with open(os.path.join(POLICY_DIR, "manifest.json")) as f:
        returns = json.load(f)["returns"]
    nom_drop = returns["nominal"]["friction_1.0"] - returns["nominal"]["friction_0.5"]
    dr_drop = returns["dr"]["friction_1.0"] - returns["dr"]["friction_0.5"]
    assert nom_drop > dr_drop, (
        f"training did not produce the intended gap: nominal drop {nom_drop:.0f} "
        f"vs dr drop {dr_drop:.0f}")


def test_dr_training_raises_the_robustness_score(analyses):
    assert analyses["dr"].score > analyses["nominal"].score


def test_friction_sensitivity_is_higher_for_the_nominal_trained_policy(analyses):
    def friction_st(res):
        names = res.sensitivity.names
        return float(res.sensitivity.ST[names.index("friction")])
    assert friction_st(analyses["nominal"]) > friction_st(analyses["dr"])


def test_nominal_trained_policy_breaks_earlier(analyses):
    """Whichever policy breaks, the DR-trained one must tolerate more friction
    loss before it does."""
    nom = analyses["nominal"].breaking_points["friction"].low
    dr = analyses["dr"].breaking_points["friction"].low
    if nom is None:
        pytest.skip("nominal-trained policy held across the whole friction range")
    assert dr is None or dr < nom
