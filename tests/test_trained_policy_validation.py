"""Validation against *trained* policies, not analytic fixtures.

Two PPO policies are trained on Hopper-v5 (see experiments/train_dr_policies.py):
one only at nominal friction, one with friction randomized every episode. If
domain randomization does what it claims, the tool must be able to see the
difference. Skipped unless the trained policies are present, since training takes
several minutes:

    python experiments/train_dr_policies.py --steps 500000

Assertions are about *relative* ordering between the two policies, never absolute
index values, because a trained network's exact numbers depend on the run.

Note on what is compared: cross-policy fragility is measured by the **score** and
the **breaking points**, not by Sobol indices. Sobol indices are normalized
variance shares -- they rank parameters *within* one policy. Comparing them
across policies is meaningless, and actively misleading here: the nominal-trained
policy fails almost everywhere, so its failure indicator is nearly constant and
there is little variance left to attribute. See DECISIONS D16.
"""
import json
import os

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco")

from sim2real_score import default_space, load_policy, run_analysis

POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "policies")
MANIFEST = os.path.join(POLICY_DIR, "manifest.json")
ENV_ID = "Hopper-v5"
SOBOL_BASE = 16
EPISODES = 2

pytestmark = pytest.mark.skipif(
    not os.path.exists(MANIFEST),
    reason="trained policies absent; run experiments/train_dr_policies.py")


def _manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def _load(name):
    return load_policy(os.path.join(POLICY_DIR, f"{name}.zip"), kind="sb3",
                       vecnormalize=os.path.join(POLICY_DIR, f"{name}_vecnormalize.pkl"))


def _space():
    space = default_space(ENV_ID)
    space.rollout["episodes"] = EPISODES
    # Read the cap the ground-truth returns were measured under. These policies
    # differ mainly in how long they survive, so analysing under a shorter cap
    # would truncate away the very effect the manifest recorded.
    space.rollout["max_steps"] = _manifest()["eval_max_steps"]
    for name, spec in space.params.items():
        if name != "friction":
            spec.low = spec.high = spec.nominal
    return space


@pytest.fixture(scope="module")
def analyses():
    return {name: run_analysis(_load(name), ENV_ID, _space(), seed=0, jobs=4,
                               sobol_base=SOBOL_BASE)
            for name in ("nominal", "dr")}


def _band(res):
    """The friction interval the policy survives in. `None` on a side means it
    never broke within the swept range there."""
    bp = res.breaking_points["friction"]
    space = res.space.param("friction")
    low = space.low if bp.low is None else bp.low
    high = space.high if bp.high is None else bp.high
    return low, high


def test_training_actually_produced_the_intended_gap():
    """Ground truth, measured during training and independent of this tool: the
    nominal-trained policy must degrade more at low friction than the DR-trained
    one. Without this the comparisons below would be meaningless."""
    returns = _manifest()["returns"]
    nom = returns["nominal"]
    dr = returns["dr"]
    nom_retained = nom["friction_0.5"] / nom["friction_1.0"]
    dr_retained = dr["friction_0.5"] / dr["friction_1.0"]
    assert dr_retained > nom_retained, (
        f"training did not produce the intended gap: nominal retained "
        f"{nom_retained:.0%} of its return at half friction, dr {dr_retained:.0%}")


def test_dr_training_raises_the_robustness_score(analyses):
    assert analyses["dr"].score > analyses["nominal"].score


def test_dr_training_widens_the_survivable_friction_band(analyses):
    """The headline claim: domain randomization should let the policy tolerate a
    wider range of friction, and the tool should measure that."""
    nom_low, nom_high = _band(analyses["nominal"])
    dr_low, dr_high = _band(analyses["dr"])
    assert (dr_high - dr_low) > (nom_high - nom_low)


def test_nominal_trained_policy_breaks_earlier_when_friction_drops(analyses):
    nom_low, _ = _band(analyses["nominal"])
    dr_low, _ = _band(analyses["dr"])
    assert dr_low < nom_low


def test_nominal_trained_policy_survives_only_near_its_training_point(analyses):
    """Overfitting signature: the band is narrow and brackets nominal friction."""
    low, high = _band(analyses["nominal"])
    nominal = analyses["nominal"].space.nominal("friction")
    assert low < nominal < high
    assert (high - low) < 0.5 * (
        analyses["nominal"].space.high("friction")
        - analyses["nominal"].space.low("friction"))
