"""Targeted randomization: `friction.foot` perturbs one geom, not the whole model.

Global multipliers cannot express the failures that actually bite in practice --
one joint's damping, one foot's friction -- because scaling everything at once
changes the task rather than probing a weak spot."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from sim2real_score import RandomizationSpace, make_env
from sim2real_score.randomization.space import base_param, split_target

ENV_ID = "Hopper-v5"


def test_param_name_splitting():
    assert split_target("friction.foot_geom") == ("friction", "foot_geom")
    assert split_target("friction") == ("friction", None)
    assert base_param("damping.thigh_joint") == "damping"


def test_targeted_friction_changes_only_that_geom():
    env = make_env(ENV_ID)
    model = env.env.unwrapped.model
    nominal = np.array(model.geom_friction, copy=True)

    env.set_domain_params({"friction.foot_geom": 2.0})
    changed = np.array(model.geom_friction, copy=True)

    import mujoco
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
    assert foot >= 0, "Hopper should have a geom named 'foot_geom'"
    assert np.allclose(changed[foot], nominal[foot] * 2.0)

    others = [i for i in range(model.ngeom) if i != foot]
    assert np.allclose(changed[others], nominal[others]), "other geoms were touched"


def test_targeted_and_global_compose():
    env = make_env(ENV_ID)
    model = env.env.unwrapped.model
    nominal = np.array(model.geom_friction, copy=True)

    env.set_domain_params({"friction": 0.5, "friction.foot_geom": 4.0})
    changed = np.array(model.geom_friction, copy=True)

    import mujoco
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
    assert np.allclose(changed[foot], nominal[foot] * 0.5 * 4.0)
    others = [i for i in range(model.ngeom) if i != foot]
    assert np.allclose(changed[others], nominal[others] * 0.5)


def test_repeated_application_does_not_compound():
    """Every call must be interpreted against the nominal model, not the current
    one, or a sweep's later points would inherit earlier perturbations."""
    env = make_env(ENV_ID)
    model = env.env.unwrapped.model
    nominal = np.array(model.geom_friction, copy=True)
    for _ in range(3):
        env.set_domain_params({"friction.foot_geom": 2.0})
    import mujoco
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
    assert np.allclose(model.geom_friction[foot], nominal[foot] * 2.0)


def test_targeted_mass_and_damping():
    import mujoco
    env = make_env(ENV_ID)
    model = env.env.unwrapped.model
    nominal_mass = np.array(model.body_mass, copy=True)
    nominal_damping = np.array(model.dof_damping, copy=True)

    env.set_domain_params({"mass.torso": 3.0, "damping.leg_joint": 0.5})

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    assert np.allclose(model.body_mass[torso], nominal_mass[torso] * 3.0)
    others = [i for i in range(model.nbody) if i != torso]
    assert np.allclose(model.body_mass[others], nominal_mass[others])

    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "leg_joint")
    dof = model.jnt_dofadr[joint]
    assert np.allclose(model.dof_damping[dof], nominal_damping[dof] * 0.5)


def test_unknown_target_fails_loudly():
    env = make_env(ENV_ID)
    with pytest.raises(ValueError, match="no_such_geom"):
        env.set_domain_params({"friction.no_such_geom": 2.0})


def test_space_accepts_targeted_params():
    space = RandomizationSpace.from_dict({
        "env": ENV_ID,
        "params": {"friction.foot": {"nominal": 1.0, "low": 0.5, "high": 2.0,
                                     "kind": "multiplicative"}},
    })
    assert "friction.foot" in space.active_names
