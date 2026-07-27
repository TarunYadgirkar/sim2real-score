"""The suggested domain-randomization config must be trainable, not just wider:
expanding past a breaking point may not collapse a range to a degenerate floor."""
import yaml
from sim2real_score import run_analysis, suggest_dr_config, dump_dr_config
from fixtures.policies import friction_overfit_policy, linear_space

ACTIVE = ("friction", "mass", "action_latency")
MIN_SANE_FRACTION = 0.25   # a suggested range may not drop below 1/4 of the value it broke at


def _result():
    return run_analysis(friction_overfit_policy(), "linear",
                        linear_space(active=ACTIVE), seed=0, jobs=1, sobol_base=16)


def test_suggested_ranges_stay_physical():
    res = _result()
    cfg = suggest_dr_config(res)
    for name, spec in cfg["params"].items():
        assert spec["low"] <= spec["high"]
        if spec["kind"] == "multiplicative":
            assert spec["low"] > 0.0, f"{name}: multiplicative range hits zero"
        if spec["kind"] == "probability":
            assert 0.0 <= spec["low"] and spec["high"] <= 1.0


def test_expansion_past_breaking_point_is_not_degenerate():
    res = _result()
    cfg = suggest_dr_config(res)
    for name, bp in res.breaking_points.items():
        if bp.low is None or bp.low <= 0:
            continue
        low = cfg["params"][name]["low"]
        assert low < bp.low, f"{name}: suggested range must cover past the breaking point"
        assert low >= MIN_SANE_FRACTION * bp.low, (
            f"{name}: suggested low {low:.4g} collapsed far below the breaking "
            f"point {bp.low:.4g}")


def test_dumped_config_reloads_as_a_valid_space(tmp_path):
    from sim2real_score import RandomizationSpace
    res = _result()
    path = dump_dr_config(res, str(tmp_path / "dr.yaml"))
    with open(path) as f:
        loaded = yaml.safe_load(f)
    space = RandomizationSpace.from_dict(loaded)
    assert set(space.params) == set(res.space.params)
    assert "friction" in space.active_names
