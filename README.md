# sim2real-score

Take a trained control policy and a MuJoCo environment, systematically find where
the policy breaks under domain shift, and produce a robustness report plus the
domain-randomization ranges to train over next.

- **Loads** stable-baselines3 checkpoints, TorchScript `.pt`, and ONNX policies.
- **Randomizes** friction, mass, damping, actuator gain, observation noise,
  action latency, and sensor dropout — configured in YAML, with defaults per env.
- **Sweeps** a coarse grid to find failure regions, then bisects to locate the
  failure boundary per parameter. Rollouts run in parallel across CPU cores.
- **Ranks** parameters by **Sobol total-order indices**, so interactions count —
  one-at-a-time deltas would miss exactly the couplings that matter.
- **Outputs** a single robustness score, per-parameter breaking points, an
  interaction matrix, a self-contained HTML report, and a suggested DR config.

## Install

```bash
git clone https://github.com/TarunYadgirkar/sim2real-score.git
cd sim2real-score
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

Optional backends, installed only if you need them:

```bash
uv pip install -e ".[torch,onnx,sb3,mujoco]"
```

> On Intel macOS, torch stops at 2.2.2, which rejects numpy 2.x. Pin `numpy<2`
> if you install the torch/sb3 extras there. The core tool has no torch or
> MuJoCo dependency.

## Usage

```bash
sim2real-score run --policy path/to/policy.pt --env Hopper-v5 --out report_dir
```

```
robustness score : 87.5/100
most sensitive   : friction, mass, actuator_gain, ...
  friction        below 0.382
  actuator_gain   above 1.49
report           : report_dir/report.html
suggested DR     : report_dir/dr_config.yaml
```

Writes `report.html` (self-contained, plots inlined), `dr_config.yaml` (ready to
train against), and `result.json`.

Useful flags: `--policy-kind {auto,sb3,torch,onnx}`, `--config space.yaml`,
`--seed N`, `--jobs N`, `--serial`, `--sobol-base N`, `--grid-res N`.
`sim2real-score envs` lists the built-in env defaults.

### Python API

```python
from sim2real_score import run_analysis, default_space, load_policy
from sim2real_score.report import build_report

policy = load_policy("policy.onnx")
result = run_analysis(policy, "Hopper-v5", default_space("Hopper-v5"), seed=0, jobs=8)

print(result.score)                        # 0..100
print(result.sensitivity.ranking())         # most influential first
print(result.breaking_points["friction"])   # BreakingPoint(low=..., high=...)
build_report(result, "out/")
```

## Policy interfaces

| Kind | File | Contract |
|---|---|---|
| `sb3` | `.zip` | Any SB3 algorithm; called with `deterministic=True`. |
| `torch` | `.pt` | **TorchScript** module, float32 `[N, obs_dim] -> [N, action_dim]`. |
| `onnx` | `.onnx` | Single input/output, float32 `[N, obs_dim] -> [N, action_dim]`. |

`examples/export_example_policies.py` exports working `.pt` and `.onnx` policies
in the expected form.

## Randomization space

```yaml
env: Hopper-v5
seed: 0
rollout: { episodes: 3, max_steps: 300 }
failure: { metric: return, threshold: 0.6, threshold_kind: fraction_of_nominal }
params:
  friction:       {nominal: 1.0, low: 0.5, high: 2.0, kind: multiplicative}
  action_latency: {nominal: 0,   low: 0,   high: 3,   kind: int}
  obs_noise:      {nominal: 0.0, low: 0.0, high: 0.05, kind: additive_std}
  sensor_dropout: {nominal: 0.0, low: 0.0, high: 0.2, kind: probability}
```

A parameter is swept only when `low != high`, so pinning one to nominal removes
it from the analysis. `threshold_kind` is `mean_reward`, `absolute`, or
`fraction_of_nominal`. `friction`/`mass`/`damping` are applied to the simulator
model; the rest are applied by a wrapper and work on any environment.

## How the score works

`robustness_score = 100 × mean(success_rate)` over the Sobol-sampled space — the
share of the randomization space where the policy still meets its success
criterion. Sensitivity analyses the **failure indicator** `1 - success_rate`, so
the variance being decomposed is variance in what actually breaks the policy.
A large gap between `ST` and `S1` for a parameter means it matters mostly *in
combination* with others.

The suggested DR config pushes each range past its measured breaking point,
further for parameters with a high total-order index, while keeping ranges on the
same physical scale as the value they broke at.

## Determinism

Every rollout is a pure function of `(policy, params, seed, index)`; per-episode
seeds derive from `numpy.SeedSequence`, never from wall clock or worker identity.
Consequently **serial and parallel runs produce byte-identical `result.json`**,
and reports reproduce exactly. This is enforced by the test suite, which also
asserts the parallel path genuinely engages rather than silently falling back.

## Tests

```bash
pytest
```

The suite is the ground truth (see `SPEC.md` §7). It builds policies with *known*
fragility from distinct physical mechanisms — one that cancels friction at
nominal (breaks when friction drops, tolerant of delay) and one with strong
derivative gain (shrugs off friction, breaks under delay) — and asserts the tool
ranks each correctly and that **the ranking flips between them**. Each fixture's
ground-truth property is verified by direct measurement, so the tests check the
tool rather than a circular fixture.

Tests for optional backends skip cleanly when the backend is absent.

## Layout

```
src/sim2real_score/
  policies/     protocol, in-memory policies, sb3/torchscript/onnx loaders
  envs/         linear (built-in), mujoco wrapper, registry + defaults
  randomization/ space (YAML) + DomainShiftWrapper
  rollout/      deterministic runner, serial/parallel executor
  sweep/        coarse grid, bisection
  sensitivity/  Saltelli sampling + Sobol S1/ST/S2
  analysis.py   orchestration, score, suggested DR config
  report/       plots + self-contained HTML
```

`demo/report.html` is a checked-in example report.

## License

MIT
