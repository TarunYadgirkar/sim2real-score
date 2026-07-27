# SPEC — sim2real-score

Systematically find where a trained control policy breaks under domain shift, and
produce a robustness report + a suggested domain-randomization (DR) config for
the next training run.

Status: authoritative. Acceptance tests are derived from this file and are the
ground truth. Implementation must satisfy them without weakening them.

---

## 1. Scope

Given (a) a trained control policy and (b) a MuJoCo/gymnasium (or built-in
synthetic) environment, the tool:

1. Loads the policy from one of: stable-baselines3 checkpoint, raw TorchScript
   `.pt`, or ONNX file. Also accepts an in-process `Policy` object.
2. Reads a **randomization space** from YAML (friction, mass, damping, actuator
   gain, observation noise, action latency, sensor dropout). Ships sensible
   per-env defaults.
3. **Sweeps** the space: coarse grid to find failure regions, then bisection to
   locate the failure boundary (breaking point) per parameter. Rollouts run in
   parallel across CPU cores.
4. **Sensitivity analysis**: ranks parameters by how much they drive failure
   using **Sobol indices** (variance-based, captures interactions), not
   one-at-a-time deltas.
5. **Outputs**: a single robustness score, per-parameter breaking points, a
   parameter-interaction matrix, and a self-contained HTML report with plots.
6. Emits a **suggested DR config**: the ranges to train over next, derived from
   where the policy is weak.

Runs headless. No proprietary simulator required.

---

## 2. Domain-shift parameters

Two families, applied per rollout:

**Dynamics params** (env-specific; the env must implement them):
- `friction`   — viscous/contact friction coefficient (multiplicative on nominal)
- `mass`       — body/link mass (multiplicative)
- `damping`    — joint/velocity damping (multiplicative)

**Universal wrapper params** (apply to any env, via `DomainShiftWrapper`):
- `actuator_gain`  — scale on action before it reaches the env (multiplicative)
- `obs_noise`      — additive Gaussian noise std on observations
- `action_latency` — integer step delay on applied actions (buffer)
- `sensor_dropout` — per-step probability an observation is held (stale) instead
                     of refreshed

An env declares which dynamics params it supports (`supported_domain_params`).
Unsupported dynamics params are ignored with a logged warning; universal params
always apply.

---

## 3. Randomization space (YAML)

```yaml
env: linear                 # env id (built-in or gymnasium/MuJoCo)
seed: 0
rollout:
  episodes: 5               # episodes averaged per parameter point
  max_steps: 200
failure:
  metric: return            # episode metric to threshold
  threshold: 0.6            # see threshold_kind
  threshold_kind: fraction_of_nominal   # or "absolute"
params:
  friction:       {nominal: 1.0, low: 0.5, high: 2.0, kind: multiplicative}
  mass:           {nominal: 1.0, low: 0.5, high: 2.0, kind: multiplicative}
  damping:        {nominal: 1.0, low: 0.5, high: 2.0, kind: multiplicative}
  actuator_gain:  {nominal: 1.0, low: 0.5, high: 1.5, kind: multiplicative}
  obs_noise:      {nominal: 0.0, low: 0.0, high: 0.3, kind: additive_std}
  action_latency: {nominal: 0,   low: 0,   high: 6,   kind: int}
  sensor_dropout: {nominal: 0.0, low: 0.0, high: 0.6, kind: probability}
```

- A param is **active** in the sweep iff `low != high`.
- `nominal` defines the nominal environment (used for the reference return).
- `kind` documents how the raw value is interpreted when applied.

---

## 4. Public interfaces

### 4.1 Policy

```python
class Policy(Protocol):
    obs_dim: int
    action_dim: int
    def predict(self, obs: np.ndarray) -> np.ndarray: ...   # deterministic
```

Loader:

```python
load_policy(path, kind="auto", obs_dim=None, action_dim=None) -> Policy
# kind in {"auto","sb3","torch","onnx"}
```

- **sb3**: a stable-baselines3 `.zip`; uses `predict(obs, deterministic=True)`.
- **torch**: a **TorchScript** module (`torch.jit.save`) mapping a float32 tensor
  `[N, obs_dim]` -> `[N, action_dim]`. (Documented interface; no side-car file.)
- **onnx**: single input `[N, obs_dim]` float32 -> single output `[N, action_dim]`.
- **auto**: dispatch by extension/content (`.zip`->sb3, `.onnx`->onnx, `.pt`->torch).

### 4.2 Environment

```python
class ControlEnv(Protocol):
    observation_space; action_space
    supported_domain_params: set[str]
    def reset(self, *, seed=None) -> tuple[obs, info]: ...
    def step(self, action) -> tuple[obs, reward, terminated, truncated, info]: ...
    def set_domain_params(self, params: dict) -> None: ...   # dynamics only
```

`make_env(env_id, seed=None) -> ControlEnv` factory. Built-in `linear`; gymnasium
ids (`Hopper-v5`, `Walker2d-v5`, `Reacher-v5`) via a wrapper when MuJoCo present.

### 4.3 Randomization

```python
RandomizationSpace.from_yaml(path) -> RandomizationSpace
space.active_params -> list[ParamSpec]      # low != high
space.nominal_params -> dict[str, float]
ParamSpec(name, nominal, low, high, kind, is_int)
```

### 4.4 Rollout

```python
run_rollout(policy, env_factory, params, seed, episodes, max_steps)
    -> RolloutResult(mean_return, min_return, success_rate, n_episodes)
```

- Deterministic given `seed`. Env reset seed and all noise/dropout RNG derive
  from `seed` only (never from wall clock or worker identity).
- `success_rate` = fraction of episodes with `return >= failure_threshold`.

### 4.5 Sweep

```python
coarse_grid(space, evaluate, resolution, seed) -> GridResult
bisect_boundary(space, param, evaluate, seed, tol) -> BreakingPoint(low, high)
```

- `evaluate(params, seed) -> RolloutResult` is the (parallelizable) unit of work.
- `bisect_boundary` moves `param` from nominal toward each bound, and bisects to
  the value where `success_rate` crosses 0.5. `low`/`high` are the breaking
  values on each side (or `None` if the policy never fails within the bound).

### 4.6 Sensitivity

```python
sobol_analysis(space, evaluate, n_base, seed) -> SobolResult(names, S1, ST, S2)
```

- Saltelli sampling over active params; Sobol first-order `S1`, total-order `ST`,
  and second-order interaction matrix `S2`. Output analysed = failure indicator
  `1 - success_rate` (variance driven by what breaks the policy).
- Ranking = params sorted by `ST` descending.

### 4.7 Score

```python
robustness_score(evaluate, space, n_samples, seed) -> float   # 0..100
```

- `100 * mean(success_rate)` over the sampled randomization space. Monotone: more
  of the space survived -> higher score. Trivially-robust policy -> ~100; random
  policy -> low.

### 4.8 Report / DR config

```python
build_report(result: AnalysisResult, out_dir) -> path_to_html
suggest_dr_config(result: AnalysisResult) -> dict            # + YAML dump
run_analysis(policy, env_id, space, seed, jobs, serial) -> AnalysisResult
```

- HTML is **self-contained** (plots inlined as base64 PNG). Contains: score,
  per-param sensitivity bar chart, breaking-points table, interaction heatmap,
  and the suggested DR config.
- Suggested DR ranges: for each active param, widen training coverage toward the
  measured breaking point, widened more for higher-`ST` (weaker) params. Robust
  params keep near-nominal ranges. Emitted as a valid randomization YAML.

### 4.9 CLI

```
sim2real-score run --policy PATH --env ENV [--policy-kind KIND]
    [--config space.yaml] [--out DIR] [--seed N] [--jobs N] [--serial]
    [--sobol-base N] [--grid-res N]
```

Writes `report.html`, `dr_config.yaml`, `result.json` into `--out`.

---

## 5. Determinism contract

- Every rollout is a pure function of `(policy, params, seed, episodes,
  max_steps)`.
- Sobol/grid/bisection derive each sample's seed deterministically from the base
  seed and the sample index.
- **Serial and parallel execution produce identical results** (same score, same
  ST, same breaking points, byte-identical `result.json`).
- Reports are reproducible: same inputs -> same `result.json`.

---

## 6. Ground-truth fixtures (for acceptance)

Built on the synthetic `linear` env — a regulated damped point mass — so behavior
is analytic, fast, and deterministic:

- **friction-overfit policy**: relies on the environment's own friction for
  closed-loop damping (low derivative gain). Fragile to friction (destabilizes
  when friction drops), robust to action latency (low gain -> large delay margin).
- **latency-fragile policy**: supplies its own strong damping (high gain), robust
  across the friction range, but high gain + action latency -> instability.
  Fragile to latency, robust to friction.
- **trivial-robust policy**: constant zero action on the stable linear env (mass
  returns to origin on its own) -> high score.
- **random policy**: uniform random actions -> low score.

Each fixture's ground-truth property is established by **direct measurement**
(sweep the axis, observe return collapse) independent of the sensitivity
machinery, so the acceptance test checks the *tool*, not a circular fixture.

---

## 7. Acceptance criteria (see `tests/`)

1. friction-overfit: tool ranks `friction` as top-`ST`; friction breaking point
   located within tolerance of the directly-measured boundary.
2. latency-fragile: tool ranks `action_latency` as top-`ST`. The friction-vs-
   latency ranking flips between fixture 1 and 2. (Core discriminating test.)
3. trivial-robust scores high (>= 90); random scores low (<= 40); high > low with
   margin.
4. Determinism: same seed -> identical score, ST, breaking points, `result.json`.
5. Serial and parallel execution produce identical results.
6. Policy loaders: a saved linear policy round-trips through `.pt`, ONNX, and SB3
   loaders to matching `predict` outputs (skipped if the backend lib is absent).
7. Smoke: full pipeline runs headless on a MuJoCo env and emits a report (skipped
   if MuJoCo absent).

---

## 8. Out of scope

- Training policies (beyond the trivial hardcoded ground-truth fixtures).
- Non-MuJoCo proprietary simulators.
- Recurrent/stateful policy internals (policies treated as `obs -> action`).
- GPU. Real-robot deployment. Online/interactive tuning.
- Image observations (vector observations only).
