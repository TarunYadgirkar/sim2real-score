# HANDOFF

Repo: https://github.com/TarunYadgirkar/sim2real-score (public)
Local: `~/TarunsCode/sim2real-score`, venv at `.venv` (Python 3.11).

```bash
cd ~/TarunsCode/sim2real-score && .venv/bin/python -m pytest
```

## What works

Everything in `SPEC.md` §7 is implemented and covered by passing tests, plus the
§7a extensions. **42 passed, 0 skipped** on this machine (all optional backends
installed).

- **Policy loading** — SB3 `.zip`, TorchScript `.pt`, ONNX. Round-trip tested
  against a known linear policy for each backend. Loaded policies are picklable
  by load-spec so they work under parallel execution.
- **Randomization** — YAML space with per-env defaults (`linear`, Hopper,
  Walker2d, Reacher). Dynamics params (friction/mass/damping) applied to the
  MuJoCo model as multipliers on cached nominal values; actuator gain, obs noise,
  action latency, and sensor dropout applied by a wrapper that works on any env.
- **Sweep** — coarse grid (deterministically subsampled past `GRID_CAP`, and it
  reports when it truncates) plus per-parameter bisection to the failure
  boundary, handling both sides and the already-broken-at-nominal case.
- **Sensitivity** — Saltelli sampling, Sobol `S1`/`ST`/`S2` over the failure
  indicator, ranked by total order. Degenerate (zero-variance) cases return zeros
  instead of NaN and are flagged.
- **Score / outputs** — 0–100 robustness score, breaking points, interaction
  matrix, self-contained HTML report (inlined base64 plots, light/dark), and a
  suggested DR config that reloads as a valid randomization space.
- **Determinism** — serial and parallel produce byte-identical `result.json`;
  a guard test asserts the parallel path actually engages so the equality test
  can't pass vacuously. Parallel is real: 3m09s → 1m16s at 487% CPU.
- **CLI** — `sim2real-score run` / `envs`.

Verified end-to-end: a TorchScript friction-overfit policy on `linear` scores
87.5/100 and its friction breaking point is located at 0.382.

## What's stubbed / limited

Nothing is stubbed behind a fake interface — `BLOCKED.md` was never needed. Real
limitations, all deliberate and in `SPEC.md` §8:

1. **The friction-vs-latency discrimination ground truth is the synthetic
   `linear` env.** Establishing that a *trained* net is provably friction- vs
   latency-fragile is research-grade and non-deterministic in CI. Trained
   policies are validated on the friction axis only (see above). (DECISIONS
   D4/D8.)
2. **Sobol indices do not compare fragility across policies** — they are
   normalized variance shares and rank parameters within one analysis. Use the
   score and breaking points to compare policies. (DECISIONS D16.)
3. **Vector observations only**; policies are treated as stateless `obs -> action`
   (no recurrent state carried across steps).
4. **Score is mean success rate** over the sampled space — deliberately simple
   and monotone. Worst-case or margin-weighted variants aren't implemented.
5. `numpy<2` is pinned in this venv because torch 2.2.2 (last Intel-macOS build)
   rejects numpy 2.x. Core tool is unaffected.

## Done since the first handoff

- **Validated on trained policies** (was next step #1). Two PPO policies on
  Hopper, one nominal-trained and one friction-randomized. The tool measures the
  nominal-trained policy surviving friction only in [0.87, 1.09] (score 13.3)
  versus [0.74, 1.38] (score 46.1) for the DR-trained one — a ~3× wider band.
  Policies are committed, so it reruns without training.
- **Per-element targeted randomization** (was #3): `friction.foot_geom`,
  `mass.torso`, `damping.leg_joint`.
- **SB3 VecNormalize support** — without it, an SB3 MuJoCo policy is evaluated
  on observations it was never trained on.

## Three highest-value next steps

1. **Close the loop on the suggested DR config.** Retrain over the emitted
   ranges, re-score, and assert the score improves. Everything needed now exists
   (`experiments/train_dr_policies.py` already trains under a randomized range),
   so this is mostly wiring the emitted YAML into the training wrapper. Still the
   feature most likely to be wrong in a way nothing currently catches — the
   expansion heuristic (D11) has never been checked against an actual retrain.
2. **Make the score robust to the "broken everywhere" regime.** A policy that
   fails across the whole swept range scores near zero and yields no usable
   ranking (observed: nominal-trained Hopper scored 0.0 with four parameters
   active, all indices zero). Adaptive range narrowing — shrink toward nominal
   until some points pass — would turn a useless report into a useful one, and
   it's the most likely first experience for a genuinely fragile policy.
3. **Per-parameter episode budgets / early stopping.** Rollouts dominate
   runtime and most are spent confirming points deep inside a failure region.
   Stopping an episode once the outcome is decided, and spending the saved
   budget near the boundary, would make MuJoCo sweeps practical at higher
   resolution.

Lower value but easy: cache nominal rollouts across bisection calls (bisection
re-evaluates nominal once per parameter), and a `--resume` for long sweeps.
