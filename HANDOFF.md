# HANDOFF

Repo: https://github.com/TarunYadgirkar/sim2real-score (public)
Local: `~/TarunsCode/sim2real-score`, venv at `.venv` (Python 3.11).

```bash
cd ~/TarunsCode/sim2real-score && .venv/bin/python -m pytest
```

## What works

Everything in `SPEC.md` §7 is implemented and covered by passing tests — the
suite runs with **no skips** on this machine (all optional backends installed).

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

1. **Ground truth is the synthetic `linear` env, not a trained MuJoCo policy.**
   Establishing that a *trained* net is provably friction- vs latency-fragile is
   research-grade and non-deterministic in CI. MuJoCo is covered by a pipeline
   smoke test, not by sensitivity ground truth. (DECISIONS D4/D8.)
2. **MuJoCo friction/damping/mass are global multipliers**, applied to every geom
   and DOF, not per-body or per-geom.
3. **Vector observations only**; policies are treated as stateless `obs -> action`
   (no recurrent state carried across steps).
4. **Score is mean success rate** over the sampled space — deliberately simple
   and monotone. Worst-case or margin-weighted variants aren't implemented.
5. `numpy<2` is pinned in this venv because torch 2.2.2 (last Intel-macOS build)
   rejects numpy 2.x. Core tool is unaffected.

## Three highest-value next steps

1. **Train a real SB3 policy on Hopper/Walker2d and validate the ranking against
   it.** The strongest remaining gap: the discrimination logic is proven on an
   analytic system but never on a learned controller. Train two policies with
   deliberately different DR exposure (one trained at fixed friction, one with
   friction randomized), then assert the tool separates them. This is the test
   that would make the tool credible for actual sim2real work.
2. **Close the loop on the suggested DR config.** Retrain over the emitted ranges
   and re-score; assert the score improves and the top-ranked parameter changes.
   That turns the DR suggestion from a plausible heuristic into a measured claim,
   and it's the feature most likely to be wrong in a way nothing currently
   catches.
3. **Per-body/per-geom randomization for MuJoCo.** Global multipliers can't
   express the failure that actually bites in practice (e.g. one joint's damping,
   the foot's friction). Needs a selector syntax in the YAML (`friction.foot`)
   and index resolution in `MujocoControlEnv`.

Lower value but easy: caching nominal rollouts across bisection calls (bisection
re-evaluates nominal per parameter), and a `--resume` for long sweeps.
