# PROGRESS

Updated every ~30 min of work. Newest at top.

## Status: feature-complete against SPEC.md §7. Full suite green, no skips.

- [x] Env recon, repo init, `src/` tree
- [x] SPEC.md (authoritative) + DECISIONS/DEPENDENCIES/PROGRESS
- [x] pyproject + uv venv + core deps
- [x] Acceptance suite written and watched fail (RED, committed separately)
- [x] Policies: protocol, in-memory, SB3 / TorchScript / ONNX loaders
- [x] Envs: built-in `linear` (spring-mass), MuJoCo wrapper, registry + defaults
- [x] Randomization: YAML space + DomainShiftWrapper (gain/noise/latency/dropout)
- [x] Rollout: deterministic runner (SeedSequence), serial/parallel executor
- [x] Ground-truth fixtures decoupled by mechanism (the hard part — see D8)
- [x] Sweep: coarse grid + per-parameter bisection
- [x] Sensitivity: Saltelli + Sobol S1/ST/S2, ranked by total order
- [x] Score, suggested DR config, JSON serialization
- [x] HTML report with inlined plots; CLI (`run`, `envs`)
- [x] All optional backends installed; loaders + MuJoCo smoke verified
- [x] README + HANDOFF + demo report checked in

## Bugs found and fixed along the way
1. **Fixtures measured backwards.** The intended friction-overfit policy was
   actually latency-fragile and friction-robust — the two axes are physically
   coupled if fragility comes from the same mechanism. Fixed by using two
   independent mechanisms (D8), found by empirical search, re-verified by the
   `test_ground_truth_*` tests.
2. **SALib sampling unseeded** → identical runs gave different indices and
   scores. Caught by the determinism acceptance test (D9).
3. **Loaded policies unpicklable** → every analysis of a policy *file* silently
   fell back to serial. Fixed by pickling the load-spec; added a guard test so
   the parallel==serial test can't pass vacuously (D10).
4. **DR config suggested a degenerate range** (`friction.low: 0.001`). Fixed by
   clamping expansion to the breaking point's scale; test written first (D11).

## Blocked
- None. `BLOCKED.md` was never needed.

## Next (see HANDOFF.md for detail)
1. Validate ranking against a *trained* SB3 policy on Hopper/Walker2d.
2. Close the loop: retrain over the suggested DR config, assert the score improves.
3. Per-body/per-geom MuJoCo randomization instead of global multipliers.

## Milestones / commits
- `1b0b976` test: spec + failing acceptance suite (RED)
- `a691d3d` feat: core implementation (WIP)
- `ee18715` feat: sweep, Sobol, score, analysis (acceptance 1–5 green)
- `868e9b3` feat: MuJoCo envs, HTML report, CLI (full suite green)
- (next) fix: parallel for loaded policies, DR range scale, docs
