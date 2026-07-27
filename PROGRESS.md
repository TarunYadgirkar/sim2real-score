# PROGRESS

Updated every ~30 min of work. Newest at top.

## Status: SPEC §7 + §7a extensions complete. 42 passed, 0 skipped.

## Session 2 (post-restart) additions
- [x] SB3 VecNormalize support (loader + CLI) — without it an SB3 MuJoCo policy
      is evaluated on observations it never saw in training
- [x] Per-element targeted randomization: `friction.foot_geom`, `mass.torso`,
      `damping.leg_joint`
- [x] Trained-policy validation: two PPO policies on Hopper (nominal vs
      friction-randomized). Tool measures the nominal-trained policy surviving
      friction only in [0.87, 1.09] (score 13.3) vs [0.74, 1.38] (score 46.1)
      for the DR-trained one. Policies committed (316K), reruns without training
- [x] Report no longer charts uninformative indices (degenerate / single-param)
- [x] Second demo report from the real Hopper policy


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

## More bugs found and fixed (session 2)
5. **Test asserted a claim the tool never made.** Cross-policy ST comparison is
   meaningless — Sobol indices are normalized variance shares, so a policy broken
   everywhere shows *lower* ST than a robust one. Tests rewritten to compare
   score and breaking points (D16).
6. **Analysis cap diverged from the ground-truth cap.** The validation analysed
   300-step episodes while the manifest recorded 1000-step returns; these
   policies differ mainly in how long they survive, so the truncation hid the
   effect. Cap now recorded in the manifest and read back by the test.
7. **Report charted zeros as a ranking.** Degenerate and single-parameter
   analyses now explain why the numbers carry no information (D17).
8. **Wrong provenance** — `--evaluate-only` rewrote the manifest with argparse's
   default step count instead of the real training budget.

## Blocked
- None. `BLOCKED.md` was never needed.

## Next (see HANDOFF.md for detail)
1. Close the loop: retrain over the suggested DR config, assert the score improves.
2. Handle the "broken everywhere" regime (score ~0, no usable ranking) via
   adaptive range narrowing.
3. Per-parameter episode budgets / early stopping to make MuJoCo sweeps cheaper.

## Milestones / commits
- `1b0b976` test: spec + failing acceptance suite (RED)
- `a691d3d` feat: core implementation (WIP)
- `ee18715` feat: sweep, Sobol, score, analysis (acceptance 1–5 green)
- `868e9b3` feat: MuJoCo envs, HTML report, CLI (full suite green)
- (next) fix: parallel for loaded policies, DR range scale, docs
