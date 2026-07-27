# PROGRESS

Updated every ~30 min of work. Newest at top.

## Now
- [x] Env recon, repo init, `src/` tree
- [x] SPEC.md (authoritative)
- [x] DECISIONS/DEPENDENCIES/PROGRESS
- [x] pyproject + uv venv + core deps
- [x] Acceptance suite written + watched fail (RED, committed)
- [x] Core impl: policies (base+loaders), linear env, registry, randomization
      (space+apply/DomainShiftWrapper), rollout runner (determinism seeding)
- [~] Fixture tuning IN PROGRESS — see note below
- [ ] sweep (grid+bisect), sensitivity (Sobol), score, analysis orchestrator
- [ ] report (HTML+plots), DR config, CLI
- [ ] verify loaders (.pt/onnx/sb3), mujoco smoke
- [ ] README + HANDOFF

## OPEN THREAD (resume here)
Ground-truth fixtures not yet behaving. Dev check showed the intended
`friction_overfit` policy came out latency-fragile + friction-robust (opposite).
Root cause: a controller leaning on env friction for damping is intrinsically
low-damping => friction-sensitivity and latency-sensitivity are physically
COUPLED. Fix in flight: decouple by making friction-fragility a low-bandwidth
(low-Kp, Kd=0, no intrinsic damping floor) underdamped-PERFORMANCE failure at low
friction (latency-robust because low bandwidth), keep latency_fragile as high
Kp+Kd. Running an empirical grid search over (dt, DAMPING0, Kp_o, Kp_l, Kd_l,
threshold) to find a regime satisfying all fixture inequalities:
  scratchpad/search.py  (prints satisfying candidates)
Once found: bake constants into src/.../envs/linear.py (DT, DAMPING0) and gains
into tests/fixtures/policies.py, then re-run scratchpad/tune.py to confirm, then
build sweep/sensitivity.

## Next
1. Finish fixture tuning (search.py) -> bake constants.
2. sweep + Sobol + score + run_analysis (unblocks most acceptance tests).
3. report + DR config + CLI.
4. loaders/mujoco verification.
5. README + HANDOFF.

## Blocked
- none hard (fixture tuning is active, not blocked). Disk ~2.3 GiB free — torch/
  mujoco installs deferred; core suite is torch-free by design.

## Milestones / commits
- 1b0b976 test: spec + failing acceptance suite (RED)
- (this commit) WIP core implementation

## Done
- Recon: py3.11.8, uv 0.11.17, git 2.41, 8 cores, tight disk.
- Repo at ~/TarunsCode/sim2real-score, git init.

## Next
1. Package skeleton (protocols + linear env + policies).
2. Rollout + determinism + parallel.
3. Sweep (grid + bisection) + Sobol sensitivity + score.
4. Report (HTML + plots) + DR config + CLI.
5. Loaders (.pt/onnx/sb3) round-trip.
6. MuJoCo defaults + smoke.
7. README + HANDOFF.

## Blocked
- none

## Milestones / commits
- (pending first commit)
