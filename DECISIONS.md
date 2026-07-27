# DECISIONS

Running log of non-obvious choices made autonomously. Newest first.

## D12 — numpy pinned <2 (environment constraint, not preference)
torch 2.2.2 is the last PyTorch build for Intel macOS and raises
"Numpy is not available" against numpy 2.x. Since the torch/SB3 extras are needed
to exercise acceptance test 6, the venv runs numpy 1.26.4. The *core* tool works
on numpy 2.x; only the torch-dependent extras force the pin. Noted in README so
it doesn't read as an arbitrary constraint.

## D11 — DR range expansion is clamped to the breaking point's scale
Expanding a suggested range linearly by the nominal-to-breaking-point gap can
overshoot through zero: friction breaking at 0.38 with a gap of 0.62 produced a
suggested low of -0.55, clamped to the 0.001 floor — i.e. "train over friction
∈ [0.001, 2.0]", physically meaningless. Suggested ranges are now floored at
`MIN_FRACTION_OF_BP × breaking_point` (and capped at `MAX_FACTOR_OF_BP ×` on the
high side), keeping the range on the same scale as the value it broke at.
Caught by a test written for the defect (`tests/test_dr_config.py`).

## D10 — Loaded policies pickle by load-spec, not by handle
TorchScript modules and ONNX InferenceSessions are unpicklable, so joblib could
not ship them to workers: every analysis of a policy *file* silently fell back to
serial (measured 3m09s at 99% CPU). Loader classes now implement
`__getstate__`/`__setstate__` that carry `(path, obs_dim, action_dim)` and rebuild
the handle in the worker — same run drops to 1m16s at 487% CPU, identical output.
A guard test asserts the parallel path actually engages, because a silent
fallback would make the parallel==serial acceptance test pass vacuously.

## D9 — SALib sampling must be explicitly seeded
SALib scrambles the Sobol' sequence with a fresh random seed unless one is
passed, so identical runs produced different indices and different scores
(observed ST for friction ranging 0.46–1.10 across three runs). Both
`sobol.sample` and `sobol.analyze` now receive the space's seed. Found by the
determinism acceptance test — exactly what it exists for.

## D8 — Fixture fragility must come from *distinct mechanisms*
First attempt made the friction-overfit policy a low-derivative-gain controller
that leans on environment friction for damping. That is self-defeating: such a
controller is intrinsically low-damping, so friction-fragility and
latency-fragility move together — measurement showed it was actually
latency-fragile and friction-*robust*, the opposite of intent. The two axes only
decouple with two independent mechanisms:
- friction-overfit: `u = -Kp·pos + c·vel`, where `c` cancels friction at nominal.
  Net damping is `(b - c)`, so the loop destabilizes when friction drops, and
  because loop gain stays low the delay margin remains large (latency-robust).
- latency-fragile: `u = -Kp·pos - Kd·vel` with strong `Kd`. Its own damping
  swamps friction variation (friction-robust), but high derivative gain on a
  delayed signal loses phase margin (latency-fragile).
The env also gained a spring term so the *uncontrolled* system is stable — that
is what makes a constant-zero policy genuinely good (acceptance 3) rather than
merely inert. Regimes were found by empirical search over the constant/gain
space, not hand-derived, and every constraint is re-checked by the
`test_ground_truth_*` tests.

Mass range in the fixture space was narrowed to [0.8, 1.3] for a real reason: at
low mass `DT·Kd/m` exceeds 1 and the explicit integrator itself destabilizes, so
mass confounded the latency axis (it outranked latency, correctly, on the wide
range). Narrowing isolates each policy's intended fragility. The *shipped* env
default keeps the wide mass range — hiding real failure modes from users would be
worse than a confounded fixture.

## D7 — Score = mean success rate over sampled space
`robustness_score = 100 * mean(success_rate)` across the Sobol-sampled
randomization space. Simple, monotone, interpretable, and directly satisfies the
trivial-robust-high / random-low acceptance test. Alternatives (margin-weighted,
worst-case) deferred; can be layered later without changing the interface.

## D6 — Sobol on failure indicator, rank by total-order ST
Analyse variance of `1 - success_rate`. Rank params by **total-order** ST so
interactions count (the whole point). S2 second-order matrix = interaction
matrix. Using SALib (well-tested Saltelli sampler + Sobol analyzer) rather than
hand-rolling. Failure indicator (not raw return) keeps the output bounded and
the "what breaks it" framing crisp.

## D5 — Determinism via per-sample seed derivation
Each rollout seed = `hash(base_seed, sample_index)` (stable, not Python's salted
hash — use a fixed mixing function). Rollout RNG (obs noise, dropout, env reset)
depends only on that seed. Guarantees serial == parallel and full
reproducibility, independent of worker scheduling. This is the linchpin for
acceptance tests 4 and 5.

## D4 — Ground truth = synthetic linear env + hardcoded policies
The friction-vs-latency discrimination tests (the core acceptance requirement)
use a pure-numpy controllable **linear** env (regulated damped point mass) and
**hardcoded linear controllers**, not trained neural nets on MuJoCo. Rationale
(contract: "pick the option easier to test"):
- Training real RL policies with a *known, provable* sensitivity profile is
  research-grade and non-deterministic in CI.
- A linear system + linear feedback has analytic stability behavior: a low-gain
  controller that leans on environment friction for damping is *provably*
  friction-fragile / latency-robust; a high-gain controller is *provably*
  friction-robust / latency-fragile (gain + delay -> instability).
- Each fixture's ground-truth property is confirmed by direct measurement (axis
  sweep) so the acceptance test validates the *tool*, not a circular fixture.
MuJoCo (Hopper/Walker2d/Reacher) is supported as a first-class env with default
DR configs + a smoke test; it is not used to manufacture ground-truth
sensitivity because we cannot cheaply guarantee a trained policy's sensitivity.

## D3 — Policy sensitivity fixtures are in-memory; loaders tested separately
Acceptance tests 1–5 use in-memory `LinearPolicy` objects (fast, no torch).
Loader support (SB3 / TorchScript `.pt` / ONNX) is verified by independent
round-trip tests (save a linear policy, load via each backend, assert matching
`predict`). Decouples the hard discrimination logic from heavy optional deps and
keeps the core suite runnable under tight disk.

## D2 — Raw `.pt` interface = TorchScript module
The "documented interface" for raw PyTorch is a **TorchScript** module
(`torch.jit.save`) taking float32 `[N, obs_dim]` and returning `[N, action_dim]`,
deterministic. Self-describing, no side-car arch file, loadable without the
original class definition. Documented in SPEC 4.1.

## D1 — Repo location
Created at `~/TarunsCode/sim2real-score/` (flat project dir), matching the
observed layout of `~/TarunsCode` (existing projects are flat folders, not the
repos/assets split). `uv`-managed venv, `src/` layout.

## D0 — Autonomy / skills
Contract forbids questions; SPEC is fully specified, so the brainstorming skill
(which asks clarifying questions) is skipped per user-instruction precedence.
Followed TDD skill: acceptance suite written first, watched fail, implement to
green.
