# DECISIONS

Running log of non-obvious choices made autonomously. Newest first.

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
