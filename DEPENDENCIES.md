# DEPENDENCIES

Every external thing pulled in, what it is, why. Updated as installed.

## Core (required)
- **numpy** — arrays, linear env dynamics, RNG. Core.
- **scipy** — stats/interp helpers for sweep + report. Core.
- **PyYAML** — read/write randomization + DR configs. Core.
- **SALib** — Saltelli sampling + Sobol (S1/ST/S2) sensitivity. The principled
  interaction-aware ranking. Core.
- **matplotlib** — plots for the HTML report (rendered to inline base64 PNG). Core.
- **Jinja2** — HTML report templating. Core.
- **joblib** — parallel rollouts across CPU cores (loky backend). Core.
- **pytest** — test runner. Dev/core (acceptance suite is ground truth).

## Optional backends (policy loaders; degrade gracefully if absent)
- **torch** (CPU-only) — TorchScript `.pt` load + SB3 dependency. Optional:
  loader/round-trip tests skip if missing. CPU wheel chosen for disk/size.
- **stable-baselines3** — SB3 checkpoint load. Optional (pulls torch).
- **onnx / onnxruntime** — ONNX policy load + export in round-trip test. Optional.

## Optional sim (MuJoCo fixtures; smoke test only)
- **gymnasium[mujoco]** / **mujoco** — Hopper/Walker2d/Reacher fixtures + smoke
  test. Optional: MuJoCo tests skip if missing. Not required for the core
  friction-vs-latency acceptance tests (those use the built-in linear env).

## Notes
- Disk was ~3.7 GiB free at start (99% full). Heavy/optional deps installed last
  and kept CPU-only; core suite intentionally has no torch/mujoco dependency so
  it runs even if the big installs can't fit. See BLOCKED.md if any install is
  abandoned.
