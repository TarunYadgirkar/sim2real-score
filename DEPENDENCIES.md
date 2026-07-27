# DEPENDENCIES

Every external thing pulled in, what it is, why. Versions are what this venv
actually resolved.

## Core (required)
- **numpy** 1.26.4 — arrays, env dynamics, `SeedSequence` seeding. See pin note.
- **scipy** 1.17.1 — pulled by SALib; numerical helpers.
- **PyYAML** 6.0.3 — read/write randomization + DR configs.
- **SALib** 1.5.2 — Saltelli sampling + Sobol `S1`/`ST`/`S2`. The principled,
  interaction-aware ranking the spec asks for. Must be seeded explicitly
  (DECISIONS D9).
- **matplotlib** 3.11.1 — report plots, rendered to inline base64 PNG (Agg).
- **Jinja2** 3.1.6 — HTML report template.
- **joblib** 1.5.3 — parallel rollouts across CPU cores (loky backend).
- **pytest** 9.1.1 — test runner; the acceptance suite is ground truth.
- **pandas** 3.0.5 — transitive via SALib, not used directly.

## Optional backends (policy loaders; tests skip cleanly if absent)
- **torch** 2.2.2 — TorchScript `.pt` loading, ONNX export in tests, SB3 backend.
- **stable-baselines3** 2.4.1 — SB3 checkpoint loading.
- **onnx** 1.22.0 / **onnxruntime** 1.23.2 — ONNX policy loading.

## Optional sim (MuJoCo fixtures + smoke test)
- **gymnasium** 1.0.0 / **mujoco** 3.10.0 — Hopper/Walker2d/Reacher. Not needed
  for the core friction-vs-latency acceptance tests, which use the built-in
  `linear` env.

## Notes
- **numpy pinned `<2`.** torch 2.2.2 is the last PyTorch build for Intel macOS
  and raises "Numpy is not available" against numpy 2.x. The core tool runs fine
  on numpy 2.x; only the torch/SB3 extras force the pin. (DECISIONS D12.)
- Disk was ~3.7 GiB free at project start, ~10 GiB after the restart, which is
  what made installing torch + MuJoCo viable. The core suite deliberately has no
  torch/MuJoCo dependency so it runs regardless.
- No external repos cloned, no model weights pulled, no Claude Code
  skills/plugins installed beyond what the session already had.
