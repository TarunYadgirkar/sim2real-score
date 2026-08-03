# Demo output

Both `.html` files are real, self-contained reports written by the tool. They are
hosted at **https://sim2real-score-report.vercel.app** — `index.html` in this
directory is the landing page linking both. `report.png` and `report_hopper.png`
are screenshots of the top of each, so the README renders on GitHub, which serves
committed HTML as raw source.

`report.html` is a real report produced by:

```bash
python examples/export_example_policies.py --out /tmp/policies
sim2real-score run --policy /tmp/policies/friction_overfit.pt --env linear --out /tmp/out
```

A friction-overfit policy on the built-in `linear` env: score 87.5/100, friction
breaking point located at 0.382, suggested friction training range widened down
to 0.153.

To redeploy this directory after regenerating either report:

```bash
vercel deploy --prod --yes
```

`report_hopper.html` is the same tool run against the checked-in nominal-trained
PPO policy on Hopper-v5, loaded with its VecNormalize statistics:

```bash
sim2real-score run --policy experiments/policies/nominal.zip --policy-kind sb3 \
  --vecnormalize experiments/policies/nominal_vecnormalize.pkl \
  --env Hopper-v5 --out out --sobol-base 8
```

Score 21.4/100, with breaking points at 2 steps of actuator lag and friction
1.37× nominal. At `--sobol-base 8` the Sobol estimator is not converged — one `S1`
comes back above 1 — so read the ranking tail there as indicative only.
