# Demo output

`report.html` is a real report produced by:

```bash
python examples/export_example_policies.py --out /tmp/policies
sim2real-score run --policy /tmp/policies/friction_overfit.pt --env linear --out /tmp/out
```

A friction-overfit policy on the built-in `linear` env: score 87.5/100, friction
breaking point located at 0.382, suggested friction training range widened down
to 0.153.

To deploy it as a static page:

```bash
cp report.html index.html && vercel deploy --prod --yes
```
