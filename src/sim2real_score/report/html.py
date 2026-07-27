"""Self-contained HTML robustness report."""
from __future__ import annotations

import os

import yaml
from jinja2 import Template

from ..analysis import suggest_dr_config
from .plots import grid_curves, interaction_heatmap, sensitivity_bar

TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sim2real-score — {{ env }}</title>
<style>
 :root { color-scheme: light dark; }
 body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        margin: 0 auto; max-width: 62rem; padding: 2rem 1.25rem 4rem; line-height: 1.55;
        color: #1c2126; background: #fff; }
 h1 { margin: 0 0 .25rem; font-size: 1.65rem; }
 h2 { margin-top: 2.5rem; font-size: 1.15rem; border-bottom: 1px solid #dfe4ea; padding-bottom: .35rem; }
 .sub { color: #66717c; margin-top: 0; }
 .score { display: flex; align-items: baseline; gap: .75rem; margin: 1.5rem 0; }
 .score b { font-size: 3rem; line-height: 1; color: {{ score_color }}; }
 .meter { height: .6rem; background: #e8ecf0; border-radius: 999px; overflow: hidden; flex: 1; }
 .meter > div { height: 100%; width: {{ '%.1f' % score }}%; background: {{ score_color }}; }
 table { border-collapse: collapse; width: 100%; font-size: .92rem; }
 th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #e8ecf0; }
 th { background: #f6f8fa; font-weight: 600; }
 td.num { text-align: right; font-variant-numeric: tabular-nums; }
 .pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .78rem; }
 .broken { background: #fbe4df; color: #99351f; }
 .safe { background: #e2f0e5; color: #245c33; }
 img { max-width: 100%; height: auto; display: block; margin: 1rem 0; }
 pre { background: #f6f8fa; padding: .9rem 1rem; border-radius: 6px; overflow-x: auto; font-size: .85rem; }
 .note { color: #66717c; font-size: .87rem; }
 @media (prefers-color-scheme: dark) {
   body { background: #14181c; color: #e6eaee; }
   th { background: #1e242b; } th, td { border-color: #2a323a; }
   .meter { background: #2a323a; } pre { background: #1e242b; }
   h2 { border-color: #2a323a; } .sub, .note { color: #93a1ad; }
 }
</style></head><body>

<h1>Robustness under domain shift — {{ env }}</h1>
<p class="sub">{{ n_active }} randomized parameter(s) · {{ episodes }} episodes ×
{{ max_steps }} steps per point · seed {{ seed }}</p>

<div class="score">
  <b>{{ '%.1f' % score }}</b><span class="sub">/ 100</span>
  <div class="meter"><div></div></div>
</div>
<p class="note">Share of the sampled randomization space in which the policy still
meets its success criterion. {{ failure_pct }}% of coarse-grid points failed.</p>

<h2>Parameter sensitivity</h2>
{% if degenerate %}
<p class="note"><strong>No variance to attribute.</strong> The policy met its
success criterion identically across every sampled point, so the Sobol indices
are all zero — that is an absence of measurable variation, not a finding that the
parameters do not matter. Widen the swept ranges or tighten the failure threshold
to get a ranking.</p>
{% elif sens_img %}
{% if single_param %}<p class="note"><strong>Only one parameter is active</strong>,
so it explains all of the variance by construction — the index near 1 below says
nothing about how fragile this policy is. Compare the score and the breaking
points instead, or activate more parameters to get a meaningful ranking.</p>
{% endif %}
<img src="{{ sens_img }}" alt="Sobol sensitivity indices">
<table><thead><tr><th>#</th><th>Parameter</th><th class="num">ST (total)</th>
<th class="num">S1 (first)</th></tr></thead><tbody>
{% for row in ranking_rows %}<tr><td>{{ loop.index }}</td><td>{{ row.name }}</td>
<td class="num">{{ '%.3f' % row.st }}</td><td class="num">{{ '%.3f' % row.s1 }}</td></tr>
{% endfor %}</tbody></table>
<p class="note">Ranked by total-order Sobol index, which counts every interaction a
parameter participates in. A large gap between ST and S1 means the parameter
matters mostly <em>in combination</em> with others.</p>
{% else %}<p class="note">No active parameters were swept.</p>{% endif %}

<h2>Breaking points</h2>
<table><thead><tr><th>Parameter</th><th class="num">Swept range</th>
<th class="num">Nominal</th><th class="num">Breaks below</th>
<th class="num">Breaks above</th><th>Status</th></tr></thead><tbody>
{% for row in bp_rows %}<tr>
 <td>{{ row.name }}</td>
 <td class="num">{{ row.range }}</td><td class="num">{{ row.nominal }}</td>
 <td class="num">{{ row.low }}</td><td class="num">{{ row.high }}</td>
 <td><span class="pill {{ 'broken' if row.broken else 'safe' }}">
   {{ 'breaks in range' if row.broken else 'holds' }}</span></td>
</tr>{% endfor %}</tbody></table>
{% if fails_at_nominal %}<p class="note"><strong>Note:</strong> the policy already
fails at nominal parameters, so breaking points are reported at nominal.</p>{% endif %}

{% if grid_img %}<h2>Where it degrades</h2><img src="{{ grid_img }}" alt="Marginal success rate">{% endif %}
{% if heat_img %}<h2>Interactions</h2><img src="{{ heat_img }}" alt="Interaction matrix">{% endif %}

<h2>Suggested domain randomization</h2>
<p class="note">Train over these ranges next. Where a breaking point was located the
range is pushed past it, further for parameters with a high total-order index.</p>
<pre>{{ dr_yaml }}</pre>

<p class="note">Generated by sim2real-score{% if truncated %} · coarse grid
subsampled to {{ n_evaluated }} of {{ n_full }} factorial points{% endif %}</p>
</body></html>""")


def _fmt(v, spec=None):
    if v is None:
        return "—"
    if spec is not None and spec.is_int:
        return f"{int(round(v))}"
    return f"{v:.3g}"


def build_report(result, out_dir: str, filename: str = "report.html") -> str:
    os.makedirs(out_dir, exist_ok=True)
    sens = result.sensitivity
    space = result.space

    ranking_rows = [
        {"name": n, "st": float(sens.ST[sens.names.index(n)]),
         "s1": float(sens.S1[sens.names.index(n)])}
        for n in sens.ranking()
    ]

    bp_rows = []
    for name in (sens.ranking() or sorted(result.breaking_points)):
        spec = space.param(name)
        bp = result.breaking_points.get(name)
        low = getattr(bp, "low", None)
        high = getattr(bp, "high", None)
        bp_rows.append({
            "name": name,
            "range": f"{_fmt(spec.low, spec)} – {_fmt(spec.high, spec)}",
            "nominal": _fmt(spec.nominal, spec),
            "low": _fmt(low, spec), "high": _fmt(high, spec),
            "broken": low is not None or high is not None,
        })

    score = float(result.score)
    html = TEMPLATE.render(
        env=result.env_id,
        score=score,
        score_color="#2e7d3a" if score >= 75 else ("#b8860b" if score >= 45 else "#c2452d"),
        n_active=len(space.active_params),
        episodes=space.rollout["episodes"], max_steps=space.rollout["max_steps"],
        seed=space.seed,
        failure_pct=f"{100.0 * result.grid.failure_fraction:.0f}",
        degenerate=bool(sens.degenerate) and len(space.active_params) > 0,
        single_param=len(space.active_params) == 1,
        sens_img=sensitivity_bar(result),
        heat_img=interaction_heatmap(result),
        grid_img=grid_curves(result),
        ranking_rows=ranking_rows,
        bp_rows=bp_rows,
        fails_at_nominal=any(getattr(b, "fails_at_nominal", False)
                             for b in result.breaking_points.values()),
        dr_yaml=yaml.safe_dump(suggest_dr_config(result), sort_keys=True,
                               default_flow_style=False),
        truncated=result.grid.truncated,
        n_evaluated=int(len(result.grid.success_rate)),
        n_full=result.grid.n_full,
    )
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write(html)
    return path
