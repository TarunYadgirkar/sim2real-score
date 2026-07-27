"""Matplotlib figures rendered to inline base64 PNG so the report stays a single
self-contained file. Headless (Agg) by construction."""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DPI = 110
ACCENT = "#2f6f9f"
WARN = "#c2452d"


def _to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def sensitivity_bar(result) -> str:
    sens = result.sensitivity
    names = sens.ranking()
    if not names:
        return ""
    idx = [sens.names.index(n) for n in names]
    st = [float(sens.ST[i]) for i in idx]
    s1 = [max(0.0, float(sens.S1[i])) for i in idx]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.4))
    ax.barh(y - 0.2, st, height=0.38, color=ACCENT, label="total order (ST)")
    ax.barh(y + 0.2, s1, height=0.38, color="#9ec6e0", label="first order (S1)")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index (variance share of policy failure)")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    return _to_data_uri(fig)


def interaction_heatmap(result) -> str:
    M = result.interaction_matrix
    names = result.sensitivity.names
    if M.size == 0:
        return ""
    fig, ax = plt.subplots(figsize=(0.75 * len(names) + 2.6, 0.75 * len(names) + 2.2))
    im = ax.imshow(M, cmap="magma_r", vmin=0.0)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 * max(M.max(), 1e-9) else "black")
    ax.set_title("Interaction matrix\n(diagonal = ST, off-diagonal = |S2|)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _to_data_uri(fig)


def grid_curves(result) -> str:
    per = result.grid.per_param
    names = [n for n in result.sensitivity.ranking() if n in per]
    if not names:
        return ""
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.7 * rows), squeeze=False)
    for k, name in enumerate(names):
        ax = axes[k // cols][k % cols]
        vals = per[name]["values"]
        sr = per[name]["success_rate"]
        ax.plot(vals, sr, marker="o", color=ACCENT)
        ax.axhline(0.5, ls="--", lw=1, color=WARN)
        bp = result.breaking_points.get(name)
        for edge in (getattr(bp, "low", None), getattr(bp, "high", None)):
            if edge is not None:
                ax.axvline(edge, ls=":", lw=1.4, color=WARN)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("value")
        ax.set_ylabel("success rate")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    for k in range(len(names), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Marginal success rate across the swept range "
                 "(dashed = failure threshold, dotted = located breaking point)",
                 fontsize=10)
    fig.tight_layout()
    return _to_data_uri(fig)
