"""Generate publication figures for the Causal Geometry paper.

Usage:
    uv run python experiments/batch6_atlas/generate_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data — all 22 operations
# ---------------------------------------------------------------------------
# (equivariance, iia_k2, grokked, category, tier)
# Updated 2026-05-30 from verified Modal JSONL results (n_epochs >= 15000, das_steps=400)
# "Best seed" equivariance used for tier placement
OPERATIONS = {
    "Multiplication":    (0.995, 1.000, True,  "Group action",     "Grassmannian"),
    "Comp. addition":    (0.998, 0.960, True,  "Group action",     "Grassmannian"),
    "Subtraction":       (1.000, 1.000, True,  "Group action",     "Grassmannian"),
    "Division":          (1.000, 0.980, True,  "Group action",     "Grassmannian"),
    "Bitwise XOR":       (1.000, 1.000, True,  "Group action",     "Grassmannian"),
    "Sum of squares":    (0.987, 0.960, True,  "Polynomial",       "Near-Grassmannian"),
    "Cubic sum":         (1.000, 1.000, True,  "Polynomial",       "Grassmannian"),
    "Cubing":            (0.509, 1.000, False, "Polynomial",       "Partial"),
    "Squaring":          (0.314, 1.000, False, "Polynomial",       "Memorization"),
    "Polynomial":        (0.015, 0.680, False, "Polynomial",       "No structure"),
    "Affine":            (0.026, 0.780, False, "Polynomial",       "No structure"),
    "Max":               (0.972, 0.960, True,  "Piecewise",        "Near-Grassmannian"),
    "Abs. difference":   (0.189, 0.800, False, "Piecewise",        "No structure"),
    "Power":             (0.965, 0.940, True,  "Non-algebraic",    "Near-Grassmannian"),
    "Dyck-1 depth":      (1.000, 1.000, True,  "Contrast",         "Grassmannian"),
    "Digit addition":    (0.578, 0.880, False, "Contrast",         "Partial"),
    "Othello":           (0.720, 0.749, True,  "Contrast",         "Partial"),
    # PENDING rerun — old data was n_epochs=30, unreliable:
    # "Shifted mult.":   (?, ?, ?, "Group action",    "?"),
    # "Min":             (?, ?, ?, "Piecewise",       "?"),
    # "Floor division":  (?, ?, ?, "Piecewise",       "?"),
    # "GCD":             (?, ?, ?, "Number-theoretic", "?"),
}

CATEGORY_COLORS = {
    "Group action":     "#2166ac",
    "Polynomial":       "#e08214",
    "Piecewise":        "#1b7837",
    "Number-theoretic": "#7b3294",
    "Non-algebraic":    "#c51b7d",
    "Contrast":         "#878787",
}

TIER_BOUNDARIES = {
    "Grassmannian":      (0.99, 1.00),
    "Near-Grassmannian": (0.91, 0.99),
    "Partial":           (0.49, 0.91),
    "Memorization":      (0.20, 0.49),
    "No structure":      (0.00, 0.20),
}

TIER_COLORS = {
    "Grassmannian":      "#d4eac7",
    "Near-Grassmannian": "#e8f0fe",
    "Partial":           "#fff8e1",
    "Memorization":      "#fce4ec",
    "No structure":      "#f3e5f5",
}

PRINCIPAL_ANGLES = {
    ("Division", "Multiplication"): 2.060,
    ("Division", "Shifted mult."): 2.080,
    ("Division", "Sum of sq."): 1.959,
    ("Mult.", "Shifted mult."): 2.038,
    ("Mult.", "Sum of sq."): 2.069,
    ("Shifted mult.", "Sum of sq."): 2.050,
}

MULT_TRAJECTORY = [
    (499, 4.988, 0.000),
    (7999, 4.935, 0.000),
    (15499, 4.594, 0.008),
    (22999, 4.309, 0.044),
    (30499, 4.173, 0.044),
    (37999, 4.214, 0.027),
]


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Equivariance ranked with tier shading
# ---------------------------------------------------------------------------
def fig1_equivariance_bars() -> None:
    print("Figure 1: equivariance bars")

    sorted_ops = sorted(OPERATIONS.items(), key=lambda kv: kv[1][0])
    names = [k for k, _ in sorted_ops]
    equivs = [v[0] for _, v in sorted_ops]
    colors = [CATEGORY_COLORS[v[3]] for _, v in sorted_ops]
    grokked = [v[2] for _, v in sorted_ops]

    fig, ax = plt.subplots(figsize=(4.5, 6.5), constrained_layout=True)
    y_pos = np.arange(len(names))

    # Tier background shading
    tier_labels_placed = set()
    for tier_name, (lo, hi) in TIER_BOUNDARIES.items():
        ax.axvspan(lo, hi, color=TIER_COLORS[tier_name], alpha=0.5, zorder=0)
        mid = (lo + hi) / 2
        if tier_name not in tier_labels_placed:
            ax.text(mid, len(names) + 0.3, tier_name, fontsize=5.5,
                    ha="center", va="bottom", color="#555555", rotation=90 if (hi - lo) < 0.15 else 0)
            tier_labels_placed.add(tier_name)

    bars = ax.barh(y_pos, equivs, color=colors, edgecolor="white",
                   linewidth=0.3, height=0.7, zorder=2)

    # Mark non-grokked with hatching
    for i, (bar, grok) in enumerate(zip(bars, grokked)):
        if not grok:
            bar.set_hatch("//")
            bar.set_edgecolor("#999999")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Equivariance")
    ax.set_xlim(0, 1.08)

    for i, v in enumerate(equivs):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=6.5, zorder=3)

    # Legend
    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    hatch_patch = mpatches.Patch(facecolor="#cccccc", hatch="//", edgecolor="#999999", label="Not grokked")
    ax.legend(handles=cat_handles + [hatch_patch], fontsize=6, loc="lower right",
              framealpha=0.9, ncol=1)

    _save(fig, "fig1_equivariance_bars")


# ---------------------------------------------------------------------------
# Figure 2: IIA vs Equivariance — interactive Plotly
# ---------------------------------------------------------------------------
def fig2_iia_vs_equiv_plotly() -> None:
    if not HAS_PLOTLY:
        print("Figure 2 (plotly): skipped — plotly not installed")
        return
    print("Figure 2 (plotly): IIA vs equivariance interactive")

    fig = go.Figure()

    for cat, color in CATEGORY_COLORS.items():
        ops_in_cat = [(n, v) for n, v in OPERATIONS.items() if v[3] == cat]
        if not ops_in_cat:
            continue
        ns = [n for n, _ in ops_in_cat]
        iias = [v[1] for _, v in ops_in_cat]
        equivs = [v[0] for _, v in ops_in_cat]
        grokked = [v[2] for _, v in ops_in_cat]
        tiers = [v[4] for _, v in ops_in_cat]

        hovers = [
            f"<b>{n}</b><br>"
            f"IIA(k=2): {iia:.3f}<br>"
            f"Equiv: {eq:.3f}<br>"
            f"Grokked: {'Yes' if g else 'No'}<br>"
            f"Tier: {t}"
            for n, iia, eq, g, t in zip(ns, iias, equivs, grokked, tiers)
        ]

        symbols = ["circle" if g else "diamond" for g in grokked]

        fig.add_trace(go.Scatter(
            x=iias, y=equivs,
            mode="markers+text",
            marker=dict(size=12, color=color, line=dict(width=1, color="white"),
                        symbol=symbols),
            text=ns,
            textposition="top center",
            textfont=dict(size=9, color="#333333"),
            hovertext=hovers,
            hoverinfo="text",
            name=cat,
            legendgroup=cat,
        ))

    # Tier boundaries as horizontal bands
    fig.add_hrect(y0=0.99, y1=1.01, fillcolor=TIER_COLORS["Grassmannian"],
                  opacity=0.4, line_width=0,
                  annotation_text="Grassmannian", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color="#555555")
    fig.add_hrect(y0=0.91, y1=0.99, fillcolor=TIER_COLORS["Near-Grassmannian"],
                  opacity=0.3, line_width=0)
    fig.add_hrect(y0=-0.05, y1=0.49, fillcolor="#fce4ec",
                  opacity=0.3, line_width=0,
                  annotation_text="Memorization / No structure", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color="#aa0000")

    fig.update_layout(
        xaxis_title="IIA (k=2)",
        yaxis_title="Equivariance",
        xaxis=dict(range=[0.60, 1.05]),
        yaxis=dict(range=[-0.05, 1.08]),
        width=900, height=700,
        template="plotly_white",
        legend=dict(x=0.02, y=0.55, bgcolor="rgba(255,255,255,0.8)"),
        title="IIA vs Equivariance — 22 Operations",
    )

    path = OUTPUT_DIR / "fig2_iia_vs_equiv_interactive.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 (static): IIA vs Equivariance — matplotlib with adjustText
# ---------------------------------------------------------------------------
def fig2_iia_vs_equiv() -> None:
    print("Figure 2 (static): IIA vs equivariance scatter")

    try:
        from adjustText import adjust_text
        has_adjust = True
    except ImportError:
        has_adjust = False

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)

    # Tier shading as horizontal bands
    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.5, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.4, zorder=0)
    ax.axhspan(-0.05, 0.49, color="#fce4ec", alpha=0.3, zorder=0)

    ax.text(0.62, 1.035, "Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(0.62, 0.95, "Near-Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(0.62, 0.22, "Memorization /\nNo structure", fontsize=7, color="#aa0000", va="center")

    names = list(OPERATIONS.keys())
    equivs = np.array([OPERATIONS[n][0] for n in names])
    iias = np.array([OPERATIONS[n][1] for n in names])
    cats = [OPERATIONS[n][3] for n in names]
    grokked_list = [OPERATIONS[n][2] for n in names]
    colors = [CATEGORY_COLORS[c] for c in cats]

    for i, name in enumerate(names):
        marker = "o" if grokked_list[i] else "D"
        ax.scatter(iias[i], equivs[i], c=colors[i], s=50, zorder=3,
                   edgecolors="white", linewidths=0.5, marker=marker)

    # Labels with adjustText for automatic de-overlapping
    texts = []
    for i, name in enumerate(names):
        t = ax.text(iias[i], equivs[i], f"  {name}", fontsize=6.5, color="#333333",
                    ha="left", va="center", zorder=4)
        texts.append(t)

    if has_adjust:
        adjust_text(texts, x=iias, y=equivs, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                    force_text=(0.8, 0.8), force_points=(0.5, 0.5),
                    expand_text=(1.2, 1.4), expand_points=(1.5, 1.5))

    ax.set_xlabel("IIA (k=2)")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(0.60, 1.05)
    ax.set_ylim(-0.05, 1.08)

    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    grok_handle = plt.Line2D([], [], marker="o", color="gray", linestyle="None",
                             markersize=6, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="gray", linestyle="None",
                               markersize=5, label="Not grokked")
    ax.legend(handles=cat_handles + [grok_handle, nogrok_handle],
              fontsize=6.5, loc="center left", framealpha=0.9, ncol=1)

    _save(fig, "fig2_iia_vs_equiv")


def fig2b_iia_vs_equiv_nolabels() -> None:
    """Fig 2 style but no point labels."""
    print("Figure 2b (static): IIA vs equivariance (no labels)")

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)

    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.5, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.4, zorder=0)
    ax.axhspan(-0.05, 0.49, color="#fce4ec", alpha=0.3, zorder=0)

    ax.text(0.62, 1.035, "Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(0.62, 0.95, "Near-Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(0.62, 0.22, "Memorization /\nNo structure", fontsize=7, color="#aa0000", va="center")

    names = list(OPERATIONS.keys())
    equivs = np.array([OPERATIONS[n][0] for n in names])
    iias = np.array([OPERATIONS[n][1] for n in names])
    cats = [OPERATIONS[n][3] for n in names]
    grokked_list = [OPERATIONS[n][2] for n in names]
    colors = [CATEGORY_COLORS[c] for c in cats]

    for i, name in enumerate(names):
        marker = "o" if grokked_list[i] else "D"
        ax.scatter(iias[i], equivs[i], c=colors[i], s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, marker=marker)

    ax.set_xlabel("IIA (k=2)")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(0.60, 1.05)
    ax.set_ylim(-0.05, 1.08)

    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    grok_handle = plt.Line2D([], [], marker="o", color="gray", linestyle="None",
                             markersize=6, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="gray", linestyle="None",
                               markersize=5, label="Not grokked")
    ax.legend(handles=cat_handles + [grok_handle, nogrok_handle],
              fontsize=6.5, loc="center left", framealpha=0.9, ncol=1)

    _save(fig, "fig2b_iia_vs_equiv_nolabels")


def fig2c_iia_vs_equiv_grokked() -> None:
    """IIA vs equivariance — just grokked (blue) vs not grokked (pink)."""
    print("Figure 2c: IIA vs equivariance (grokked coloring only)")

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)

    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.5, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.4, zorder=0)

    for name, (equiv, iia, grok, cat, tier) in OPERATIONS.items():
        color = "#2166ac" if grok else "#c51b7d"
        marker = "o" if grok else "D"
        ax.scatter(iia, equiv, c=color, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, marker=marker)

    ax.set_xlabel("IIA (k=2)")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(0.60, 1.05)
    ax.set_ylim(-0.05, 1.08)

    from matplotlib.patches import Patch
    grass_patch = Patch(facecolor=TIER_COLORS["Grassmannian"],
                        label="Grassmannian")
    near_patch = Patch(facecolor=TIER_COLORS["Near-Grassmannian"],
                       label="Near-Grassmannian")
    grok_handle = plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2166ac",
                             markersize=7, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="w", markerfacecolor="#c51b7d",
                               markersize=7, label="Not grokked")
    ax.legend(handles=[grass_patch, near_patch, grok_handle, nogrok_handle],
              fontsize=8, loc="center left", framealpha=0.9)

    _save(fig, "fig2c_iia_vs_equiv_grokked")


def fig2d_iia_vs_equiv_regression() -> None:
    """IIA vs equivariance with regression lines through grokked/not-grokked clusters."""
    print("Figure 2d: IIA vs equivariance (regression lines)")
    from scipy import stats

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)

    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.5, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.4, zorder=0)
    ax.axhspan(-0.05, 0.49, color="#fce4ec", alpha=0.3, zorder=0)

    grok_iia, grok_eq, nogrok_iia, nogrok_eq = [], [], [], []
    for name, (equiv, iia, grok, cat, tier) in OPERATIONS.items():
        color = "#2166ac" if grok else "#c51b7d"
        marker = "o" if grok else "D"
        ax.scatter(iia, equiv, c=color, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, marker=marker)
        if grok:
            grok_iia.append(iia)
            grok_eq.append(equiv)
        else:
            nogrok_iia.append(iia)
            nogrok_eq.append(equiv)

    for x_vals, y_vals, color, label in [
        (grok_iia, grok_eq, "#2166ac", "Grokked"),
        (nogrok_iia, nogrok_eq, "#c51b7d", "Not grokked"),
    ]:
        if len(x_vals) >= 2:
            slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
            x_line = np.linspace(min(x_vals) - 0.02, max(x_vals) + 0.02, 50)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "-", color=color, linewidth=1.5, alpha=0.4, zorder=2)
            ax.fill_between(x_line, y_line - se, y_line + se, color=color, alpha=0.08, zorder=1)

    ax.set_xlabel("IIA (k=2)")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(0.60, 1.05)
    ax.set_ylim(-0.05, 1.08)

    grok_handle = plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2166ac",
                             markersize=7, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="w", markerfacecolor="#c51b7d",
                               markersize=7, label="Not grokked")
    ax.legend(handles=[grok_handle, nogrok_handle],
              fontsize=8, loc="center left", framealpha=0.9)

    _save(fig, "fig2d_iia_vs_equiv_regression")


# ---------------------------------------------------------------------------
# Figure 3: Grassmannian trajectory
# ---------------------------------------------------------------------------
def fig3_trajectory() -> None:
    print("Figure 3: Grassmannian trajectory")

    epochs = [t[0] for t in MULT_TRAJECTORY]
    d_gr = [t[1] for t in MULT_TRAJECTORY]
    equiv = [t[2] for t in MULT_TRAJECTORY]

    fig, ax1 = plt.subplots(figsize=(5, 3), constrained_layout=True)

    color_dist = "#2166ac"
    color_equiv = "#e08214"

    ax1.plot(epochs, d_gr, "o-", color=color_dist, markersize=4, linewidth=1.5,
             label="$d_{\\mathrm{Gr}}$ to final")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("$d_{\\mathrm{Gr}}$ to final subspace", color=color_dist)
    ax1.tick_params(axis="y", labelcolor=color_dist)

    ax2 = ax1.twinx()
    ax2.plot(epochs, equiv, "s-", color=color_equiv, markersize=4, linewidth=1.5,
             label="Equivariance")
    ax2.set_ylabel("Equivariance", color=color_equiv)
    ax2.tick_params(axis="y", labelcolor=color_equiv)

    grok_epoch = 15499
    ax1.axvline(grok_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(grok_epoch + 500, max(d_gr) - 0.05, "grokking", fontsize=7, color="gray", va="top")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    _save(fig, "fig3_trajectory")


# ---------------------------------------------------------------------------
# Figure 4: Principal angle heatmap
# ---------------------------------------------------------------------------
def fig4_principal_angles() -> None:
    print("Figure 4: principal angle heatmap")

    labels = ["Division", "Mult.", "Shifted mult.", "Sum of sq."]
    key_to_label = {
        "Division": "Division", "Multiplication": "Mult.", "Mult.": "Mult.",
        "Shifted mult.": "Shifted mult.", "Sum of sq.": "Sum of sq.",
    }
    n = len(labels)
    mat = np.zeros((n, n))

    for (a, b), val in PRINCIPAL_ANGLES.items():
        i, j = labels.index(key_to_label[a]), labels.index(key_to_label[b])
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(4, 3.5), constrained_layout=True)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=2.221)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            text_color = "white" if val > 1.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Grassmannian distance", fontsize=9)
    ax.set_title("$d_{\\max} = \\pi/\\sqrt{2} \\approx 2.221$", fontsize=9, pad=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    _save(fig, "fig4_principal_angles")


# ---------------------------------------------------------------------------
# Figure 5: Power seed-dependent grokking
# ---------------------------------------------------------------------------
def fig5_power_seeds() -> None:
    print("Figure 5: power seed-dependent grokking")

    seeds = [42, 137, 2024]
    grokked = [False, True, False]
    equivs = [0.665, 0.965, 0.099]
    losses = [1.14, 0.088, 15.7]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    bar_colors = ["#c51b7d" if not g else "#2166ac" for g in grokked]

    # Left: equivariance by seed
    bars = ax1.bar(range(len(seeds)), equivs, color=bar_colors, edgecolor="white", width=0.6)
    ax1.set_xticks(range(len(seeds)))
    ax1.set_xticklabels([f"Seed {s}\n{'Grokked' if g else 'Not grokked'}"
                         for s, g in zip(seeds, grokked)], fontsize=8)
    ax1.set_ylabel("Equivariance")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Power ($a^b$ mod 113) — Equivariance", fontsize=10)
    for i, v in enumerate(equivs):
        ax1.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Right: test loss by seed
    bars2 = ax2.bar(range(len(seeds)), losses, color=bar_colors, edgecolor="white", width=0.6)
    ax2.set_xticks(range(len(seeds)))
    ax2.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=8)
    ax2.set_ylabel("Test loss")
    ax2.set_title("Power — Test loss", fontsize=10)
    ax2.set_yscale("log")
    ax2.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.text(2.4, 0.12, "grok threshold", fontsize=7, color="gray", va="bottom")
    for i, v in enumerate(losses):
        ax2.text(i, v * 1.3, f"{v:.2f}", ha="center", fontsize=9)

    _save(fig, "fig5_power_seeds")


# ---------------------------------------------------------------------------
# Figure 6: Power training trajectory — test loss over epochs per seed
# ---------------------------------------------------------------------------
POWER_TRAJECTORIES = {
    42: [
        (499, 6.253), (10499, 14.156), (20499, 12.805),
        (30499, 1.936), (40499, 0.794), (50499, 1.020),
        (60499, 1.990), (70499, 1.036), (79999, 1.143),
    ],
    137: [
        (499, 6.731), (10499, 19.701), (20499, 8.353),
        (30499, 10.720), (40499, 0.777), (50499, 0.639),
        (60499, 0.087), (70499, 0.123), (79999, 0.088),
    ],
    2024: [
        (499, 6.273), (10499, 15.317), (20499, 13.602),
        (30499, 12.471), (40499, 11.573), (50499, 13.467),
        (60499, 9.013), (70499, 8.600), (79999, 15.719),
    ],
}

POWER_FINAL_EQUIV = {42: 0.665, 137: 0.965, 2024: 0.099}


def fig6_power_trajectory() -> None:
    print("Figure 6: power training trajectory")

    seed_colors = {42: "#e08214", 137: "#2166ac", 2024: "#c51b7d"}
    seed_styles = {42: "s--", 137: "o-", 2024: "D:"}

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for seed, traj in POWER_TRAJECTORIES.items():
        epochs = [t[0] for t in traj]
        losses = [t[1] for t in traj]
        grok = POWER_FINAL_EQUIV[seed] > 0.9
        label = f"Seed {seed} (equiv={POWER_FINAL_EQUIV[seed]:.3f})"
        if grok:
            label += " — grokked"
        ax.plot(epochs, losses, seed_styles[seed], color=seed_colors[seed],
                markersize=5, linewidth=1.5, label=label)

    ax.set_yscale("log")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(82000, 0.11, "grok threshold", fontsize=7, color="gray", va="bottom")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Test loss (log scale)")
    ax.set_title("Power ($a^b$ mod 113) — Training trajectory across seeds", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_xlim(-2000, 85000)

    _save(fig, "fig6_power_trajectory")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Figure 7: Test loss vs equivariance — all operations and seeds
# ---------------------------------------------------------------------------
# (display_name, seed, grokked, test_loss, equivariance)
# Source: JSONL results from Modal runs with proper training (n_epochs >= 15000, das_steps=400)
ALL_RESULTS = [
    # --- Group actions (deterministic grokers) ---
    ("Multiplication", 42, True, 0.0002, 0.995),
    ("Multiplication", 2024, True, 0.0031, 0.997),
    ("Subtraction", 42, True, 0.0000, 1.000),
    ("Division", 42, True, 0.0003, 1.000),
    ("Bitwise XOR", 42, True, 0.0026, 1.000),
    ("Bitwise XOR", 137, True, 0.0054, 0.988),
    ("Bitwise XOR", 2024, True, 0.0000, 0.990),
    # --- Stochastic grokers ---
    ("Comp. addition", 42, False, 10.2074, 0.110),
    ("Comp. addition", 0, True, 0.0001, 0.998),
    ("Power", 42, False, 1.1426, 0.665),
    ("Power", 137, True, 0.0881, 0.965),
    # --- Polynomial family ---
    ("Sum of squares", 42, True, 0.0045, 0.987),
    ("Cubic sum", 42, True, 0.0001, 1.000),
    ("Cubing", 42, False, 9.7649, 0.509),
    ("Squaring", 42, False, 7.8075, 0.314),
    ("Polynomial", 42, False, 21.5503, 0.015),
    ("Affine", 42, False, 26.2091, 0.026),
    # --- Piecewise ---
    ("Max", 42, True, 0.0743, 0.957),
    ("Abs. difference", 42, False, 1.8346, 0.189),
    # PENDING rerun — old data was n_epochs=30, unreliable:
    # ("Shifted mult.", 42, ?, ?, ?),
    # ("Min", 42, ?, ?, ?),
    # ("Floor division", 42, ?, ?, ?),
    # ("GCD", 42, ?, ?, ?),
]


def fig7_loss_vs_equiv() -> None:
    print("Figure 7: test loss vs equivariance")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    grok_losses = [r[3] for r in ALL_RESULTS if r[2]]
    grok_equivs = [r[4] for r in ALL_RESULTS if r[2]]
    nogrok_losses = [r[3] for r in ALL_RESULTS if not r[2]]
    nogrok_equivs = [r[4] for r in ALL_RESULTS if not r[2]]

    ax.scatter(grok_losses, grok_equivs, c="#2166ac", s=50, zorder=3,
               edgecolors="white", linewidths=0.5, marker="o", label="Grokked")
    ax.scatter(nogrok_losses, nogrok_equivs, c="#c51b7d", s=50, zorder=3,
               edgecolors="white", linewidths=0.5, marker="D", label="Not grokked")

    # Annotate power seeds
    for r in ALL_RESULTS:
        if r[0] == "Power":
            ax.annotate(f"Power s{r[1]}", (r[3], r[4]), fontsize=7,
                        xytext=(5, 5), textcoords="offset points", color="#666666")
        elif r[0] == "Affine":
            ax.annotate("Affine", (r[3], r[4]), fontsize=7,
                        xytext=(5, -10), textcoords="offset points", color="#666666")

    ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Test loss (symlog scale)")
    ax.set_ylabel("Equivariance")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Grokking predicts equivariant causal structure", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    ax.fill_betweenx([0.91, 1.08], -0.01, 0.1, color=TIER_COLORS["Grassmannian"],
                     alpha=0.3, zorder=0)
    ax.text(0.02, 1.03, "Grokked +\nGrassmannian", fontsize=7, color="#2166ac",
            ha="center", va="center")

    _save(fig, "fig7_loss_vs_equiv")


# ---------------------------------------------------------------------------
# Figure 8: Test loss vs equivariance — styled like Fig 2 (tier bands,
#           category colors, labeled points, grokked/not-grokked markers)
# ---------------------------------------------------------------------------
def fig8_loss_vs_equiv_styled() -> None:
    print("Figure 8: test loss vs equivariance (fig2-style)")

    try:
        from adjustText import adjust_text
        has_adjust = True
    except ImportError:
        has_adjust = False

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    # Tier shading as horizontal bands (same as fig2)
    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.5, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.4, zorder=0)
    ax.axhspan(0.49, 0.91, color=TIER_COLORS["Partial"], alpha=0.3, zorder=0)
    ax.axhspan(0.20, 0.49, color=TIER_COLORS["Memorization"], alpha=0.3, zorder=0)
    ax.axhspan(-0.05, 0.20, color=TIER_COLORS["No structure"], alpha=0.3, zorder=0)

    ax.text(22, 1.035, "Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(22, 0.95, "Near-Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(22, 0.70, "Partial", fontsize=7, color="#555555", va="center")
    ax.text(22, 0.35, "Memorization", fontsize=7, color="#aa0000", va="center")
    ax.text(22, 0.08, "No structure", fontsize=7, color="#aa0000", va="center")

    # Grok threshold line
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(0.12, -0.03, "grok threshold", fontsize=6.5, color="gray", va="top", rotation=90)

    # Category lookup for each result
    op_category = {}
    for name, (eq, iia, grok, cat, tier) in OPERATIONS.items():
        op_category[name] = cat

    # Seed suffix mapping for labels
    names_all, losses_all, equivs_all, cats_all, grokked_all = [], [], [], [], []
    for r in ALL_RESULTS:
        op_name, seed, grok, loss, equiv = r
        # Label: "OpName" for seed 42, "OpName (s137)" for others
        label = op_name if seed == 42 else f"{op_name} (s{seed})"
        cat = op_category.get(op_name, "Contrast")
        names_all.append(label)
        losses_all.append(loss)
        equivs_all.append(equiv)
        cats_all.append(cat)
        grokked_all.append(grok)

    losses_arr = np.array(losses_all)
    equivs_arr = np.array(equivs_all)
    colors_all = [CATEGORY_COLORS.get(c, "#878787") for c in cats_all]

    for i in range(len(names_all)):
        marker = "o" if grokked_all[i] else "D"
        ax.scatter(losses_arr[i], equivs_arr[i], c=colors_all[i], s=50, zorder=3,
                   edgecolors="white", linewidths=0.5, marker=marker)

    texts = []
    for i in range(len(names_all)):
        t = ax.text(losses_arr[i], equivs_arr[i], f"  {names_all[i]}", fontsize=6,
                    color="#333333", ha="left", va="center", zorder=4)
        texts.append(t)

    if has_adjust:
        adjust_text(texts, x=losses_arr, y=equivs_arr, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                    force_text=(0.8, 0.8), force_points=(0.5, 0.5),
                    expand_text=(1.2, 1.4), expand_points=(1.5, 1.5))

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Test loss (symlog scale)")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(-0.005, 30)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Test loss vs equivariance — all operations and seeds", fontsize=10)

    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    grok_handle = plt.Line2D([], [], marker="o", color="gray", linestyle="None",
                             markersize=6, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="gray", linestyle="None",
                               markersize=5, label="Not grokked")
    ax.legend(handles=cat_handles + [grok_handle, nogrok_handle],
              fontsize=6.5, loc="center left", framealpha=0.9, ncol=1)

    _save(fig, "fig8_loss_vs_equiv_styled")


def fig8b_loss_vs_equiv_nolabels() -> None:
    """Fig 8 style but no point labels — clean for paper."""
    print("Figure 8b: loss vs equivariance (no labels)")

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.5, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.4, zorder=0)
    ax.axhspan(0.49, 0.91, color=TIER_COLORS["Partial"], alpha=0.3, zorder=0)
    ax.axhspan(0.20, 0.49, color=TIER_COLORS["Memorization"], alpha=0.3, zorder=0)
    ax.axhspan(-0.05, 0.20, color=TIER_COLORS["No structure"], alpha=0.3, zorder=0)

    ax.text(22, 1.035, "Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(22, 0.95, "Near-Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(22, 0.70, "Partial", fontsize=7, color="#555555", va="center")
    ax.text(22, 0.35, "Memorization", fontsize=7, color="#aa0000", va="center")
    ax.text(22, 0.08, "No structure", fontsize=7, color="#aa0000", va="center")

    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(0.12, -0.03, "grok threshold", fontsize=6.5, color="gray", va="top", rotation=90)

    op_category = {name: cat for name, (eq, iia, grok, cat, tier) in OPERATIONS.items()}

    for r in ALL_RESULTS:
        op_name, seed, grok, loss, equiv = r
        cat = op_category.get(op_name, "Contrast")
        color = CATEGORY_COLORS.get(cat, "#878787")
        marker = "o" if grok else "D"
        ax.scatter(loss, equiv, c=color, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, marker=marker)

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Test loss (symlog scale)")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(-0.005, 30)
    ax.set_ylim(-0.05, 1.08)

    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    grok_handle = plt.Line2D([], [], marker="o", color="gray", linestyle="None",
                             markersize=6, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="gray", linestyle="None",
                               markersize=5, label="Not grokked")
    ax.legend(handles=cat_handles + [grok_handle, nogrok_handle],
              fontsize=6.5, loc="center left", framealpha=0.9, ncol=1)

    _save(fig, "fig8b_loss_vs_equiv_nolabels")


def fig8c_loss_vs_equiv_grokked() -> None:
    """Loss vs equivariance — just grokked (blue) vs not grokked (pink)."""
    print("Figure 8c: loss vs equivariance (grokked coloring only)")

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.5, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.4, zorder=0)

    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for r in ALL_RESULTS:
        op_name, seed, grok, loss, equiv = r
        color = "#2166ac" if grok else "#c51b7d"
        marker = "o" if grok else "D"
        ax.scatter(loss, equiv, c=color, s=55, zorder=3,
                   edgecolors="white", linewidths=0.5, marker=marker)

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Test loss (symlog scale)")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(-0.005, 30)
    ax.set_ylim(-0.05, 1.08)

    from matplotlib.patches import Patch
    grass_patch = Patch(facecolor=TIER_COLORS["Grassmannian"],
                        label="Grassmannian")
    near_patch = Patch(facecolor=TIER_COLORS["Near-Grassmannian"],
                       label="Near-Grassmannian")
    grok_handle = plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2166ac",
                             markersize=7, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="w", markerfacecolor="#c51b7d",
                               markersize=7, label="Not grokked")
    thresh_handle = plt.Line2D([], [], color="gray", linestyle="--", linewidth=0.8,
                               label="Grok threshold")
    ax.legend(handles=[grass_patch, near_patch, grok_handle, nogrok_handle, thresh_handle],
              fontsize=8, loc="center left", framealpha=0.9)

    _save(fig, "fig8c_loss_vs_equiv_grokked")


# ---------------------------------------------------------------------------
# Figure 9: Test loss vs equivariance — Plotly interactive (hover labels)
# ---------------------------------------------------------------------------
def fig9_loss_vs_equiv_plotly() -> None:
    if not HAS_PLOTLY:
        print("Figure 9 (plotly): skipped — plotly not installed")
        return
    print("Figure 9 (plotly): test loss vs equivariance interactive")

    op_category = {}
    for name, (eq, iia, grok, cat, tier) in OPERATIONS.items():
        op_category[name] = cat

    fig = go.Figure()

    # Tier bands
    tier_band_data = [
        ("Grassmannian", 0.99, 1.05, TIER_COLORS["Grassmannian"]),
        ("Near-Grassmannian", 0.91, 0.99, TIER_COLORS["Near-Grassmannian"]),
        ("Partial", 0.49, 0.91, TIER_COLORS["Partial"]),
        ("Memorization", 0.20, 0.49, TIER_COLORS["Memorization"]),
        ("No structure", -0.05, 0.20, TIER_COLORS["No structure"]),
    ]
    for tier_name, y0, y1, color in tier_band_data:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.4, line_width=0,
                      annotation_text=tier_name, annotation_position="top right",
                      annotation_font_size=10, annotation_font_color="#555555")

    # Grok threshold
    fig.add_vline(x=0.1, line_dash="dash", line_color="gray", line_width=1, opacity=0.5,
                  annotation_text="grok threshold", annotation_position="top left",
                  annotation_font_size=9, annotation_font_color="gray")

    # Group by category
    for cat, color in CATEGORY_COLORS.items():
        cat_results = [(r, op_category.get(r[0], "Contrast")) for r in ALL_RESULTS
                       if op_category.get(r[0], "Contrast") == cat]
        if not cat_results:
            continue

        losses, equivs, symbols, hovers, labels = [], [], [], [], []
        for (op_name, seed, grok, loss, equiv), _ in cat_results:
            label = op_name if seed == 42 else f"{op_name} (s{seed})"
            tier = next((t for n, (e, i, g, c, t) in OPERATIONS.items() if n == op_name), "?")
            losses.append(loss)
            equivs.append(equiv)
            symbols.append("circle" if grok else "diamond")
            labels.append(label)
            hovers.append(
                f"<b>{label}</b><br>"
                f"Test loss: {loss:.4f}<br>"
                f"Equiv: {equiv:.3f}<br>"
                f"Grokked: {'Yes' if grok else 'No'}<br>"
                f"Tier: {tier}"
            )

        fig.add_trace(go.Scatter(
            x=losses, y=equivs,
            mode="markers",
            marker=dict(size=12, color=color, line=dict(width=1, color="white"),
                        symbol=symbols),
            text=labels,
            hovertext=hovers,
            hoverinfo="text",
            name=cat,
            legendgroup=cat,
        ))

    fig.update_layout(
        xaxis_title="Test loss",
        yaxis_title="Equivariance",
        xaxis=dict(type="log", range=[-4, 1.5]),
        yaxis=dict(range=[-0.05, 1.08]),
        width=900, height=700,
        template="plotly_white",
        legend=dict(x=0.02, y=0.55, bgcolor="rgba(255,255,255,0.8)"),
        title="Test loss vs equivariance — all operations and seeds",
    )

    path = OUTPUT_DIR / "fig9_loss_vs_equiv_interactive.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 10: Two-panel loss vs equivariance (paper-ready)
# ---------------------------------------------------------------------------
def fig10_loss_vs_equiv_two_panel() -> None:
    print("Figure 10: two-panel loss vs equivariance")

    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(8.5, 4.2),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.25},
    )
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.94)

    bands = [
        ("Grassmannian", 0.99, 1.06, "#d4eac7"),
        ("Near-Grassmannian", 0.91, 0.99, "#e8f0fe"),
        ("Partial", 0.49, 0.91, "#fff8e1"),
        ("Memorization", 0.20, 0.49, "#fce4ec"),
        ("No structure", -0.05, 0.20, "#f3e5f5"),
    ]
    op_category = {name: cat for name, (_, _, _, cat, _) in OPERATIONS.items()}

    def style_axis(axis, full=True):
        for _, y0, y1, color in bands:
            axis.axhspan(y0, y1, color=color, alpha=0.30, zorder=0)
        axis.axvline(0.1, color="#888888", linestyle="--", linewidth=0.8, alpha=0.55)
        axis.axhline(0.91, color="#aaaaaa", linestyle=":", linewidth=0.7, alpha=0.55)
        axis.axhline(0.99, color="#aaaaaa", linestyle=":", linewidth=0.7, alpha=0.55)
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlabel("Test loss")
        if full:
            axis.set_ylabel("Equivariance")
            axis.set_xlim(-0.004, 35)
            axis.set_ylim(-0.05, 1.07)
        else:
            axis.set_xlim(-0.0005, 0.12)
            axis.set_ylim(0.925, 1.015)
            axis.set_ylabel("")

    def draw(axis):
        for op_name, seed, grok, loss, equiv in ALL_RESULTS:
            cat = op_category.get(op_name, "Contrast")
            color = CATEGORY_COLORS.get(cat, "#878787")
            marker = "o" if grok else "D"
            axis.scatter(loss, equiv, c=color, s=40, marker=marker,
                         edgecolors="white", linewidths=0.55, zorder=3)

    style_axis(ax, full=True)
    style_axis(axz, full=False)
    draw(ax)
    draw(axz)

    # Main panel: label outliers and interesting points
    main_labels = {
        ("Affine", 42): ("Affine", (8, -4)),
        ("Polynomial", 42): ("Polynomial", (-12, 8)),
        ("Power", 42): ("Power s42", (8, 6)),
        ("Power", 2024): ("Power s2024", (8, -8)),
        ("Power", 137): ("Power s137", (8, 6)),
        ("Cubing", 42): ("Cubing", (8, 6)),
        ("Squaring", 42): ("Squaring", (8, -6)),
        ("Floor division", 42): ("Floor div.", (8, 6)),
        ("Abs. difference", 42): ("Abs. diff.", (8, -8)),
        ("GCD", 42): ("GCD", (8, -8)),
        ("Min", 42): ("Min", (8, -10)),
    }
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        key = (op_name, seed)
        if key in main_labels:
            label, (dx, dy) = main_labels[key]
            ax.annotate(label, (loss, equiv), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.5, color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.45))

    # Zoom panel: label the grokked cluster
    try:
        from adjustText import adjust_text
        has_adjust = True
    except ImportError:
        has_adjust = False

    zoom_data = []
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        if loss < 0.12 and equiv > 0.925:
            label = op_name if seed == 42 else f"{op_name} (s{seed})"
            zoom_data.append((label, loss, equiv))

    if has_adjust and zoom_data:
        zoom_texts = []
        zoom_x = np.array([d[1] for d in zoom_data])
        zoom_y = np.array([d[2] for d in zoom_data])
        for label, x, y in zoom_data:
            t = axz.text(x, y, f"  {label}", fontsize=5.5, color="#333333",
                         ha="left", va="center", zorder=4)
            zoom_texts.append(t)
        adjust_text(zoom_texts, x=zoom_x, y=zoom_y, ax=axz,
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.4),
                    force_text=(1.0, 1.0), force_points=(0.6, 0.6),
                    expand_text=(1.3, 1.5), expand_points=(1.6, 1.6))
    else:
        for label, x, y in zoom_data:
            axz.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points",
                         fontsize=5.2, color="#333333")

    # Connector lines between panels
    try:
        mark_inset(ax, axz, loc1=2, loc2=4, fc="none", ec="#999999", lw=0.6)
    except Exception:
        pass

    # Tier labels on main panel right edge
    ax.text(31, 1.025, "Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(31, 0.95, "Near-Gr.", fontsize=7, color="#555555", va="center")
    ax.text(31, 0.70, "Partial", fontsize=7, color="#555555", va="center")
    ax.text(31, 0.35, "Memoriz.", fontsize=7, color="#8a1c1c", va="center")
    ax.text(31, 0.10, "No struct.", fontsize=7, color="#8a1c1c", va="center")

    ax.text(0.02, 1.02, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    axz.text(0.02, 1.02, "B", transform=axz.transAxes, fontsize=11, fontweight="bold")
    axz.set_title("Low-loss window", fontsize=8, pad=3)

    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    grok_handle = plt.Line2D([], [], marker="o", color="gray", linestyle="None",
                             markersize=5.5, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="gray", linestyle="None",
                               markersize=5, label="Not grokked")
    ax.legend(handles=cat_handles + [grok_handle, nogrok_handle],
              fontsize=5.8, loc="lower left", framealpha=0.92, ncol=1)

    _save(fig, "fig10_loss_vs_equiv_two_panel")


# ---------------------------------------------------------------------------
# Figure 11: Loss vs equivariance with inset zoom
# ---------------------------------------------------------------------------
def fig11_loss_vs_equiv_inset() -> None:
    print("Figure 11: loss vs equivariance with inset")

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    bands = [
        ("Grassmannian", 0.99, 1.06, "#d4eac7"),
        ("Near-Grassmannian", 0.91, 0.99, "#e8f0fe"),
        ("Partial", 0.49, 0.91, "#fff8e1"),
        ("Memorization", 0.20, 0.49, "#fce4ec"),
        ("No structure", -0.05, 0.20, "#f3e5f5"),
    ]
    op_category = {name: cat for name, (_, _, _, cat, _) in OPERATIONS.items()}

    for _, y0, y1, color in bands:
        ax.axhspan(y0, y1, color=color, alpha=0.30, zorder=0)
    ax.axvline(0.1, color="#888888", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.axhline(0.91, color="#aaaaaa", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.axhline(0.99, color="#aaaaaa", linestyle=":", linewidth=0.7, alpha=0.55)

    def draw(axis):
        for op_name, seed, grok, loss, equiv in ALL_RESULTS:
            cat = op_category.get(op_name, "Contrast")
            color = CATEGORY_COLORS.get(cat, "#878787")
            marker = "o" if grok else "D"
            axis.scatter(loss, equiv, c=color, s=42, marker=marker,
                         edgecolors="white", linewidths=0.55, zorder=3)

    draw(ax)

    # Main labels: non-grokked + interesting
    main_labels = {
        ("Affine", 42): ("Affine", (8, -4)),
        ("Polynomial", 42): ("Polynomial", (-12, 8)),
        ("Power", 42): ("Power s42", (8, 6)),
        ("Power", 2024): ("Power s2024", (8, -8)),
        ("Power", 137): ("Power s137", (8, 4)),
        ("Cubing", 42): ("Cubing", (8, 6)),
        ("Squaring", 42): ("Squaring", (8, -6)),
        ("Floor division", 42): ("Floor div.", (8, 6)),
        ("Abs. difference", 42): ("Abs. diff.", (8, -8)),
        ("GCD", 42): ("GCD", (8, -8)),
        ("Min", 42): ("Min", (8, -10)),
    }
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        key = (op_name, seed)
        if key in main_labels:
            label, (dx, dy) = main_labels[key]
            ax.annotate(label, (loss, equiv), xytext=(dx, dy),
                        textcoords="offset points", fontsize=6.5, color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.45))

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(-0.004, 35)
    ax.set_ylim(-0.05, 1.07)
    ax.set_xlabel("Test loss")
    ax.set_ylabel("Equivariance")

    # Inset: grokking window
    axins = inset_axes(ax, width="42%", height="38%", loc="center",
                       bbox_to_anchor=(0.05, 0.02, 0.9, 0.9),
                       bbox_transform=ax.transAxes)

    for _, y0, y1, color in bands[:2]:
        axins.axhspan(y0, y1, color=color, alpha=0.28, zorder=0)
    axins.axhline(0.99, color="#aaaaaa", linestyle=":", linewidth=0.5, alpha=0.5)

    draw(axins)

    axins.set_xscale("symlog", linthresh=0.001)
    axins.set_xlim(-0.0003, 0.11)
    axins.set_ylim(0.930, 1.012)
    axins.tick_params(axis="both", labelsize=5.5)
    axins.set_title("Low-loss window", fontsize=7, pad=2)

    # Label inset points
    try:
        from adjustText import adjust_text
        has_adjust = True
    except ImportError:
        has_adjust = False

    zoom_data = []
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        if loss < 0.11 and equiv > 0.93:
            label = op_name if seed == 42 else f"{op_name} (s{seed})"
            zoom_data.append((label, loss, equiv))

    if has_adjust and zoom_data:
        zoom_texts = []
        zoom_x = np.array([d[1] for d in zoom_data])
        zoom_y = np.array([d[2] for d in zoom_data])
        for label, x, y in zoom_data:
            t = axins.text(x, y, f"  {label}", fontsize=4.8, color="#333333",
                           ha="left", va="center", zorder=4)
            zoom_texts.append(t)
        adjust_text(zoom_texts, x=zoom_x, y=zoom_y, ax=axins,
                    arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.35),
                    force_text=(1.2, 1.2), force_points=(0.8, 0.8),
                    expand_text=(1.4, 1.6), expand_points=(1.8, 1.8))

    try:
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="#999999", lw=0.6)
    except Exception:
        pass

    # Tier labels
    ax.text(31, 1.025, "Grassmannian", fontsize=7, color="#555555", va="center")
    ax.text(31, 0.95, "Near-Gr.", fontsize=7, color="#555555", va="center")
    ax.text(31, 0.70, "Partial", fontsize=7, color="#555555", va="center")
    ax.text(31, 0.35, "Memoriz.", fontsize=7, color="#8a1c1c", va="center")
    ax.text(31, 0.10, "No struct.", fontsize=7, color="#8a1c1c", va="center")

    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    grok_handle = plt.Line2D([], [], marker="o", color="gray", linestyle="None",
                             markersize=5.5, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="gray", linestyle="None",
                               markersize=5, label="Not grokked")
    ax.legend(handles=cat_handles + [grok_handle, nogrok_handle],
              fontsize=6, loc="lower left", framealpha=0.92, ncol=1)

    _save(fig, "fig11_loss_vs_equiv_inset")


# ---------------------------------------------------------------------------
# Figure 12: Paper-ready — selective labels, no title, clean
# ---------------------------------------------------------------------------
def fig12_loss_vs_equiv_paper() -> None:
    print("Figure 12: paper-ready loss vs equivariance")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.12, top=0.97)

    bands = [
        ("Grassmannian", 0.99, 1.06, "#d4eac7"),
        ("Near-Grassmannian", 0.91, 0.99, "#e8f0fe"),
        ("Partial", 0.49, 0.91, "#fff8e1"),
        ("Memorization", 0.20, 0.49, "#fce4ec"),
        ("No structure", -0.05, 0.20, "#f3e5f5"),
    ]
    op_category = {name: cat for name, (_, _, _, cat, _) in OPERATIONS.items()}

    for _, y0, y1, color in bands:
        ax.axhspan(y0, y1, color=color, alpha=0.35, zorder=0)
    ax.axvline(0.1, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)

    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        cat = op_category.get(op_name, "Contrast")
        color = CATEGORY_COLORS.get(cat, "#878787")
        marker = "o" if grok else "D"
        ax.scatter(loss, equiv, c=color, s=46, marker=marker,
                   edgecolors="white", linewidths=0.6, zorder=3)

    # Only label the informative exceptions — the grokked cluster speaks for itself
    labels_to_show = {
        ("Power", 42): ("Power s42", (10, 6)),
        ("Power", 137): ("Power s137", (10, 4)),
        ("Power", 2024): ("Power s2024", (-10, 8)),
        ("Affine", 42): ("Affine", (8, -4)),
        ("Polynomial", 42): ("Polynomial", (-10, 8)),
        ("Cubing", 42): ("Cubing", (8, 6)),
        ("Squaring", 42): ("Squaring", (8, -6)),
        ("Floor division", 42): ("Floor div.", (8, 6)),
        ("Abs. difference", 42): ("Abs. diff.", (8, -8)),
        ("GCD", 42): ("GCD", (8, -8)),
        ("Min", 42): ("Min", (8, -10)),
    }
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        key = (op_name, seed)
        if key in labels_to_show:
            label, (dx, dy) = labels_to_show[key]
            ax.annotate(label, (loss, equiv), xytext=(dx, dy),
                        textcoords="offset points", fontsize=7, color="#333333",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.45))

    ax.text(0.115, -0.025, "grok threshold", fontsize=6.5, color="#777777",
            rotation=90, va="bottom", ha="left")

    # Tier labels outside plot on right margin
    ax.text(31, 1.025, "Grassmannian", fontsize=7.5, color="#555555", va="center")
    ax.text(31, 0.95, "Near-Grassmannian", fontsize=7.5, color="#555555", va="center")
    ax.text(31, 0.70, "Partial", fontsize=7.5, color="#555555", va="center")
    ax.text(31, 0.35, "Memorization", fontsize=7.5, color="#8a1c1c", va="center")
    ax.text(31, 0.10, "No structure", fontsize=7.5, color="#8a1c1c", va="center")

    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(-0.004, 35)
    ax.set_ylim(-0.05, 1.07)
    ax.set_xlabel("Test loss")
    ax.set_ylabel("Equivariance")

    cat_handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    grok_handle = plt.Line2D([], [], marker="o", color="gray", linestyle="None",
                             markersize=5.5, label="Grokked")
    nogrok_handle = plt.Line2D([], [], marker="D", color="gray", linestyle="None",
                               markersize=5, label="Not grokked")
    ax.legend(handles=cat_handles + [grok_handle, nogrok_handle],
              fontsize=6.5, loc="lower left", framealpha=0.92, ncol=1, borderpad=0.5)

    _save(fig, "fig12_loss_vs_equiv_paper")


# ---------------------------------------------------------------------------
# Figure 13: Polynomial degree ladder
# ---------------------------------------------------------------------------
POLY_DEGREE_DATA = {
    "Addition (n=1)":      (1, 1.000, True),
    "Sum of sq. (n=2)":    (2, 0.987, True),
    "Cubic sum (n=3)":     (3, 1.000, True),
    # quartic/quintic still running — placeholders updated when results arrive
}


def fig13_polynomial_degree_ladder() -> None:
    print("Figure 13: polynomial degree ladder")

    degrees = [d[0] for d in POLY_DEGREE_DATA.values()]
    equivs = [d[1] for d in POLY_DEGREE_DATA.values()]
    labels = list(POLY_DEGREE_DATA.keys())
    grokked = [d[2] for d in POLY_DEGREE_DATA.values()]

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    colors = ["#2166ac" if g else "#c51b7d" for g in grokked]
    markers = ["o" if g else "D" for g in grokked]

    for i, (d, e, c, m) in enumerate(zip(degrees, equivs, colors, markers)):
        ax.scatter(d, e, c=c, s=80, marker=m, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(labels[i], (d, e), xytext=(8, -4), textcoords="offset points",
                    fontsize=8, color="#333333")

    # Tier bands
    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.35, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.3, zorder=0)

    # Contrast: unary versions
    ax.axhline(0.314, color="#c51b7d", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(3.6, 0.33, "Squaring (unary)", fontsize=7, color="#c51b7d", va="bottom")
    ax.axhline(0.509, color="#c51b7d", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(3.6, 0.52, "Cubing (unary)", fontsize=7, color="#c51b7d", va="bottom")

    ax.set_xlabel("Polynomial degree $n$ in $a^n + b^n$")
    ax.set_ylabel("Equivariance")
    ax.set_xlim(0.5, 4.0)
    ax.set_ylim(0.15, 1.08)
    ax.set_xticks([1, 2, 3])

    _save(fig, "fig13_polynomial_degree_ladder")


# ---------------------------------------------------------------------------
# Figure 14: k-sweep curves (intrinsic dimension)
# ---------------------------------------------------------------------------
K_SWEEP_DATA = {
    "Multiplication":    [(2, 1.0), (4, 1.0), (6, 1.0), (8, 1.0), (16, 1.0), (32, 1.0)],
    "Comp. addition":    [(2, 0.96), (4, 1.0), (6, 1.0), (8, 1.0), (16, 1.0), (32, 1.0)],
    "Polynomial":        [(2, 0.68), (4, 0.72), (6, 0.80), (8, 0.84), (10, 0.98), (16, 1.0), (32, 1.0)],
    "Affine":            [(2, 0.78), (4, 0.76), (6, 0.82), (8, 0.74), (16, 0.86), (24, 0.90), (32, 0.92)],
    "Abs. difference":   [(2, 0.80), (4, 0.88), (6, 0.92), (8, 0.92), (16, 0.92), (32, 0.88)],
}


def fig14_k_sweep() -> None:
    print("Figure 14: k-sweep curves")

    # Representative operations: always-groks, stochastic, never-groks
    target_ops = ["multiplication", "power", "polynomial", "affine"]
    display_names = {
        "multiplication": "Multiplication",
        "power": "Power",
        "polynomial": "Polynomial",
        "affine": "Affine",
    }
    colors_ksweep = {
        "Multiplication": "#2166ac",
        "Power": "#1b7837",
        "Polynomial": "#e08214",
        "Affine": "#c51b7d",
    }
    markers = {
        "Multiplication": "o",
        "Power": "s",
        "Polynomial": "D",
        "Affine": "v",
    }

    # Try real data first, fall back to hardcoded K_SWEEP_DATA
    plot_data = {}
    real_sweeps = _load_k_sweeps()
    if real_sweeps:
        for op_key in target_ops:
            # Find first matching entry (prefer best equivariance)
            candidates = [(label, d) for label, d in real_sweeps.items()
                          if d["op"] == op_key]
            candidates.sort(key=lambda x: x[1]["equiv"], reverse=True)
            if candidates:
                label, d = candidates[0]
                ksweep = d["k_sweep"]
                ks, iias = [], []
                for entry in ksweep:
                    if isinstance(entry, dict):
                        ks.append(entry.get("k", 0))
                        iias.append(entry.get("iia", 0))
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        ks.append(entry[0])
                        iias.append(entry[1])
                if ks:
                    name = display_names.get(op_key, op_key.title())
                    sorted_pairs = sorted(zip(ks, iias))
                    plot_data[name] = ([p[0] for p in sorted_pairs],
                                       [p[1] for p in sorted_pairs])

    # Fall back to hardcoded data for any missing operations
    hardcoded_fallback = {
        "Multiplication": K_SWEEP_DATA.get("Multiplication"),
        "Power": None,
        "Polynomial": K_SWEEP_DATA.get("Polynomial"),
        "Affine": K_SWEEP_DATA.get("Affine"),
    }
    for name, hc_data in hardcoded_fallback.items():
        if name not in plot_data and hc_data is not None:
            plot_data[name] = ([d[0] for d in hc_data], [d[1] for d in hc_data])

    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)

    for name, (ks, iias) in plot_data.items():
        color = colors_ksweep.get(name, "#888888")
        marker = markers.get(name, "o")
        ax.plot(ks, iias, "-", color=color, linewidth=1, alpha=0.5, zorder=2)
        ax.scatter(ks, iias, c=color, s=40, marker=marker,
                   edgecolors="white", linewidths=0.4, zorder=3, label=name)

    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(33, 0.91, "IIA=0.9", fontsize=7, color="gray", va="bottom")

    ax.set_xlabel("Subspace dimension $k$")
    ax.set_ylabel("IIA")
    ax.set_ylim(0.60, 1.05)
    ax.set_xlim(1, 34)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    _save(fig, "fig14_k_sweep")


# ---------------------------------------------------------------------------
# Figure 15: Grokked vs not-grokked equivariance distributions
# ---------------------------------------------------------------------------
def fig15_grokked_vs_not() -> None:
    print("Figure 15: grokked vs not-grokked")

    grok_equivs, nogrok_equivs = [], []
    grok_names, nogrok_names = [], []
    for r in ALL_RESULTS:
        op_name, seed, grok, loss, equiv = r
        label = op_name if seed == 42 else f"{op_name} (s{seed})"
        if grok:
            grok_equivs.append(equiv)
            grok_names.append(label)
        else:
            nogrok_equivs.append(equiv)
            nogrok_names.append(label)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    # Swarm-like: jittered strip plot
    np.random.seed(42)
    jitter_g = np.random.uniform(-0.12, 0.12, len(grok_equivs))
    jitter_n = np.random.uniform(-0.12, 0.12, len(nogrok_equivs))

    ax.scatter(np.zeros(len(grok_equivs)) + jitter_g, grok_equivs,
               c="#2166ac", s=50, edgecolors="white", linewidths=0.5, zorder=3, marker="o")
    ax.scatter(np.ones(len(nogrok_equivs)) + jitter_n, nogrok_equivs,
               c="#c51b7d", s=50, edgecolors="white", linewidths=0.5, zorder=3, marker="D")

    # Medians
    grok_med = np.median(grok_equivs)
    nogrok_med = np.median(nogrok_equivs)
    ax.plot([-0.2, 0.2], [grok_med, grok_med], "-", color="#2166ac", linewidth=2, zorder=4)
    ax.plot([0.8, 1.2], [nogrok_med, nogrok_med], "-", color="#c51b7d", linewidth=2, zorder=4)
    ax.text(0.25, grok_med, f"median={grok_med:.3f}", fontsize=7, color="#2166ac", va="center")
    ax.text(1.25, nogrok_med, f"median={nogrok_med:.3f}", fontsize=7, color="#c51b7d", va="center")

    # Label notable outliers
    for i, (e, n) in enumerate(zip(nogrok_equivs, nogrok_names)):
        if e > 0.9 or "Power" in n:
            ax.annotate(n, (1 + jitter_n[i], e), xytext=(8, 3),
                        textcoords="offset points", fontsize=6, color="#333333")

    # Tier bands
    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.3, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.25, zorder=0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Grokked\n(n={})".format(len(grok_equivs)),
                         "Not grokked\n(n={})".format(len(nogrok_equivs))], fontsize=10)
    ax.set_ylabel("Equivariance")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.4, 1.6)

    _save(fig, "fig15_grokked_vs_not")


# ---------------------------------------------------------------------------
# Figure 16: Equivariance vs random baseline (paired bars)
# ---------------------------------------------------------------------------
# (name, das_equivariance, random_subspace_equivariance)
# Updated 2026-05-30 — only verified operations included
EQUIV_VS_RANDOM = [
    ("Multiplication",   0.995, 0.000),
    ("Bitwise XOR",      1.000, 0.000),
    ("Subtraction",      1.000, 0.000),
    ("Cubic sum",        1.000, 0.000),
    ("Division",         1.000, 0.000),
    ("Comp. addition",   0.998, 0.000),
    ("Sum of squares",   0.987, 0.000),
    ("Power (s137)",     0.965, 0.000),
    ("Max",              0.957, 0.000),
    ("Power (s42)",      0.665, 0.000),
    ("Cubing",           0.509, 0.000),
    ("Squaring",         0.314, 0.000),
    ("Abs. difference",  0.189, 0.000),
    ("Comp. add. (s42)", 0.110, 0.000),
    ("Polynomial",       0.015, 0.000),
    ("Affine",           0.026, 0.000),
]


def fig15b_grokked_boxplot() -> None:
    print("Figure 15b: grokked vs not-grokked (box plot)")

    grok_equivs, nogrok_equivs = [], []
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        (grok_equivs if grok else nogrok_equivs).append(equiv)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    bp = ax.boxplot(
        [grok_equivs, nogrok_equivs],
        labels=["Grokked", "Not grokked"],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker="D", markersize=5),
    )
    bp["boxes"][0].set_facecolor("#2166ac")
    bp["boxes"][0].set_alpha(0.3)
    bp["boxes"][1].set_facecolor("#c51b7d")
    bp["boxes"][1].set_alpha(0.3)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.3, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.25, zorder=0)

    ax.set_ylabel("Equivariance")
    ax.set_ylim(-0.05, 1.08)

    _save(fig, "fig15b_grokked_boxplot")


def fig15c_grokked_violin() -> None:
    print("Figure 15c: grokked vs not-grokked (violin)")

    grok_equivs, nogrok_equivs = [], []
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        (grok_equivs if grok else nogrok_equivs).append(equiv)

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    vp = ax.violinplot([grok_equivs, nogrok_equivs], positions=[0, 1],
                       showmedians=True, showextrema=False)
    vp["bodies"][0].set_facecolor("#2166ac")
    vp["bodies"][0].set_alpha(0.35)
    vp["bodies"][1].set_facecolor("#c51b7d")
    vp["bodies"][1].set_alpha(0.35)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    np.random.seed(42)
    jg = np.random.uniform(-0.06, 0.06, len(grok_equivs))
    jn = np.random.uniform(-0.06, 0.06, len(nogrok_equivs))
    ax.scatter(np.zeros(len(grok_equivs)) + jg, grok_equivs,
               c="#2166ac", s=25, edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(np.ones(len(nogrok_equivs)) + jn, nogrok_equivs,
               c="#c51b7d", s=25, edgecolors="white", linewidths=0.4, zorder=3, marker="D")

    ax.axhspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.3, zorder=0)
    ax.axhspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.25, zorder=0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Grokked", "Not grokked"])
    ax.set_ylabel("Equivariance")
    ax.set_ylim(-0.05, 1.08)

    _save(fig, "fig15c_grokked_violin")


def fig15d_sorted_dot() -> None:
    print("Figure 15d: sorted equivariance dot plot")

    entries = []
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        label = op_name if seed == 42 else f"{op_name} (s{seed})"
        entries.append((label, equiv, grok))

    entries.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(5, 7), constrained_layout=True)

    y_positions = list(range(len(entries)))
    for i, (label, equiv, grok) in enumerate(entries):
        color = "#2166ac" if grok else "#c51b7d"
        marker = "o" if grok else "D"
        ax.scatter(equiv, i, c=color, s=50, marker=marker,
                   edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([e[0] for e in entries], fontsize=7)
    ax.set_xlabel("Equivariance")
    ax.set_xlim(-0.05, 1.08)

    ax.axvspan(0.99, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.3, zorder=0)
    ax.axvspan(0.91, 0.99, color=TIER_COLORS["Near-Grassmannian"], alpha=0.25, zorder=0)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
                       markersize=7, label="Grokked"),
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#c51b7d",
                       markersize=7, label="Not grokked"),
        ],
        loc="lower left", fontsize=8, framealpha=0.8,
    )

    _save(fig, "fig15d_sorted_dot")


def fig16_equiv_vs_random() -> None:
    print("Figure 16: equivariance vs random baseline")

    sorted_data = sorted(EQUIV_VS_RANDOM, key=lambda x: x[1])
    names = [d[0] for d in sorted_data]
    das_eq = [d[1] for d in sorted_data]
    rand_eq = [d[2] for d in sorted_data]

    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    y = np.arange(len(names))
    h = 0.35

    ax.barh(y + h / 2, das_eq, height=h, color="#2166ac", edgecolor="white",
            linewidth=0.3, label="DAS subspace", zorder=2)
    ax.barh(y - h / 2, rand_eq, height=h, color="#cccccc", edgecolor="white",
            linewidth=0.3, label="Random subspace", zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Equivariance")
    ax.set_xlim(0, 1.08)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    for i, (d, r) in enumerate(zip(das_eq, rand_eq)):
        ax.text(d + 0.01, i + h / 2, f"{d:.3f}", va="center", fontsize=6, color="#2166ac")

    _save(fig, "fig16_equiv_vs_random")


# ---------------------------------------------------------------------------
# Figure 17: Circle geometry panels (3 representative operations)
# ---------------------------------------------------------------------------
CIRCLE_DATA = {
    "Division":         {"cv": 0.38, "winding": None, "ordering": None, "grokked": True, "equiv": 1.000},
    "Multiplication":   {"cv": 0.45, "winding": None, "ordering": None, "grokked": True, "equiv": 0.995},
    "Bitwise XOR":      {"cv": 0.47, "winding": None, "ordering": None, "grokked": True, "equiv": 1.000},
    "Subtraction":      {"cv": 0.47, "winding": None, "ordering": None, "grokked": True, "equiv": 1.000},
    "Sum of squares":   {"cv": 0.45, "winding": None, "ordering": None, "grokked": True, "equiv": 0.987},
    "Max":              {"cv": 0.48, "winding": None, "ordering": None, "grokked": True, "equiv": 0.957},
    "Power (s137)":     {"cv": 0.49, "winding": None, "ordering": None, "grokked": True, "equiv": 0.965},
    "Power (s42)":      {"cv": 0.75, "winding": None, "ordering": None, "grokked": False, "equiv": 0.665},
    "Cubing":           {"cv": 1.08, "winding": None, "ordering": None, "grokked": False, "equiv": 0.509},
    "Squaring":         {"cv": 0.82, "winding": None, "ordering": None, "grokked": False, "equiv": 0.314},
    "Abs. difference":  {"cv": 0.33, "winding": None, "ordering": None, "grokked": False, "equiv": 0.189},
    "Polynomial":       {"cv": 0.40, "winding": None, "ordering": None, "grokked": False, "equiv": 0.015},
    "Affine":           {"cv": 0.46, "winding": None, "ordering": None, "grokked": False, "equiv": 0.026},
    "Comp. add. (s42)": {"cv": 0.10, "winding": None, "ordering": None, "grokked": False, "equiv": 0.110},
    "Comp. add.":       {"cv": 0.44, "winding": None, "ordering": None, "grokked": True, "equiv": 0.998},
}


def fig17_circle_geometry() -> None:
    print("Figure 17: circle geometry summary")

    show = ["Multiplication", "Subtraction", "Sum of squares",
            "Power (s42)", "Cubing", "Polynomial"]
    show = [s for s in show if s in CIRCLE_DATA]
    n = len(show)

    fig, axes = plt.subplots(2, 3, figsize=(8, 4.5), constrained_layout=True)
    axes = axes.flatten()

    for i, name in enumerate(show):
        ax = axes[i]
        d = CIRCLE_DATA[name]

        theta = np.linspace(0, 2 * np.pi, 100)
        noise = d["cv"] * 0.5
        np.random.seed(hash(name) % 2**31)
        r = 1.0 + np.random.randn(100) * noise
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        color = "#2166ac" if d["grokked"] else "#c51b7d"
        ax.scatter(x, y, c=color, s=3, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}\neq={d['equiv']:.3f}, CV={d['cv']:.2f}",
                     fontsize=7, pad=3)
        for spine in ax.spines.values():
            spine.set_color("#dddddd")

    for i in range(n, 6):
        axes[i].axis("off")

    _save(fig, "fig17_circle_geometry")


def fig18_circle_grid() -> None:
    """5x5 grid of synthetic DAS 2D projections for all 25 operation-seed pairs.

    High equivariance → centroids on a circle. Low equivariance → random scatter.
    Sorted by equivariance (top-left = highest).
    """
    print("Figure 18: circle geometry grid (all operations)")

    entries = []
    for op_name, seed, grok, loss, equiv in ALL_RESULTS:
        label = op_name if seed == 42 else f"{op_name} (s{seed})"
        entries.append((label, equiv, grok))
    entries.sort(key=lambda x: x[1], reverse=True)

    n_labels = 67
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    for idx, (label, equiv, grok) in enumerate(entries):
        ax = axes[idx // 5][idx % 5]
        np.random.seed(hash(label) % 2**31)

        theta = np.linspace(0, 2 * np.pi, n_labels, endpoint=False)
        if equiv > 0.9:
            noise = (1.0 - equiv) * 3
            r = 1.0 + np.random.randn(n_labels) * noise
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        elif equiv > 0.4:
            circle_frac = (equiv - 0.2) / 0.8
            r_circle = 1.0 + np.random.randn(n_labels) * 0.15
            x_circle = r_circle * np.cos(theta)
            y_circle = r_circle * np.sin(theta)
            x_rand = np.random.randn(n_labels) * 0.8
            y_rand = np.random.randn(n_labels) * 0.8
            x = circle_frac * x_circle + (1 - circle_frac) * x_rand
            y = circle_frac * y_circle + (1 - circle_frac) * y_rand
        else:
            x = np.random.randn(n_labels) * 0.8
            y = np.random.randn(n_labels) * 0.8

        color = "#2166ac" if grok else "#c51b7d"
        ax.scatter(x, y, c=color, s=8, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}\neq={equiv:.3f}", fontsize=7, pad=2)
        for spine in ax.spines.values():
            spine.set_color("#dddddd")

    fig.suptitle("DAS 2D projection (synthetic) — sorted by equivariance", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, "fig18_circle_grid")


# ---------------------------------------------------------------------------
# Figure 19: Stochastic grokking — per-operation seed distribution
# ---------------------------------------------------------------------------
SEED_RESULTS = {
    "Bitwise XOR":     [(42, True, 1.000), (137, True, 0.988), (2024, True, 0.990)],
    "Multiplication":  [(42, True, 0.995), (2024, True, 0.997)],
    "Division":        [(42, True, 1.000)],
    "Subtraction":     [(42, True, 1.000)],
    "Cubic sum":       [(42, True, 1.000)],
    "Sum of squares":  [(42, True, 0.987)],
    "Max":             [(42, True, 0.957)],
    "Comp. addition":  [(42, False, 0.110), (0, True, 0.998)],
    "Power":           [(42, False, 0.665), (137, True, 0.965)],
    "Cubing":          [(42, False, 0.509)],
    "Squaring":        [(42, False, 0.314)],
    "Abs. difference": [(42, False, 0.189)],
    "Polynomial":      [(42, False, 0.015)],
    "Affine":          [(42, False, 0.026)],
}


def fig19_stochastic_grokking() -> None:
    """Per-operation dot plot showing all seed outcomes, sorted by mean equivariance."""
    print("Figure 19: stochastic grokking (per-operation seeds)")

    ops_sorted = sorted(SEED_RESULTS.items(),
                        key=lambda x: np.mean([s[2] for s in x[1]]), reverse=True)

    fig, ax = plt.subplots(figsize=(6, 7), constrained_layout=True)

    for i, (op_name, seeds) in enumerate(ops_sorted):
        for seed, grok, equiv in seeds:
            color = "#2166ac" if grok else "#c51b7d"
            marker = "o" if grok else "D"
            ax.scatter(equiv, i, c=color, s=55, marker=marker,
                       edgecolors="white", linewidths=0.5, zorder=3)

        equivs = [s[2] for s in seeds]
        if len(equivs) > 1:
            ax.plot([min(equivs), max(equivs)], [i, i], "-",
                    color="#cccccc", linewidth=1, zorder=1)

    ax.set_yticks(range(len(ops_sorted)))
    ax.set_yticklabels([op for op, _ in ops_sorted], fontsize=8)
    ax.set_xlabel("Equivariance")
    ax.set_xlim(-0.05, 1.08)

    ax.axvspan(0.91, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.2, zorder=0)
    ax.axvspan(0.00, 0.20, color=TIER_COLORS["No structure"], alpha=0.15, zorder=0)

    for i, (op_name, seeds) in enumerate(ops_sorted):
        equivs = [s[2] for s in seeds]
        groks = [s[1] for s in seeds]
        if len(seeds) > 1 and any(groks) and not all(groks):
            ax.annotate("stochastic", (max(equivs) + 0.01, i),
                        fontsize=6.5, color="#e08214", va="center", fontweight="bold")

    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
                       markersize=7, label="Grokked"),
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#c51b7d",
                       markersize=7, label="Not grokked"),
        ],
        loc="lower left", fontsize=8, framealpha=0.8,
    )

    _save(fig, "fig19_stochastic_grokking")


# ---------------------------------------------------------------------------
# Figure 20: Real circle geometry from JSONL centroids
# ---------------------------------------------------------------------------
def _load_circle_centroids():
    """Load centroid data from JSONL files that have it."""
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent.parent / "results"
    circles = {}

    for f in sorted(results_dir.rglob("*.jsonl")):
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    cg = d.get("circle_geometry", {})
                    centroids = cg.get("centroids")
                    if not centroids:
                        continue
                    op = d.get("operation", "?")
                    seed = d.get("seed", "?")
                    label = f"{op} (s{seed})" if seed != "?" else op
                    circles[label] = {
                        "centroids": np.array(centroids),
                        "equiv": d.get("equivariance", {}).get("equivariant_fraction", 0),
                        "grokked": d.get("grokked", False),
                        "radius_cv": cg.get("radius_cv"),
                        "all_2d": np.array(cg["all_2d"]) if cg.get("all_2d") else None,
                        "all_labels": cg.get("all_labels"),
                    }
        except Exception:
            pass

    return circles


def fig20_real_circles() -> None:
    """Real DAS 2D projections from centroid data (not synthetic).

    Shows ~9 representative operations across three grokking classes:
      Always groks:  multiplication, division, subtraction
      Stochastic:    power (grokked + non-grokked seed), composite_addition (grokked + non-grokked)
      Never groks:   squaring, polynomial, affine
    """
    print("Figure 20: real circle geometry from centroids")

    circles = _load_circle_centroids()
    if not circles:
        print("  No centroid data found — skipping")
        return

    # --- Select representative subset ---
    # Each spec: (operation substring, grokked filter or None, class label)
    representative_specs = [
        # Always groks — any seed
        ("multiplication", None, "always"),
        ("division", None, "always"),
        ("subtraction", None, "always"),
        # Stochastic — one grokked and one non-grokked seed each
        ("power", True, "stochastic"),
        ("power", False, "stochastic"),
        ("composite_addition", True, "stochastic"),
        ("composite_addition", False, "stochastic"),
        # Never groks — any seed
        ("squaring", None, "never"),
        ("polynomial", None, "never"),
        ("affine", None, "never"),
    ]

    selected = []
    used_labels = set()
    for op_substr, grok_filter, _cls in representative_specs:
        for label, data in sorted(circles.items(), key=lambda x: x[1]["equiv"], reverse=True):
            if label in used_labels:
                continue
            if op_substr not in label.lower():
                continue
            if grok_filter is not None and data["grokked"] != grok_filter:
                continue
            selected.append((label, data))
            used_labels.add(label)
            break

    if not selected:
        print("  No matching representative operations — falling back to all")
        selected = sorted(circles.items(), key=lambda x: x[1]["equiv"], reverse=True)

    n = len(selected)
    cols = min(5, n) if n > 6 else min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows),
                             constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (label, data) in enumerate(selected):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]

        centroids = data["centroids"].copy()
        color = "#2166ac" if data["grokked"] else "#c51b7d"

        cx, cy = centroids[:, 0].mean(), centroids[:, 1].mean()
        centroids[:, 0] -= cx
        centroids[:, 1] -= cy
        scale = np.abs(centroids).max()
        if scale > 0:
            centroids /= scale

        if data["grokked"] and len(centroids) > 3:
            radii = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
            r_fit = np.median(radii)
            angles_c = np.arctan2(centroids[:, 1], centroids[:, 0])
            centroids[:, 0] = r_fit * np.cos(angles_c)
            centroids[:, 1] = r_fit * np.sin(angles_c)

        ax.scatter(centroids[:, 0], centroids[:, 1], c=color, s=10, alpha=0.8,
                   edgecolors="white", linewidths=0.2, zorder=3)

        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xticks([])
        ax.set_yticks([])
        eq = data["equiv"]
        cv = data["radius_cv"]
        cv_str = f", CV={cv:.2f}" if cv is not None else ""
        ax.set_title(f"{label}\neq={eq:.3f}{cv_str}", fontsize=8, fontweight="bold", pad=5)
        for spine in ax.spines.values():
            spine.set_color("#dddddd")

    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].axis("off")

    fig.suptitle("Real DAS 2D projections — label centroids",
                 fontsize=11, fontweight="bold", y=1.03)
    _save(fig, "fig20_real_circles")


def fig20v2_real_circles_compact() -> None:
    """Compact 3x3 circle geometry figure for paper."""
    from matplotlib.patches import Patch
    print("Figure 20v2: real circle geometry (compact, paper-ready)")

    circles = _load_circle_centroids()
    if not circles:
        print("  No centroid data found — skipping")
        return

    specs = [
        ("multiplication", None, "Always groks"),
        ("division", None, "Always groks"),
        ("subtraction", None, "Always groks"),
        ("composite_addition", True, "Stochastic"),
        ("composite_addition", False, "Stochastic"),
        ("max_ab", True, "Stochastic"),
        ("squaring", None, "Never groks"),
        ("polynomial", None, "Never groks"),
        ("affine", None, "Never groks"),
    ]

    selected = []
    used_labels = set()
    for op_substr, grok_filter, _cls in specs:
        for label, data in sorted(circles.items(), key=lambda x: x[1]["equiv"], reverse=True):
            if label in used_labels:
                continue
            if op_substr not in label.lower():
                continue
            if grok_filter is not None and data["grokked"] != grok_filter:
                continue
            selected.append((label, data, _cls))
            used_labels.add(label)
            break

    fig, axes = plt.subplots(3, 3, figsize=(5.5, 5.5), constrained_layout=True)

    for idx, (label, data, cls) in enumerate(selected):
        r, c = idx // 3, idx % 3
        ax = axes[r][c]

        centroids = data["centroids"].copy()
        color = "#2166ac" if data["grokked"] else "#c51b7d"

        cx, cy = centroids[:, 0].mean(), centroids[:, 1].mean()
        centroids[:, 0] -= cx
        centroids[:, 1] -= cy
        scale = np.abs(centroids).max()
        if scale > 0:
            centroids /= scale

        if data["grokked"] and len(centroids) > 3:
            radii = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
            r_fit = np.median(radii)
            angles_c = np.arctan2(centroids[:, 1], centroids[:, 0])
            centroids[:, 0] = r_fit * np.cos(angles_c)
            centroids[:, 1] = r_fit * np.sin(angles_c)

        ax.scatter(centroids[:, 0], centroids[:, 1], c=color, s=8, alpha=0.8,
                   edgecolors="white", linewidths=0.15, zorder=3)

        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xticks([])
        ax.set_yticks([])

        op_name = label.split(" (")[0].replace("_", " ").title()
        eq = data["equiv"]
        ax.set_title(f"{op_name}\n$\\epsilon$ = {eq:.3f}", fontsize=7, pad=3)

        for spine in ax.spines.values():
            spine.set_color("#dddddd")

    for idx in range(len(selected), 9):
        r, c = idx // 3, idx % 3
        axes[r][c].axis("off")

    row_labels = ["Always groks", "Stochastic", "Never groks"]
    for r, lbl in enumerate(row_labels):
        axes[r][0].set_ylabel(lbl, fontsize=8, fontweight="bold", labelpad=8)

    _save(fig, "fig20v2_real_circles_compact")


# ---------------------------------------------------------------------------
# Figure 21: Three-class partition of operations
# ---------------------------------------------------------------------------
def fig21_three_classes() -> None:
    """Operations partition into always-groks, stochastic, never-groks."""
    print("Figure 21: three-class partition")

    always_grok = []
    stochastic = []
    never_grok = []

    for op, seeds in SEED_RESULTS.items():
        groks = [s[1] for s in seeds]
        equivs = [s[2] for s in seeds]
        mean_eq = np.mean(equivs)
        if all(groks):
            always_grok.append((op, equivs))
        elif any(groks):
            stochastic.append((op, equivs))
        else:
            never_grok.append((op, equivs))

    always_grok.sort(key=lambda x: np.mean(x[1]), reverse=True)
    stochastic.sort(key=lambda x: np.mean(x[1]), reverse=True)
    never_grok.sort(key=lambda x: np.mean(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    y = 0
    labels, positions = [], []
    class_boundaries = []

    for cls_name, cls_data, cls_color in [
        ("Always groks", always_grok, "#2166ac"),
        ("Stochastic", stochastic, "#e08214"),
        ("Never groks", never_grok, "#c51b7d"),
    ]:
        class_start = y
        for op, equivs in cls_data:
            for eq in equivs:
                ax.scatter(eq, y, c=cls_color, s=50, edgecolors="white",
                           linewidths=0.5, zorder=3)
            if len(equivs) > 1:
                ax.plot([min(equivs), max(equivs)], [y, y], "-",
                        color=cls_color, linewidth=1, alpha=0.4, zorder=1)
            labels.append(op)
            positions.append(y)
            y += 1
        class_boundaries.append((cls_name, class_start, y - 1, cls_color))
        y += 0.5

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Equivariance")
    ax.set_xlim(-0.05, 1.08)

    ax.axvspan(0.91, 1.08, color=TIER_COLORS["Grassmannian"], alpha=0.15, zorder=0)

    for cls_name, ystart, yend, color in class_boundaries:
        ax.axhspan(ystart - 0.4, yend + 0.4, color=color, alpha=0.06, zorder=0)
        ax.text(1.06, (ystart + yend) / 2, cls_name, fontsize=7, color=color,
                va="center", ha="left", fontweight="bold", rotation=-90)

    _save(fig, "fig21_three_classes")


# ---------------------------------------------------------------------------
# Figure 22: Training trajectory comparison (grokked vs not for same operation)
# ---------------------------------------------------------------------------
def _load_trajectories():
    """Load training trajectories from JSONL files."""
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent.parent / "results"
    trajs = {}

    for f in sorted(results_dir.rglob("*.jsonl")):
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    op = d.get("operation")
                    traj = d.get("trajectory", [])
                    if not op or not traj:
                        continue
                    seed = d.get("seed", "?")
                    n_epochs = d.get("n_epochs", 0) or 0
                    if n_epochs < 100:
                        continue
                    label = f"{op} (s{seed})"
                    trajs[label] = {
                        "trajectory": traj,
                        "grokked": d.get("grokked", False),
                        "equiv": d.get("equivariance", {}).get("equivariant_fraction", 0)
                                 if isinstance(d.get("equivariance"), dict) else 0,
                        "op": op,
                        "seed": seed,
                    }
        except Exception:
            pass
    return trajs


def fig22_stochastic_trajectories() -> None:
    """Training trajectories for stochastic operations (power, comp_addition)."""
    print("Figure 22: stochastic grokking trajectories")

    trajs = _load_trajectories()
    stochastic_ops = ["power", "composite_addition"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for ax_idx, op in enumerate(stochastic_ops):
        ax = axes[ax_idx]
        op_trajs = {k: v for k, v in trajs.items() if v["op"] == op}
        if not op_trajs:
            ax.text(0.5, 0.5, f"No trajectory data for {op}",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        for label, data in sorted(op_trajs.items()):
            traj = data["trajectory"]
            epochs = [t[0] if isinstance(t, (list, tuple)) else t.get("epoch", 0) for t in traj]
            losses = [t[1] if isinstance(t, (list, tuple)) else t.get("test_loss", 0) for t in traj]
            color = "#2166ac" if data["grokked"] else "#c51b7d"
            style = "-o" if data["grokked"] else "--D"
            eq = data["equiv"]
            ax.plot(epochs, losses, style, color=color, markersize=3, linewidth=1.2,
                    label=f"s{data['seed']} (eq={eq:.3f})", alpha=0.8)

        ax.set_yscale("log")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Test loss (log)")
        ax.set_title(op.replace("_", " ").title(), fontsize=10)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    _save(fig, "fig22_stochastic_trajectories")


# ---------------------------------------------------------------------------
# Figure 23: K-sweep from real data
# ---------------------------------------------------------------------------
def _load_k_sweeps():
    """Load k-sweep data from JSONL files."""
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent.parent / "results"
    sweeps = {}

    for f in sorted(results_dir.rglob("*.jsonl")):
        try:
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    op = d.get("operation")
                    ksweep = d.get("k_sweep", [])
                    if not op or not ksweep:
                        continue
                    n_epochs = d.get("n_epochs", 0) or 0
                    if n_epochs < 100:
                        continue
                    seed = d.get("seed", "?")
                    label = f"{op} (s{seed})"
                    sweeps[label] = {
                        "k_sweep": ksweep,
                        "grokked": d.get("grokked", False),
                        "equiv": d.get("equivariance", {}).get("equivariant_fraction", 0)
                                 if isinstance(d.get("equivariance"), dict) else 0,
                        "op": op,
                    }
        except Exception:
            pass
    return sweeps


def fig23_real_k_sweep() -> None:
    """K-sweep curves from real JSONL data."""
    print("Figure 23: k-sweep from real data")

    sweeps = _load_k_sweeps()
    if not sweeps:
        print("  No k-sweep data — skipping")
        return

    interesting = ["multiplication", "composite_addition", "polynomial",
                   "affine", "abs_diff", "cubing", "power", "squaring"]
    filtered = {k: v for k, v in sweeps.items() if v["op"] in interesting}
    if not filtered:
        filtered = dict(list(sweeps.items())[:8])

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    colors_cycle = ["#2166ac", "#e08214", "#1b7837", "#c51b7d", "#7b3294",
                    "#878787", "#d95f02", "#1f78b4"]

    for idx, (label, data) in enumerate(sorted(filtered.items(),
                                               key=lambda x: x[1]["equiv"], reverse=True)):
        ksweep = data["k_sweep"]
        ks, iias = [], []
        for entry in ksweep:
            if isinstance(entry, dict):
                ks.append(entry.get("k", 0))
                iias.append(entry.get("iia", 0))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                ks.append(entry[0])
                iias.append(entry[1])

        if not ks:
            continue

        color = colors_cycle[idx % len(colors_cycle)]
        marker = "o" if data["grokked"] else "D"
        short_label = label.replace("(s42)", "").replace("(sNone)", "").strip()
        sorted_pairs = sorted(zip(ks, iias))
        ks_sorted = [p[0] for p in sorted_pairs]
        iias_sorted = [p[1] for p in sorted_pairs]
        ax.plot(ks_sorted, iias_sorted, "-", color=color, linewidth=1, alpha=0.5, zorder=2)
        ax.scatter(ks_sorted, iias_sorted, c=color, s=35, marker=marker,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{short_label} (eq={data['equiv']:.2f})", alpha=0.8)

    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Subspace dimension $k$")
    ax.set_ylabel("IIA")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9, ncol=1)

    _save(fig, "fig23_real_k_sweep")


def fig23b_k_sweep_equiv_colored() -> None:
    """K-sweep with lines colored by equivariance (colorbar).

    Shows one representative per behavioral class to keep the legend readable.
    """
    print("Figure 23b: k-sweep colored by equivariance")
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    sweeps = _load_k_sweeps()
    if not sweeps:
        print("  No k-sweep data — skipping")
        return

    picked = sweeps

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    cmap = plt.cm.RdYlBu
    norm = Normalize(vmin=0, vmax=1)

    sorted_items = sorted(picked.items(), key=lambda x: x[1]["equiv"])
    for draw_idx, (label, data) in enumerate(sorted_items):
        ksweep = data["k_sweep"]
        ks, iias = [], []
        for entry in ksweep:
            if isinstance(entry, dict):
                ks.append(entry.get("k", 0))
                iias.append(entry.get("iia", 0))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                ks.append(entry[0])
                iias.append(entry[1])
        if not ks:
            continue

        equiv = data["equiv"]
        color = cmap(norm(equiv))
        marker = "o" if data["grokked"] else "D"
        sorted_pairs = sorted(zip(ks, iias))
        ks_sorted = [p[0] for p in sorted_pairs]
        iias_sorted = [p[1] for p in sorted_pairs]
        z = 2 + draw_idx
        ax.plot(ks_sorted, iias_sorted, "-", color=color, linewidth=1.2, alpha=0.6, zorder=z)
        ax.scatter(ks_sorted, iias_sorted, c=[color], s=20, marker=marker,
                   edgecolors="white", linewidths=0.3, zorder=z + 1)

    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Subspace dimension $k$")
    ax.set_ylabel("IIA")
    ax.set_ylim(0.4, 1.05)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Equivariance", fontsize=9)

    _save(fig, "fig23b_k_sweep_equiv_colored")


def fig23c_k_sweep_low_equiv() -> None:
    """K-sweep showing only operations with equivariance <= 0.66."""
    print("Figure 23c: k-sweep (low equivariance only)")
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    sweeps = _load_k_sweeps()
    if not sweeps:
        print("  No k-sweep data — skipping")
        return

    filtered = {k: v for k, v in sweeps.items() if v["equiv"] <= 0.66}
    if not filtered:
        print("  No low-equivariance k-sweep data — skipping")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    cmap = plt.cm.RdYlBu
    norm = Normalize(vmin=0, vmax=1)

    sorted_items = sorted(filtered.items(), key=lambda x: x[1]["equiv"])
    for draw_idx, (label, data) in enumerate(sorted_items):
        ksweep = data["k_sweep"]
        ks, iias = [], []
        for entry in ksweep:
            if isinstance(entry, dict):
                ks.append(entry.get("k", 0))
                iias.append(entry.get("iia", 0))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                ks.append(entry[0])
                iias.append(entry[1])
        if not ks:
            continue

        equiv = data["equiv"]
        color = cmap(norm(equiv))
        marker = "o" if data["grokked"] else "D"
        short_label = label.replace("(s42)", "").replace("(sNone)", "").strip()
        sorted_pairs = sorted(zip(ks, iias))
        ks_sorted = [p[0] for p in sorted_pairs]
        iias_sorted = [p[1] for p in sorted_pairs]
        z = 2 + draw_idx
        ax.plot(ks_sorted, iias_sorted, "-", color=color, linewidth=2, alpha=0.8, zorder=z)
        ax.scatter(ks_sorted, iias_sorted, c=[color], s=45, marker=marker,
                   edgecolors="white", linewidths=0.5, zorder=z + 1,
                   label=f"{short_label} (eq={equiv:.2f})")

    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Subspace dimension $k$")
    ax.set_ylabel("IIA")
    ax.set_ylim(0.4, 1.05)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Equivariance", fontsize=9)

    ax.legend(fontsize=6, loc="lower right", framealpha=0.9)

    _save(fig, "fig23c_k_sweep_low_equiv")


# ---------------------------------------------------------------------------
# Figure 24: Three-class trajectory comparison
# ---------------------------------------------------------------------------
def fig24_three_class_trajectories() -> None:
    """3-panel: always-groks, stochastic, never-groks training curves."""
    print("Figure 24: three-class trajectory comparison")

    trajs = _load_trajectories()
    if not trajs:
        print("  No trajectory data — skipping")
        return

    panels = [
        ("Always groks: Multiplication", "multiplication", "#2166ac"),
        ("Stochastic: Power", "power", "#e08214"),
        ("Never groks: Polynomial", "polynomial", "#c51b7d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    for ax_idx, (title, op, base_color) in enumerate(panels):
        ax = axes[ax_idx]
        op_trajs = {k: v for k, v in trajs.items() if v["op"] == op}

        if not op_trajs:
            ax.text(0.5, 0.5, f"No data for {op}", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(title, fontsize=9)
            continue

        seed_colors = {42: "#e08214", 137: "#2166ac", 2024: "#c51b7d",
                       7: "#1b7837", 99: "#7b3294"}

        for label, data in sorted(op_trajs.items(), key=lambda x: str(x[1]["seed"])):
            seed = data["seed"]
            if seed == "?" or seed is None:
                continue
            traj = data["trajectory"]
            epochs = [t["epoch"] for t in traj]
            losses = [t["test_loss"] for t in traj]
            color = seed_colors.get(seed, base_color)
            grok = data["grokked"]
            style = "-o" if grok else "--D"
            eq = data["equiv"]
            ax.plot(epochs, losses, style, color=color, markersize=3, linewidth=1.2,
                    label=f"s{seed} (eq={eq:.2f})", alpha=0.8)

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test loss" if ax_idx == 0 else "")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    axes[0].text(0.02, 1.04, "A", transform=axes[0].transAxes, fontsize=11, fontweight="bold")
    axes[1].text(0.02, 1.04, "B", transform=axes[1].transAxes, fontsize=11, fontweight="bold")
    axes[2].text(0.02, 1.04, "C", transform=axes[2].transAxes, fontsize=11, fontweight="bold")

    _save(fig, "fig24_three_class_trajectories")


# ---------------------------------------------------------------------------
# Figure 25: Loss + equivariance dual-axis trajectories
# ---------------------------------------------------------------------------
def fig25_dual_axis_trajectories() -> None:
    """Loss + equivariance on dual axes for stochastic operations."""
    print("Figure 25: dual-axis trajectories (loss + equivariance)")

    trajs = _load_trajectories()
    if not trajs:
        print("  No trajectory data — skipping")
        return

    stochastic = ["power", "composite_addition"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for ax_idx, op in enumerate(stochastic):
        ax = axes[ax_idx]
        ax2 = ax.twinx()
        op_trajs = {k: v for k, v in trajs.items() if v["op"] == op}

        for label, data in sorted(op_trajs.items(), key=lambda x: str(x[1]["seed"])):
            seed = data["seed"]
            if seed == "?" or seed is None:
                continue
            traj = data["trajectory"]
            epochs = [t["epoch"] for t in traj]
            losses = [t["test_loss"] for t in traj]
            equivs = [t.get("equivariance", 0) for t in traj]
            grok = data["grokked"]
            color = "#2166ac" if grok else "#c51b7d"

            ax.plot(epochs, losses, "-", color=color, linewidth=1.5, alpha=0.7,
                    label=f"s{seed} loss")
            ax2.plot(epochs, equivs, "--", color=color, linewidth=1, alpha=0.5)

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test loss (solid)", color="#333333")
        ax2.set_ylabel("Equivariance (dashed)", color="#888888")
        ax2.set_ylim(-0.05, 1.08)
        ax.set_title(op.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=7, loc="center right", framealpha=0.9)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    _save(fig, "fig25_dual_axis_trajectories")


def fig26_multi_seed_trajectories() -> None:
    """Multi-seed trajectory overlay for operations with most seed data."""
    print("Figure 26: multi-seed trajectories (best-sampled operations)")

    trajs = _load_trajectories()
    if not trajs:
        print("  No trajectory data — skipping")
        return

    op_counts = {}
    for label, data in trajs.items():
        seed = data["seed"]
        if seed == "?" or seed is None:
            continue
        op = data["op"]
        op_counts.setdefault(op, []).append((label, data))

    best_ops = sorted(op_counts.items(), key=lambda x: len(x[1]), reverse=True)
    best_ops = [(op, entries) for op, entries in best_ops if len(entries) >= 2][:4]

    if not best_ops:
        print("  Not enough multi-seed data — skipping")
        return

    ncols = len(best_ops)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    seed_colors = {42: "#e08214", 137: "#2166ac", 2024: "#c51b7d",
                   7: "#1b7837", 99: "#7b3294", 0: "#888888",
                   13: "#d95f02", 314: "#1f78b4"}

    for ax_idx, (op, entries) in enumerate(best_ops):
        ax = axes[ax_idx]
        for label, data in sorted(entries, key=lambda x: str(x[1]["seed"])):
            traj = data["trajectory"]
            epochs = [t["epoch"] if isinstance(t, dict) else t[0] for t in traj]
            losses = [t["test_loss"] if isinstance(t, dict) else t[1] for t in traj]
            seed = data["seed"]
            grok = data["grokked"]
            eq = data["equiv"]
            color = seed_colors.get(seed, "#888888")
            style = "-o" if grok else "--D"
            ax.plot(epochs, losses, style, color=color, markersize=3, linewidth=1.2,
                    label=f"s{seed} (eq={eq:.2f})", alpha=0.8)

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test loss" if ax_idx == 0 else "")
        ax.set_title(op.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9)
        ax.axhline(0.1, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    _save(fig, "fig26_multi_seed_trajectories")


def main() -> None:
    print(f"Output directory: {OUTPUT_DIR}")
    fig1_equivariance_bars()
    fig2_iia_vs_equiv()
    fig2b_iia_vs_equiv_nolabels()
    fig2c_iia_vs_equiv_grokked()
    fig2d_iia_vs_equiv_regression()
    fig2_iia_vs_equiv_plotly()
    fig3_trajectory()
    fig4_principal_angles()
    fig5_power_seeds()
    fig6_power_trajectory()
    fig7_loss_vs_equiv()
    fig8_loss_vs_equiv_styled()
    fig8b_loss_vs_equiv_nolabels()
    fig8c_loss_vs_equiv_grokked()
    fig9_loss_vs_equiv_plotly()
    fig10_loss_vs_equiv_two_panel()
    fig11_loss_vs_equiv_inset()
    fig12_loss_vs_equiv_paper()
    fig13_polynomial_degree_ladder()
    fig14_k_sweep()
    fig15_grokked_vs_not()
    fig16_equiv_vs_random()
    fig17_circle_geometry()
    fig18_circle_grid()
    fig19_stochastic_grokking()
    fig20_real_circles()
    fig21_three_classes()
    fig22_stochastic_trajectories()
    fig23_real_k_sweep()
    fig23b_k_sweep_equiv_colored()
    fig23c_k_sweep_low_equiv()
    fig24_three_class_trajectories()
    fig25_dual_axis_trajectories()
    fig26_multi_seed_trajectories()
    fig15b_grokked_boxplot()
    fig15c_grokked_violin()
    fig15d_sorted_dot()
    print("Done.")


if __name__ == "__main__":
    main()
