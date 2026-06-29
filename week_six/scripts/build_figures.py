"""
Publication-ready figures for the TAIRO paper.

Reads:  results/data/episode_results.csv
        results/data/summary.csv
Writes: results/figures/fig1_success_by_condition.png
        results/figures/fig2_b0_b3_layers.png
        results/figures/fig3_recovery_comparison.png
        results/figures/fig4_trustworthiness_scores.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import DATA_DIR, FIGURES_DIR, ALL_CONDITIONS, BENCHMARK_LAYERS

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

METHOD_COLORS = {
    "sac_her":              "#F57C00",
    "sac_her_recovery_v2":  "#1565C0",
    "sac_her_recovery_v3":  "#1B5E20",
}

METHOD_LABELS = {
    "sac_her":              "SAC+HER",
    "sac_her_recovery_v2":  "SAC+HER+Recovery v2",
    "sac_her_recovery_v3":  "SAC+HER+Recovery v3",
}

CONDITION_LABELS = {
    "clean":                "Clean",
    "sensor_dropout":       "Sensor\nDropout",
    "sensor_bias":          "Sensor\nBias",
    "action_clipping":      "Action\nClipping",
    "action_delay":         "Action\nDelay",
    "action_reversal":      "Action\nReversal",
    "goal_spoof_immediate": "Goal Spoof\n(imm.)",
    "goal_spoof_midep":     "Goal Spoof\n(mid-ep)",
}

LAYER_COLORS  = ["#BBDEFB", "#64B5F6", "#1976D2", "#0D47A1"]
LAYER_LABELS  = {
    "B0": "B0 — Clean baseline",
    "B1": "B1 — No recovery",
    "B2": "B2 — Recovery v2",
    "B3": "B3 — Recovery v3",
}


def _savefig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] → {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Success rate per condition: rule_based vs sac_her (B1, no recovery)
# ---------------------------------------------------------------------------

def fig1_success_by_condition(ep_df: pd.DataFrame) -> None:
    b1 = ep_df[ep_df["benchmark_layer"].isin(["B0", "B1"])]
    conditions = ALL_CONDITIONS
    base_methods = ["sac_her"]

    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, method in enumerate(base_methods):
        rates = [
            b1[(b1["method"] == method) & (b1["condition"] == c)]["success"].mean()
            for c in conditions
        ]
        ax.bar(
            x + i * width, rates, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#999"),
            alpha=0.85, edgecolor="white",
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success Rate")
    ax.set_title(
        "Fig 1 — SAC+HER Baseline Vulnerability per Attack Condition\n"
        "(B1: no recovery, 5 seeds × 30 episodes each)",
        fontsize=11,
    )
    ax.legend(loc="upper right")

    _savefig(fig, "fig1_success_by_condition.png")


# ---------------------------------------------------------------------------
# Figure 2 — B0–B3 layer comparison for SAC+HER across conditions
# ---------------------------------------------------------------------------

def fig2_b0_b3_layers(ep_df: pd.DataFrame) -> None:
    conditions = ALL_CONDITIONS
    x = np.arange(len(conditions))
    n = len(BENCHMARK_LAYERS)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, layer in enumerate(BENCHMARK_LAYERS):
        layer_df = ep_df[
            (ep_df["benchmark_layer"] == layer)
            & (ep_df["method"].isin(["sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3"]))
        ]
        rates = [
            layer_df[layer_df["condition"] == c]["success"].mean()
            for c in conditions
        ]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(
            x + offset, rates, width,
            label=LAYER_LABELS[layer],
            color=LAYER_COLORS[i],
            alpha=0.9, edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success Rate (SAC+HER)")
    ax.set_title(
        "Fig 2 — B0–B3 Benchmark Layers: SAC+HER Success Rate per Condition",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)

    _savefig(fig, "fig2_b0_b3_layers.png")


# ---------------------------------------------------------------------------
# Figure 3 — Recovery comparison: none vs v2 vs v3 (SAC+HER only, attacked)
# ---------------------------------------------------------------------------

def fig3_recovery_comparison(ep_df: pd.DataFrame) -> None:
    sac_df = ep_df[ep_df["method"].isin(["sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3"])]
    # Exclude clean — there is nothing to recover from.
    conditions = [c for c in ALL_CONDITIONS if c != "clean"]
    x = np.arange(len(conditions))

    configs = [
        ("sac_her",             "No Recovery (B1)",  "#F57C00", "o--"),
        ("sac_her_recovery_v2", "Recovery v2 (B2)", "#1565C0", "s-"),
        ("sac_her_recovery_v3", "Recovery v3 (B3)", "#1B5E20", "^-"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method, label, color, fmt in configs:
        rates = [
            sac_df[(sac_df["method"] == method) & (sac_df["condition"] == c)][
                "success"
            ].mean()
            for c in conditions
        ]
        marker, ls = fmt[0], fmt[1:]
        ax.plot(x, rates, ls, marker=marker, label=label,
                color=color, linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel("Success Rate (SAC+HER)")
    ax.set_title(
        "Fig 3 — Recovery Comparison: No Recovery vs v2 vs v3 (SAC+HER under attack)",
        fontsize=11,
    )
    ax.legend(loc="upper right")

    _savefig(fig, "fig3_recovery_comparison.png")


# ---------------------------------------------------------------------------
# Figure 4 — Weighted trustworthiness scores: B1 vs B3 comparison
# ---------------------------------------------------------------------------

def fig4_trustworthiness_scores(summary_df: pd.DataFrame) -> None:
    layers  = ["B1", "B3"]
    methods = ["sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    for ax, layer in zip(axes, layers):
        layer_df = summary_df[summary_df["benchmark_layer"] == layer]
        y_pos = np.arange(len(methods))
        scores = []
        for method in methods:
            m_rows = layer_df[layer_df["method"] == method]
            score = m_rows["trustworthiness_score_weighted"].mean() if not m_rows.empty else 0.0
            scores.append(score)

        bars = ax.barh(
            y_pos, scores,
            color=[METHOD_COLORS.get(m, "#999") for m in methods],
            alpha=0.85, edgecolor="white",
        )
        for bar, score in zip(bars, scores):
            ax.text(
                score + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Weighted Trustworthiness Score")
        ax.set_title(LAYER_LABELS.get(layer, layer), fontsize=10)

    fig.suptitle(
        "Fig 4 — TAIRO Weighted Trustworthiness Scores (C1–C5, argued weights)\n"
        "B1 (no recovery) vs B3 (recovery v3)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    _savefig(fig, "fig4_trustworthiness_scores.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ep_path  = os.path.join(DATA_DIR, "episode_results.csv")
    sum_path = os.path.join(DATA_DIR, "summary.csv")

    print(f"[figures] Reading {ep_path}")
    ep_df = pd.read_csv(ep_path)

    print(f"[figures] Reading {sum_path}")
    sum_df = pd.read_csv(sum_path)

    fig1_success_by_condition(ep_df)
    fig2_b0_b3_layers(ep_df)
    fig3_recovery_comparison(ep_df)
    fig4_trustworthiness_scores(sum_df)

    print(f"[figures] All 4 figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
