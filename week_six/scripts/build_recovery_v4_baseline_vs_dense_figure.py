#!/usr/bin/env python3
"""Grouped comparison: no-recovery (B1/sac_her) success vs dense Recovery-v4 success.

All 11 attack conditions, clean_2M PickAndPlace. Both series are success rate in [0, 1],
so a single shared y-axis is used (no dual axis).

Baseline provenance (CONFIRMED post-seed-fix):
  no-recovery : results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv
                method=sac_her, collected 2026-07-15 (AFTER the seed-independence fix,
                commit 5812ff9 on 2026-07-14). This is the exact baseline the dense-sweep
                C5 numbers were computed against. NOTE: the paper's published table lists
                grip_state_falsification at 0.0% — that is a PRE-seed-fix value; post-fix it
                is 0.207. The discrepancy is the seed fix, not a computation bug. This chart
                deliberately uses the post-fix baseline, NOT the paper's stale table.
  dense v4    : results/data_recovery_v4_dense/episode_results_sac_her_pickandplace_clean_2M.csv
                (2026-07-15)

Output: paper_figures/recovery_v4_baseline_vs_dense_comparison.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # week_six/

BASELINE_CSV = os.path.join(
    ROOT, "results", "data_recovery_v4",
    "episode_results_sac_her_pickandplace_clean_2M.csv",
)
DENSE_CSV = os.path.join(
    ROOT, "results", "data_recovery_v4_dense",
    "episode_results_sac_her_pickandplace_clean_2M.csv",
)
OUT_DIR = os.path.join(ROOT, "paper_figures")
OUT_PATH = os.path.join(OUT_DIR, "recovery_v4_baseline_vs_dense_comparison.png")

# Reference-vs-highlight encoding: muted gray = no-recovery baseline, blue = v4.
COLOR_BASELINE = "#9a9a95"  # muted neutral (reference)
COLOR_V4 = "#2a78d6"        # validated categorical blue (highlight)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

# clean-first then by v4 success descending — matches the sibling dense-sweep figure.
ORDER = [
    "clean", "action_clipping", "action_delay", "grip_state_falsification",
    "object_pose_spoof", "goal_spoof_immediate", "goal_spoof_midep",
    "sensor_bias", "action_reversal", "contact_dropout", "sensor_dropout",
]


def _sr(csv_path, method=None):
    df = pd.read_csv(csv_path)
    if method is not None:
        df = df[df["method"] == method]
    return df.groupby("condition")["success"].mean()


def main():
    base = _sr(BASELINE_CSV, method="sac_her")
    dense = _sr(DENSE_CSV)

    labels = [c.replace("_", " ") for c in ORDER]
    base_vals = [float(base[c]) for c in ORDER]
    v4_vals = [float(dense[c]) for c in ORDER]

    x = np.arange(len(ORDER))
    w = 0.4

    fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    b1 = ax.bar(x - w / 2, base_vals, w, label="No recovery (sac_her, B1)",
                color=COLOR_BASELINE, zorder=3)
    b2 = ax.bar(x + w / 2, v4_vals, w, label="Recovery v4 (dense CCAR)",
                color=COLOR_V4, zorder=3)

    def _fmt(v):
        if v == 0.0:
            return "0"
        if v == 1.0:
            return "1.00"
        return f"{v:.3f}".rstrip("0").rstrip(".")

    for bars, vals in ((b1, base_vals), (b2, v4_vals)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                    _fmt(v), ha="center", va="bottom", fontsize=7.5,
                    color=INK_SECONDARY)

    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Success rate  (0–1)", fontsize=11, color=INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=9.5,
                       color=INK_PRIMARY)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)

    ax.set_title(
        "No-recovery baseline vs Recovery v4 (dense CCAR) — clean_2M PickAndPlace\n"
        "Task success rate by attack condition",
        fontsize=13, color=INK_PRIMARY, pad=14, loc="left",
    )

    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfcec9")

    ax.legend(frameon=False, fontsize=10, loc="upper right",
              labelcolor=INK_PRIMARY)

    fig.text(
        0.008, 0.006,
        "Baseline: sac_her (no recovery), results/data_recovery_v4/"
        "episode_results_sac_her_pickandplace_clean_2M.csv — POST seed-independence fix "
        "(commit 5812ff9, 2026-07-14); data collected 2026-07-15. "
        "v4: results/data_recovery_v4_dense/ (2026-07-15). n = 5 seeds x 30 eps = 150 / "
        "condition. NB: post-fix grip_state_falsification baseline = 0.207 (paper's 0.0% is "
        "a pre-fix value).",
        fontsize=6.5, color=INK_SECONDARY,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")

    print(f"{'condition':26s} {'baseline':>9s} {'v4':>7s} {'delta':>8s}")
    for c in ORDER:
        print(f"{c:26s} {float(base[c]):9.3f} {float(dense[c]):7.3f} "
              f"{float(dense[c]) - float(base[c]):+8.3f}")


if __name__ == "__main__":
    main()
