#!/usr/bin/env python3
"""Build the dense Recovery-v4 sweep figure (Phase C).

Grouped bar chart, all 11 attack conditions, two series:
  - success rate         (dense classifier-conditioned v4, clean_2M PickAndPlace)
  - C5 recovery_score    (max(0, dense_v4_success - sac_her_baseline_success); clean = 1.0)

Both series are quantities in [0, 1], so a single shared y-axis is used (no dual axis).

Sources (recovery-v4-tier1 worktree):
  dense v4 :  results/data_recovery_v4_dense/episode_results_sac_her_pickandplace_clean_2M.csv
  baseline :  results/data_recovery_v4/episode_results_sac_her_pickandplace_clean_2M.csv (method=sac_her)

The baseline source is data_recovery_v4/ (NOT data_seedfix/): it is the sac_her run the
verified dense-sweep C5 numbers were computed against, and reproduces all 11 verified
(success, C5) pairs exactly. Output: paper_figures/recovery_v4_dense_sweep_by_condition.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # week_six/

DENSE_CSV = os.path.join(
    ROOT, "results", "data_recovery_v4_dense",
    "episode_results_sac_her_pickandplace_clean_2M.csv",
)
BASELINE_CSV = os.path.join(
    ROOT, "results", "data_recovery_v4",
    "episode_results_sac_her_pickandplace_clean_2M.csv",
)
OUT_DIR = os.path.join(ROOT, "paper_figures")
OUT_PATH = os.path.join(OUT_DIR, "recovery_v4_dense_sweep_by_condition.png")

# Validated dataviz palette (categorical slots 1 & 3), light surface.
COLOR_SUCCESS = "#2a78d6"  # blue
COLOR_C5 = "#eda100"       # yellow
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def _success_by_condition(csv_path, method=None):
    df = pd.read_csv(csv_path)
    if method is not None:
        df = df[df["method"] == method]
    return df.groupby("condition")["success"].mean()


def main():
    dense = _success_by_condition(DENSE_CSV)
    base = _success_by_condition(BASELINE_CSV, method="sac_her")

    conditions = list(dense.index)
    success = {c: float(dense[c]) for c in conditions}
    c5 = {}
    for c in conditions:
        c5[c] = 1.0 if c == "clean" else max(0.0, success[c] - float(base[c]))

    # Order by success rate descending, clean pinned first (canonical baseline).
    ordered = ["clean"] + sorted(
        [c for c in conditions if c != "clean"],
        key=lambda c: (-success[c], c),
    )

    labels = [c.replace("_", " ") for c in ordered]
    sr_vals = [success[c] for c in ordered]
    c5_vals = [c5[c] for c in ordered]

    x = np.arange(len(ordered))
    w = 0.4

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    b1 = ax.bar(x - w / 2, sr_vals, w, label="Success rate",
                color=COLOR_SUCCESS, zorder=3)
    b2 = ax.bar(x + w / 2, c5_vals, w, label="C5 recovery score",
                color=COLOR_C5, zorder=3)

    def _fmt(v):
        if v == 0.0:
            return "0"
        if v == 1.0:
            return "1.00"
        return f"{v:.3f}".rstrip("0").rstrip(".")

    for bars, vals in ((b1, sr_vals), (b2, c5_vals)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.012, _fmt(v),
                    ha="center", va="bottom", fontsize=7.5,
                    color=INK_SECONDARY)

    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Rate  (0–1)", fontsize=11, color=INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=9.5,
                       color=INK_PRIMARY)
    ax.tick_params(axis="y", colors=INK_SECONDARY, labelsize=9)

    ax.set_title(
        "Recovery v4 (classifier-conditioned adaptive recovery) — clean_2M "
        "PickAndPlace\nSuccess rate and C5 recovery score by attack condition",
        fontsize=13, color=INK_PRIMARY, pad=14, loc="left",
    )

    # Recessive grid / spines.
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
        "n = 5 seeds x 30 episodes = 150 per condition.  "
        "C5 = max(0, v4 success - sac_her baseline success); clean fixed at 1.0.  "
        "Dense classifier-conditioned v4; baseline = sac_her (data_recovery_v4).",
        fontsize=7, color=INK_SECONDARY,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")

    # Echo the plotted numbers for the record.
    print(f"{'condition':26s} {'success':>8s} {'C5':>7s}")
    for c in ordered:
        print(f"{c:26s} {success[c]:8.3f} {c5[c]:7.3f}")


if __name__ == "__main__":
    main()
