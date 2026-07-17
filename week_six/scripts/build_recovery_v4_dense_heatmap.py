#!/usr/bin/env python3
"""Trustworthiness heatmap for the dense Recovery-v4 sweep (clean_2M PickAndPlace).

Computes C1-C5 + weighted composite T per attack condition using the UNMODIFIED
evaluation/metrics.py pipeline (summarize_results + add_trustworthiness_scores),
then renders a conditions x {C1,C2,C3,C4,C5,T} heatmap on a fixed 0-1 scale.

C5 correctness: the dense CSV contains only sac_her_recovery_v4 (B4) rows, so the
B1 sac_her no-recovery baseline is loaded separately from data_recovery_v4/ (the
exact baseline the verified dense-sweep C5 numbers were computed against) and
forwarded via add_trustworthiness_scores(baseline_summary=...). This mirrors
analyze_recovery_v4_phase6.py; running build_benchmark_table.py directly on the
dense CSV would find no B1 rows and inflate C5.

metrics.py's formula (weights 0.10/0.20/0.25/0.15/0.30, C4=Safety) differs from the
paper's published Eq. 2 (C4=Adaptation, weights 0.10/0.25/0.25/0.05/0.35, Safety
separate) -- see findings.md reconciliation note. Read-only; edits no formula.

Output: paper_figures/recovery_v4_dense_trustworthiness_heatmap.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.metrics import summarize_results, add_trustworthiness_scores

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
OUT_PATH = os.path.join(OUT_DIR, "recovery_v4_dense_trustworthiness_heatmap.png")

# Row order = earlier bar chart grouping: clean first, then success descending,
# structural-floor conditions at the bottom.
ROW_ORDER = [
    "clean", "action_clipping", "action_delay", "grip_state_falsification",
    "object_pose_spoof", "goal_spoof_immediate", "goal_spoof_midep",
    "sensor_bias", "action_reversal", "contact_dropout", "sensor_dropout",
]
COLS = ["reliability_score", "robustness_score", "cyber_resilience_score",
        "safety_score", "recovery_score", "trustworthiness_score_weighted"]
COL_LABELS = ["C1\nReliability", "C2\nRobustness", "C3\nCyber\nResilience",
              "C4\nSafety", "C5\nRecovery", "T\n(weighted)"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def main():
    dense = pd.read_csv(DENSE_CSV)
    base = pd.read_csv(BASELINE_CSV)
    base = base[base["method"] == "sac_her"]

    b1_baseline = summarize_results(base)
    summ = add_trustworthiness_scores(summarize_results(dense),
                                      baseline_summary=b1_baseline)
    summ = summ.set_index("condition")

    # -------- report the full table first --------
    print("Dense Recovery-v4 sweep — clean_2M — C1-C5 + weighted T per condition")
    print("(metrics.py formula: weights 0.10/0.20/0.25/0.15/0.30, C4=Safety)\n")
    hdr = f"{'condition':26s}" + "".join(f"{c:>9s}" for c in
                                          ["C1", "C2", "C3", "C4", "C5", "T"])
    print(hdr)
    matrix = []
    for cond in ROW_ORDER:
        vals = [float(summ.loc[cond, c]) for c in COLS]
        matrix.append(vals)
        print(f"{cond:26s}" + "".join(f"{v:>9.3f}" for v in vals))
    matrix = np.array(matrix)
    print(f"\nmean weighted T over conditions = {matrix[:, -1].mean():.4f}")

    # -------- heatmap --------
    fig, ax = plt.subplots(figsize=(9.2, 8.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Sequential single-hue (light->dark blue), fixed 0-1 scale across all columns.
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(COL_LABELS)))
    ax.set_xticklabels(COL_LABELS, fontsize=9.5, color=INK_PRIMARY)
    ax.set_yticks(np.arange(len(ROW_ORDER)))
    ax.set_yticklabels([c.replace("_", " ") for c in ROW_ORDER],
                       fontsize=10, color=INK_PRIMARY)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()

    # Per-cell value labels; ink flips to white on dark cells for contrast.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else INK_PRIMARY)

    # Thin separator before the composite T column (it is not a C-score).
    ax.axvline(len(COLS) - 1 - 0.5, color=SURFACE, linewidth=3)

    ax.set_title(
        "Recovery v4 (dense CCAR) — clean_2M PickAndPlace\n"
        "Trustworthiness sub-scores C1–C5 and weighted composite T",
        fontsize=13, color=INK_PRIMARY, pad=34, loc="left",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("score (0–1)", fontsize=10, color=INK_SECONDARY)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)

    for s in ax.spines.values():
        s.set_visible(False)

    fig.text(
        0.008, 0.008,
        "C1–C5 per metrics.py (weights 0.10/0.20/0.25/0.15/0.30, C4=Safety); differs from "
        "the paper's published Eq. 2 (C4=Adaptation, weights 0.10/0.25/0.25/0.05/0.35, Safety "
        "reported separately) — see findings.md for reconciliation status.\n"
        "sac_her_recovery_v4, dense classifier, clean_2M. n = 5 seeds x 30 eps = 150/condition. "
        "C5 baseline = sac_her (data_recovery_v4).",
        fontsize=6.6, color=INK_SECONDARY,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
