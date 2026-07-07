"""
Phase 3 diagnostic: apply failure-mode labeling to the Phase 2 step logs
and produce annotated distance-over-time plots.

Reads:
    results/data/diag_step_logs_phase2.csv   (produced by diagnostic_phase2.py)

Writes:
    results/figures/diag_phase3/diag_phase3_<condition>.png  (one per condition)
    results/data/diag_labels_phase3.csv                      (one row per episode)

Does NOT re-run the environment or touch any sweep data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, FIGURES_DIR
from evaluation.failure_mode_labeling import (
    label_episode, label_batch, _detect_grasp,
    LABEL_SUCCESS, LABEL_NEVER_REACHED, LABEL_REACH_NO_GRASP,
    LABEL_GRASPED_DROPPED, LABEL_SPOOFED_GOAL,
    LABEL_WRONG_DIRECTION, LABEL_ACTION_CORRUPTION,
)

STEP_LOG_PATH = os.path.join(DATA_DIR, "diag_step_logs_phase2.csv")
LABEL_OUT     = os.path.join(DATA_DIR, "diag_labels_phase3.csv")
OUT_DIR       = os.path.join(FIGURES_DIR, "diag_phase3")

LABEL_COLORS = {
    LABEL_SUCCESS:           "#2ecc71",
    LABEL_NEVER_REACHED:     "#e74c3c",
    LABEL_REACH_NO_GRASP:    "#e67e22",
    LABEL_GRASPED_DROPPED:   "#9b59b6",
    LABEL_SPOOFED_GOAL:      "#3498db",
    LABEL_WRONG_DIRECTION:   "#f39c12",
    LABEL_ACTION_CORRUPTION: "#95a5a6",
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(STEP_LOG_PATH)
    print(f"[phase3] Loaded {len(df)} step rows, "
          f"{df['episode_idx'].nunique()} episodes, "
          f"{df['condition'].nunique()} conditions.")

    # ------------------------------------------------------------------
    # Apply labels
    # ------------------------------------------------------------------
    labels_df = label_batch(df)
    # Attach condition for reporting
    ep_meta = df[["episode_idx", "condition", "seed", "is_success"]].groupby("episode_idx").agg(
        condition=("condition", "first"),
        seed=("seed", "first"),
        any_success=("is_success", "max"),
    ).reset_index()
    labels_df = labels_df.merge(ep_meta, on="episode_idx")
    labels_df.to_csv(LABEL_OUT, index=False)
    print(f"[phase3] Labels saved → {LABEL_OUT}\n")

    # ------------------------------------------------------------------
    # Print label summary per condition
    # ------------------------------------------------------------------
    print("[phase3] Label assignments per condition:")
    for cond, grp in labels_df.groupby("condition"):
        counts = grp["failure_mode"].value_counts().to_dict()
        print(f"  {cond:<32} {counts}")
    print()

    # ------------------------------------------------------------------
    # Plots — one figure per condition, all 3 episodes side-by-side
    # ------------------------------------------------------------------
    conditions = df["condition"].unique()
    for cond in sorted(conditions):
        cond_eps = labels_df[labels_df["condition"] == cond].sort_values("episode_idx")
        ep_ids   = cond_eps["episode_idx"].tolist()

        fig, axes = plt.subplots(3, len(ep_ids), figsize=(5 * len(ep_ids), 9), squeeze=False)
        fig.suptitle(f"Phase 3 — condition: {cond}", fontsize=12, y=1.01)

        for col_idx, ep_id in enumerate(ep_ids):
            ep_data  = df[df["episode_idx"] == ep_id].sort_values("timestep")
            label    = cond_eps.loc[cond_eps["episode_idx"] == ep_id, "failure_mode"].iloc[0]
            color    = LABEL_COLORS.get(label, "gray")
            grasp_st = _detect_grasp(ep_data)

            t    = ep_data["timestep"].values
            dto  = ep_data["distance_to_object"].values
            dttg = ep_data["distance_to_true_goal"].values
            dtpg = ep_data["distance_to_perceived_goal"].values

            succ_steps = ep_data[ep_data["is_success"] == 1.0]["timestep"].values

            ax_title = f"ep {ep_id} — [{label}]"

            for row_idx, (y, ylabel, title_suffix) in enumerate([
                (dto,  "dist_to_object (m)",       "Gripper→Object"),
                (dttg, "dist_to_true_goal (m)",    "Object→True Goal"),
                (dtpg, "dist_to_perceived_goal (m)","Object→Perceived Goal"),
            ]):
                ax = axes[row_idx][col_idx]
                ax.plot(t, y, color="steelblue", linewidth=1.2)

                # Overlay true goal on perceived-goal panel for easy comparison
                if row_idx == 2:
                    ax.plot(t, dttg, color="forestgreen", linestyle="--",
                            alpha=0.5, linewidth=1.0, label="to true goal")

                # Mark first success (gold dotted vertical)
                if len(succ_steps) > 0:
                    ax.axvline(succ_steps[0], color="gold", linewidth=1.5,
                               linestyle=":", alpha=0.9, label="first success")

                # Mark confirmed grasp onset (label-colored dashed vertical)
                if grasp_st is not None:
                    ax.axvline(grasp_st, color=color, linewidth=2.0,
                               linestyle="--", alpha=0.8, label="grasp onset")

                ax.set_ylabel(ylabel, fontsize=8)
                ax.grid(True, alpha=0.25)

                # Render legend after all artists are added so every marker appears
                handles, lbs = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7)

                if row_idx == 0:
                    ax.set_title(ax_title, fontsize=8, color=color, fontweight="bold")
                if row_idx == 2:
                    ax.set_xlabel("Timestep", fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"diag_phase3_{cond}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {out_path}")

    print("\n[phase3] Done.")


if __name__ == "__main__":
    main()
