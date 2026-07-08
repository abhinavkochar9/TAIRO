"""
Phase 7 spot-check — annotated plots for condition/label combos new to Phase 7.

Targets combos NOT present in the Phase 3 diagnostic (seed=0, clean_2M, 3 eps each):
  - grasped_but_dropped     × clean              (randomized_2M)
  - grasped_but_dropped     × goal_spoof_midep   (randomized_2M)
  - action_control_corruption × object_pose_spoof (clean_2M)
  - wrong_direction           × object_pose_spoof (clean_2M)
  - reached_but_failed_grasp  × sensor_bias       (clean_2M)
  - never_reached_object      × grip_state_falsification (clean_2M — the 60 that didn't reach)
  - reached_but_failed_grasp  × action_clipping   (randomized_2M)
  - reached_but_failed_grasp  × action_reversal   (randomized_2M)

3 episodes per combo → 24 total spot-check plots.

Output: results/figures/diag_phase7/<combo>.png
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
    _detect_grasp,
    LABEL_SUCCESS, LABEL_NEVER_REACHED, LABEL_REACH_NO_GRASP,
    LABEL_GRASPED_DROPPED, LABEL_SPOOFED_GOAL,
    LABEL_WRONG_DIRECTION, LABEL_ACTION_CORRUPTION,
)

OUT_DIR = os.path.join(FIGURES_DIR, "diag_phase7")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_COLORS = {
    LABEL_SUCCESS:           "#2ecc71",
    LABEL_NEVER_REACHED:     "#e74c3c",
    LABEL_REACH_NO_GRASP:    "#e67e22",
    LABEL_GRASPED_DROPPED:   "#9b59b6",
    LABEL_SPOOFED_GOAL:      "#3498db",
    LABEL_WRONG_DIRECTION:   "#f39c12",
    LABEL_ACTION_CORRUPTION: "#95a5a6",
}

# ── Combos to spot-check: (model, condition, label, n_episodes) ──────────────
TARGETS = [
    ("clean_2M",      "grip_state_falsification", LABEL_NEVER_REACHED,     3),
    ("clean_2M",      "sensor_bias",              LABEL_REACH_NO_GRASP,    3),
    ("clean_2M",      "object_pose_spoof",        LABEL_WRONG_DIRECTION,   3),
    ("clean_2M",      "object_pose_spoof",        LABEL_ACTION_CORRUPTION, 3),
    ("randomized_2M", "clean",                    LABEL_GRASPED_DROPPED,   3),
    ("randomized_2M", "goal_spoof_midep",         LABEL_GRASPED_DROPPED,   3),
    ("randomized_2M", "action_clipping",          LABEL_REACH_NO_GRASP,    3),
    ("randomized_2M", "action_reversal",          LABEL_REACH_NO_GRASP,    3),
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("[phase7 spotcheck] Loading step logs and labels ...")
step_cache  = {}
label_cache = {}
for model in {t[0] for t in TARGETS}:
    step_cache[model]  = pd.read_csv(
        os.path.join(DATA_DIR, f"step_logs_sac_her_pickandplace_{model}.csv"))
    label_cache[model] = pd.read_csv(
        os.path.join(DATA_DIR, f"labels_sac_her_pickandplace_{model}.csv"))


def _plot_episodes(ep_ids, step_df, label_df, title_prefix, out_path):
    """3-panel (dist_to_object / dist_to_true_goal / dist_to_perceived_goal)
    side-by-side for each episode in ep_ids."""
    n = len(ep_ids)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 9), squeeze=False)
    fig.suptitle(title_prefix, fontsize=11, y=1.01)

    for col, ep_id in enumerate(ep_ids):
        ep_data  = step_df[step_df["episode_idx"] == ep_id].sort_values("timestep")
        label    = label_df.loc[label_df["episode_idx"] == ep_id, "failure_mode"].iloc[0]
        color    = LABEL_COLORS.get(label, "gray")
        grasp_st = _detect_grasp(ep_data)

        t    = ep_data["timestep"].values
        dto  = ep_data["distance_to_object"].values
        dttg = ep_data["distance_to_true_goal"].values
        dtpg = ep_data["distance_to_perceived_goal"].values
        succ_steps = ep_data[ep_data["is_success"] == 1.0]["timestep"].values

        cond = ep_data["condition"].iloc[0]
        seed = ep_data["seed"].iloc[0]
        ax_title = f"ep {ep_id} s={seed} [{label}]"

        for row, (y, ylabel, panel) in enumerate([
            (dto,  "dist_to_object (m)",        "Gripper→Object"),
            (dttg, "dist_to_true_goal (m)",     "Object→True Goal"),
            (dtpg, "dist_to_perceived_goal (m)","Object→Perceived Goal"),
        ]):
            ax = axes[row][col]
            ax.plot(t, y, color="steelblue", linewidth=1.2)

            if row == 2:
                ax.plot(t, dttg, color="forestgreen", linestyle="--",
                        alpha=0.5, linewidth=1.0, label="to true goal")

            if len(succ_steps) > 0:
                ax.axvline(succ_steps[0], color="gold", linewidth=1.5,
                           linestyle=":", alpha=0.9, label="first success")

            if grasp_st is not None:
                ax.axvline(grasp_st, color=color, linewidth=2.0,
                           linestyle="--", alpha=0.8, label="grasp onset")

            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.25)

            handles, lbs = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)

            if row == 0:
                ax.set_title(ax_title, fontsize=8, color=color, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Timestep", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Generate plots ────────────────────────────────────────────────────────────
np.random.seed(42)

for model, condition, label, n_eps in TARGETS:
    step_df  = step_cache[model]
    label_df = label_cache[model]

    # Find episodes matching this combo
    matching = label_df[
        (label_df["condition"] == condition) &
        (label_df["failure_mode"] == label)
    ]["episode_idx"].values

    if len(matching) == 0:
        print(f"[skip] {model} / {condition} / {label} — no episodes")
        continue

    chosen = np.random.choice(matching, size=min(n_eps, len(matching)), replace=False)
    chosen = sorted(chosen)

    title = f"Phase 7 spot-check | model={model} | cond={condition} | label={label}"
    slug  = f"{model}__{condition}__{label}".replace(" ", "_")
    out   = os.path.join(OUT_DIR, f"p7_{slug}.png")
    _plot_episodes(chosen, step_df, label_df, title, out)

print(f"\n[phase7 spotcheck] Done — {len(TARGETS)} figures written to {OUT_DIR}")
