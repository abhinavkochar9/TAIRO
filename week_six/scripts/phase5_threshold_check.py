"""
Phase 5 — SPOOFED_GOAL_PERCEIVED_MAX threshold calibration check.

Tests whether loosening SPOOFED_GOAL_PERCEIVED_MAX from 0.05m to 0.08m
introduces false positives on clean / action_clipping / action_delay episodes,
and whether the known goal_spoof_immediate edge case (episode 18) resolves.

Reads: results/data/diag_step_logs_phase2.csv  (33 episodes, do NOT regenerate)
Prints: label comparison table and FP analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import config

# ── Load diagnostic data ──────────────────────────────────────────────────────
DIAG_PATH = "results/data/diag_step_logs_phase2.csv"
diag = pd.read_csv(DIAG_PATH)

# Build an episode_idx→condition lookup (each episode has exactly one condition)
ep_meta = diag.groupby("episode_idx")[["condition", "seed"]].first().reset_index()

# ── Helper: run label_episode with an overridden SPOOFED_GOAL_PERCEIVED_MAX ──

from evaluation.failure_mode_labeling import (
    label_episode,
    LABEL_NEVER_REACHED, LABEL_REACH_NO_GRASP, LABEL_GRASPED_DROPPED,
    LABEL_SUCCESS, LABEL_SPOOFED_GOAL, LABEL_WRONG_DIRECTION,
    LABEL_ACTION_CORRUPTION,
    REACH_THRESHOLD, GRASP_DIST_THRESHOLD, GRASP_TRACKING_THRESHOLD,
    GRASP_WINDOW, GRASP_LIFT_THRESHOLD, GRASP_LIFT_WINDOW,
    DROP_SEPARATION_THRESHOLD, SPOOFED_GOAL_TRUE_MIN, WRONG_DIR_WINDOW,
    _detect_grasp, _detect_drop, _is_diverging,
)


def label_episode_with_threshold(step_df: pd.DataFrame,
                                  spoofed_goal_max: float) -> str:
    """Identical to label_episode() but with a custom SPOOFED_GOAL_PERCEIVED_MAX."""
    _REQUIRED_SPATIAL = [
        "distance_to_object", "distance_to_true_goal", "distance_to_perceived_goal",
        "object_velp_x", "object_velp_y", "object_velp_z",
        "grip_velp_x", "grip_velp_y", "grip_velp_z",
        "object_pos_z", "is_success",
    ]
    if any(
        col not in step_df.columns or step_df[col].isna().all()
        for col in _REQUIRED_SPATIAL
    ):
        return LABEL_ACTION_CORRUPTION

    dto  = step_df["distance_to_object"].values
    dttg = step_df["distance_to_true_goal"].values
    dtpg = step_df["distance_to_perceived_goal"].values

    if not np.any(dto < REACH_THRESHOLD):
        return LABEL_NEVER_REACHED

    grasp_step = _detect_grasp(step_df)
    if grasp_step is None:
        return LABEL_REACH_NO_GRASP

    if _detect_drop(step_df, grasp_step):
        return LABEL_GRASPED_DROPPED

    if step_df["is_success"].iloc[-1] > 0:
        return LABEL_SUCCESS

    goal_spoof_active = float(np.max(np.abs(dtpg - dttg))) > 0.01
    post_pg = dtpg[grasp_step:]
    post_tg = dttg[grasp_step:]
    if (goal_spoof_active and
            len(post_pg) > 0 and
            post_pg.min() < spoofed_goal_max and          # ← threshold under test
            post_tg.max() > SPOOFED_GOAL_TRUE_MIN):
        return LABEL_SPOOFED_GOAL

    if _is_diverging(dttg):
        return LABEL_WRONG_DIRECTION

    return LABEL_ACTION_CORRUPTION


# ── Run both thresholds on all 33 episodes ───────────────────────────────────

THRESH_OLD = 0.05
THRESH_NEW = 0.08

rows = []
for ep_id, ep_df in diag.groupby("episode_idx"):
    condition = ep_meta.loc[ep_meta["episode_idx"] == ep_id, "condition"].iloc[0]
    seed      = ep_meta.loc[ep_meta["episode_idx"] == ep_id, "seed"].iloc[0]
    label_05  = label_episode_with_threshold(ep_df, THRESH_OLD)
    label_08  = label_episode_with_threshold(ep_df, THRESH_NEW)

    # Also track min distance_to_perceived_goal post-grasp for diagnostics
    dto  = ep_df["distance_to_object"].values
    dtpg = ep_df["distance_to_perceived_goal"].values
    dttg = ep_df["distance_to_true_goal"].values
    from evaluation.failure_mode_labeling import _detect_grasp
    grasp_step = _detect_grasp(ep_df)
    min_dtpg_post_grasp = dtpg[grasp_step:].min() if grasp_step is not None else np.nan

    rows.append({
        "episode_idx": ep_id,
        "condition": condition,
        "seed": seed,
        "label_0.05": label_05,
        "label_0.08": label_08,
        "changed": label_05 != label_08,
        "min_dtpg_post_grasp": round(min_dtpg_post_grasp, 4) if not np.isnan(min_dtpg_post_grasp) else np.nan,
        "min_dttg_post_grasp": round(dttg[(grasp_step or 0):].max(), 4) if grasp_step is not None else np.nan,
    })

df = pd.DataFrame(rows)

# ── Report 1: Which episodes changed? ────────────────────────────────────────
print("=" * 70)
print("LABEL CHANGES: 0.05m → 0.08m SPOOFED_GOAL_PERCEIVED_MAX")
print("=" * 70)
changed = df[df["changed"]]
if changed.empty:
    print("No label changes detected.\n")
else:
    print(changed[["episode_idx","condition","seed","label_0.05","label_0.08",
                    "min_dtpg_post_grasp"]].to_string(index=False))
    print()

# ── Report 2: FP check on clean / action_clipping / action_delay ─────────────
print("=" * 70)
print("FALSE-POSITIVE CHECK: clean / action_clipping / action_delay episodes")
print("(these should never get label='spoofed_goal')")
print("=" * 70)
safe_conditions = ["clean", "action_clipping", "action_delay"]
safe_eps = df[df["condition"].isin(safe_conditions)]
print(safe_eps[["episode_idx","condition","seed","label_0.05","label_0.08",
                 "min_dtpg_post_grasp"]].to_string(index=False))
fps_08 = safe_eps[safe_eps["label_0.08"] == "spoofed_goal"]
print(f"\nFalse positives at 0.08m: {len(fps_08)} / {len(safe_eps)} safe-condition episodes")
print()

# ── Report 3: Goal-spoof episodes — both thresholds ──────────────────────────
print("=" * 70)
print("GOAL-SPOOF EPISODES: label comparison")
print("=" * 70)
spoof_eps = df[df["condition"].isin(["goal_spoof_immediate", "goal_spoof_midep"])]
print(spoof_eps[["episode_idx","condition","seed","label_0.05","label_0.08",
                  "min_dtpg_post_grasp","min_dttg_post_grasp"]].to_string(index=False))
print()

# ── Report 4: Full comparison table ──────────────────────────────────────────
print("=" * 70)
print("FULL LABEL TABLE (all 33 episodes)")
print("=" * 70)
print(df[["episode_idx","condition","seed","label_0.05","label_0.08","changed",
          "min_dtpg_post_grasp"]].to_string(index=False))
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY")
print("=" * 70)
n_changed_total = df["changed"].sum()
n_changed_safe  = fps_08.shape[0]
n_changed_spoof = (spoof_eps["label_0.05"] != spoof_eps["label_0.08"]).sum()
print(f"Total episodes: {len(df)}")
print(f"Changed labels: {n_changed_total}")
print(f"  - FPs introduced in clean/action_clipping/action_delay: {n_changed_safe}")
print(f"  - Changes in goal_spoof episodes: {n_changed_spoof}")
