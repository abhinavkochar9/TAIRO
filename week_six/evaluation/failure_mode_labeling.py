"""
Rule-based failure-mode labeling for TAIRO PickAndPlace episodes.

Entry point
-----------
    label_episode(step_df) -> str

Returns one of six labels based on trajectory behavior alone — the
`condition` and `attack_level` columns are never used as features.

Labels
------
    success                  — is_success became 1 at any timestep
    never_reached_object     — gripper never came within REACH_THRESHOLD of object
    reached_but_failed_grasp — gripper reached but no confirmed grasp (kinematic + lift)
    grasped_but_dropped      — confirmed grasp, then object separated before any success
    spoofed_goal             — object converged to perceived goal, not true goal
    divergent_transport      — object trajectory diverged from true goal over episode,
                                OR unclassified fallback (elevated action-norm / safety
                                signal). Merged 2026-07-11 from the former
                                `wrong_direction` + `action_control_corruption` split —
                                see CLAUDE.md §14 changelog for why.

Threshold constants live in config.py (never hardcoded here).

Requires step_df to contain the new spatial columns written by episode_runner.py:
    distance_to_object, distance_to_true_goal, distance_to_perceived_goal,
    object_velp_{x,y,z}, grip_velp_{x,y,z}, object_pos_z, is_success,
    action_norm, intended_action_norm, safety_violation.

FetchReach episodes (10-dim obs → NaN spatial fields) are returned as
divergent_transport immediately — they are not in scope for this labeler.
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    REACH_THRESHOLD,
    GRASP_DIST_THRESHOLD,
    GRASP_TRACKING_THRESHOLD,
    GRASP_WINDOW,
    GRASP_LIFT_THRESHOLD,
    GRASP_LIFT_WINDOW,
    DROP_SEPARATION_THRESHOLD,
    SPOOFED_GOAL_PERCEIVED_MAX,
    SPOOFED_GOAL_TRUE_MIN,
    WRONG_DIR_WINDOW,
)

# Canonical label strings — import these instead of hardcoding in callers.
LABEL_SUCCESS              = "success"
LABEL_NEVER_REACHED        = "never_reached_object"
LABEL_REACH_NO_GRASP       = "reached_but_failed_grasp"
LABEL_GRASPED_DROPPED      = "grasped_but_dropped"
LABEL_SPOOFED_GOAL         = "spoofed_goal"
# Merged 2026-07-11: former LABEL_WRONG_DIRECTION ("wrong_direction") and
# LABEL_ACTION_CORRUPTION ("action_control_corruption") collapsed into one
# label. Diagnostic found the two were not cleanly separable (F1 0.452/0.100
# on test seed 4, collapsing further under held-out-condition generalization)
# and that `action_control_corruption` never actually reflected action-space
# corruption in this data (tail_action_div identically zero across all 128
# episodes carrying either label). See CLAUDE.md §14 changelog.
LABEL_DIVERGENT_TRANSPORT  = "divergent_transport"

ALL_LABELS = [
    LABEL_SUCCESS,
    LABEL_NEVER_REACHED,
    LABEL_REACH_NO_GRASP,
    LABEL_GRASPED_DROPPED,
    LABEL_SPOOFED_GOAL,
    LABEL_DIVERGENT_TRANSPORT,
]

_REQUIRED_SPATIAL = [
    "distance_to_object",
    "distance_to_true_goal",
    "distance_to_perceived_goal",
    "object_velp_x", "object_velp_y", "object_velp_z",
    "grip_velp_x",   "grip_velp_y",   "grip_velp_z",
    "object_pos_z",
    "is_success",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_grasp(step_df: pd.DataFrame) -> Optional[int]:
    """Return the step index of confirmed grasp onset, or None.

    A grasp is confirmed when two kinematic conditions hold for GRASP_WINDOW
    consecutive steps AND the object subsequently rises by GRASP_LIFT_THRESHOLD
    within GRASP_LIFT_WINDOW steps — distinguishing a real 3-D lift from
    the object being dragged along the table surface.

    Kinematic criteria (per step):
      (1) distance_to_object < GRASP_DIST_THRESHOLD  (physical contact)
      (2) ||object_velp - grip_velp|| < GRASP_TRACKING_THRESHOLD  (co-moving)

    Lift confirmation (post-window):
      (3) object_pos_z rises >= GRASP_LIFT_THRESHOLD within the next
          GRASP_LIFT_WINDOW steps after the window onset.
    """
    dto   = step_df["distance_to_object"].values
    obj_v = step_df[["object_velp_x", "object_velp_y", "object_velp_z"]].values
    grp_v = step_df[["grip_velp_x",   "grip_velp_y",   "grip_velp_z"]].values
    obj_z = step_df["object_pos_z"].values
    n     = len(step_df)

    consecutive = 0
    for i in range(n):
        in_contact = dto[i] < GRASP_DIST_THRESHOLD
        tracking   = (in_contact and
                      np.linalg.norm(obj_v[i] - grp_v[i]) < GRASP_TRACKING_THRESHOLD)
        if in_contact and tracking:
            consecutive += 1
            if consecutive >= GRASP_WINDOW:
                onset    = i - GRASP_WINDOW + 1
                lift_end = min(i + GRASP_LIFT_WINDOW, n)
                if obj_z[onset:lift_end].max() - obj_z[onset] >= GRASP_LIFT_THRESHOLD:
                    return onset
                # Kinematic window met but lift not yet seen; keep scanning.
        else:
            consecutive = 0

    return None


def _detect_drop(step_df: pd.DataFrame, grasp_step: int) -> bool:
    """True if distance_to_object exceeds DROP_SEPARATION_THRESHOLD after grasp
    and *before* any is_success flag — a placement (post-success gripper
    release) does not count as a drop.
    """
    is_succ   = step_df["is_success"].values
    dto       = step_df["distance_to_object"].values
    succ_idx  = np.where(is_succ > 0)[0]
    end_step  = int(succ_idx[0]) if len(succ_idx) > 0 else len(dto)
    post      = dto[grasp_step:end_step]
    return bool(len(post) > 0 and np.any(post > DROP_SEPARATION_THRESHOLD))


def _is_diverging(dttg: np.ndarray) -> bool:
    """True if distance_to_true_goal trend is non-decreasing over the last
    WRONG_DIR_WINDOW steps (positive or zero slope → object not converging).
    """
    segment = dttg[-min(WRONG_DIR_WINDOW, len(dttg)):]
    if len(segment) < 10:
        return False
    slope = float(np.polyfit(np.arange(len(segment)), segment, 1)[0])
    return slope >= 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_episode(step_df: pd.DataFrame) -> str:
    """Label a single episode by its failure mode.

    Args:
        step_df: Per-step log DataFrame for one episode (produced by
                 run_episode + the Phase 1 logging patch).  Must contain
                 the spatial columns listed in _REQUIRED_SPATIAL.

    Returns:
        One of the label strings in ALL_LABELS.  Never raises.
    """
    # Guard: spatial columns absent or all-NaN (e.g. FetchReach episodes)
    if any(
        col not in step_df.columns or step_df[col].isna().all()
        for col in _REQUIRED_SPATIAL
    ):
        return LABEL_DIVERGENT_TRANSPORT

    dto  = step_df["distance_to_object"].values
    dttg = step_df["distance_to_true_goal"].values
    dtpg = step_df["distance_to_perceived_goal"].values

    # 1. Never reached the object
    if not np.any(dto < REACH_THRESHOLD):
        return LABEL_NEVER_REACHED

    # 2. Reached but failed to establish a confirmed grasp
    grasp_step = _detect_grasp(step_df)
    if grasp_step is None:
        return LABEL_REACH_NO_GRASP

    # 3. Grasped but object separated before any success
    if _detect_drop(step_df, grasp_step):
        return LABEL_GRASPED_DROPPED

    # 4. Success — use the FINAL-step flag, not max(), because is_success is
    #    non-sticky: it returns to 0 if the object is moved away after initial
    #    placement (e.g. goal_spoof_midep places the object at the true goal
    #    transiently at step 12, then the mid-episode spoof causes the policy
    #    to move it away).  Checking max() would misclassify those episodes.
    if step_df["is_success"].iloc[-1] > 0:
        return LABEL_SUCCESS

    # 5. Spoofed goal: object converged near perceived goal while staying
    #    far from true goal — policy chased the wrong target.
    #
    #    Guard: only fire when perceived_goal is actually different from
    #    true_goal (i.e. a goal-spoof attack is active).  For object_pose_spoof
    #    the goal is NOT shifted (only the observed object position is), so
    #    dtpg == dttg everywhere — the check would fire spuriously because
    #    "converged to perceived" == "converged to true" in that case.
    goal_spoof_active = float(np.max(np.abs(dtpg - dttg))) > 0.01
    post_pg = dtpg[grasp_step:]
    post_tg = dttg[grasp_step:]
    if (goal_spoof_active and
            len(post_pg) > 0 and
            post_pg.min() < SPOOFED_GOAL_PERCEIVED_MAX and
            post_tg.max() > SPOOFED_GOAL_TRUE_MIN):
        return LABEL_SPOOFED_GOAL

    # 6. Divergent transport: everything remaining — true-goal distance
    #    trending away/flat over the episode tail, or any other unclassified
    #    failure (includes object_pose_spoof, where perceived == true goal so
    #    spoofed_goal cannot fire). Formerly split into wrong_direction (the
    #    `_is_diverging` case) and action_control_corruption (fallback);
    #    merged 2026-07-11 — see module docstring. `_is_diverging` is kept
    #    only as a helper for historical diagnostic scripts, not called here
    #    since both former branches now return the same label.
    return LABEL_DIVERGENT_TRANSPORT


def label_batch(step_df_all: pd.DataFrame,
                episode_col: str = "episode_idx") -> pd.DataFrame:
    """Apply label_episode to every episode in a concatenated step log.

    Args:
        step_df_all: Concatenated step-log DataFrame (all episodes).
        episode_col: Column that identifies individual episodes.

    Returns:
        DataFrame with columns [episode_col, "failure_mode"] — one row
        per episode, suitable for merging with episode_results.
    """
    rows = []
    for ep_id, ep_df in step_df_all.groupby(episode_col):
        rows.append({episode_col: ep_id, "failure_mode": label_episode(ep_df)})
    return pd.DataFrame(rows)
