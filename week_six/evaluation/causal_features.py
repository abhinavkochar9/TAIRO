"""
Phase 9 — causal/online feature extraction for the online failure-mode
classifier (`results/classifier/online_failure_classifier.pkl`, not yet
trained as of this module's creation).

Distinct from `scripts/train_failure_classifier.py` (Phase 8, post-hoc,
full-episode aggregate features). This module produces ONLY features that
are computable at inference time step t using observation/action history
from steps [0, t] — no feature here may use information from steps > t.
See CLAUDE.md §14 and the Phase 9 prompt for the taxonomy and the hard
no-hindsight constraint.

Entry points
------------
    build_causal_features(step_df) -> DataFrame
        One row per (episode_idx, checkpoint_t) for a single model's step log.
    build_causal_features_online(step_df_upto_t) -> dict
        Single-row causal feature dict for one episode's history up to
        "now" (the last row of the passed-in DataFrame) — used by Phase C
        for per-step online inference. No checkpoint stride: caller decides
        which step to call this at.

Checkpoints (batch extraction only; not used by the online path)
------------------------------------------------------------------
Episodes in the existing step logs are all exactly 150 steps (verified
empirically, see Phase A report — no early termination). Checkpoints are a
fixed stride over the episode: CAUSAL_CHECKPOINT_START (19) to
MAX_EPISODE_STEPS_PICKANDPLACE - 1 (149), every CAUSAL_CHECKPOINT_STRIDE
(10) steps -> 13 checkpoints per episode. CAUSAL_CHECKPOINT_START = 19 is
the earliest step with a full CAUSAL_WINDOW_SHORT (20-step) trailing window
available (steps 0..19). range(19, 150, 10) yields 14 checkpoints per
episode (19, 29, ..., 149), not 13 — 149 is included because 150 is an
exact multiple of the stride plus the start offset.

Window sizes (config.py, reused from the existing taxonomy — not new
arbitrary values):
    CAUSAL_WINDOW_SHORT = GRASP_LIFT_WINDOW (20) — recent-dynamics window
    CAUSAL_WINDOW_LONG  = WRONG_DIR_WINDOW  (50) — sustained-trend window,
        matches the window failure_mode_labeling.py's own (hindsight)
        divergent-transport check uses.

Hard constraint
---------------
`condition` and `attack_level` are never features — same discipline as
Phase 8 (`scripts/train_failure_classifier.py`).
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    REACH_THRESHOLD,
    GRASP_DIST_THRESHOLD,
    GRASP_TRACKING_THRESHOLD,
    GRASP_WINDOW,
    DROP_SEPARATION_THRESHOLD,
    CAUSAL_WINDOW_SHORT,
    CAUSAL_WINDOW_LONG,
    CAUSAL_CHECKPOINT_START,
    CAUSAL_CHECKPOINT_STRIDE,
    MAX_EPISODE_STEPS_PICKANDPLACE,
)

FORBIDDEN_FEATURES = {"condition", "attack_level", "method"}

_REQUIRED_SPATIAL = [
    "distance_to_object",
    "distance_to_true_goal",
    "distance_to_perceived_goal",
    "object_velp_z",
    "object_pos_z",
    "gripper_aperture",
    "action_norm",
    "intended_action_norm",
    "safety_violation",
    "is_success",
    "reward",
]


def _slope(y: np.ndarray) -> float:
    """Causal linear-regression slope over whatever window `y` already is."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _grasp_streak_series(dto: np.ndarray, obj_v: np.ndarray, grp_v: np.ndarray) -> np.ndarray:
    """Causal running streak of consecutive steps meeting the two kinematic
    grasp criteria (in-contact + co-moving), ending at each index i.
    streak[i] depends only on data at indices <= i.
    """
    n = len(dto)
    streak = np.zeros(n, dtype=int)
    consecutive = 0
    for i in range(n):
        in_contact = dto[i] < GRASP_DIST_THRESHOLD
        tracking = in_contact and (np.linalg.norm(obj_v[i] - grp_v[i]) < GRASP_TRACKING_THRESHOLD)
        consecutive = consecutive + 1 if (in_contact and tracking) else 0
        streak[i] = consecutive
    return streak


def _row_features(ep: pd.DataFrame, t: int) -> dict:
    """Compute one causal feature row using only ep.iloc[0:t+1].

    `t` is a positional index into `ep` (already sorted by timestep), so
    "now" = ep.iloc[t] and history = ep.iloc[0:t+1]. Every quantity below is
    either the current-step value, a cumulative statistic over [0, t], or a
    statistic over a trailing window ending at t — never a value derived
    from steps > t.
    """
    past = ep.iloc[: t + 1]
    n_past = len(past)

    dto = past["distance_to_object"].values
    dttg = past["distance_to_true_goal"].values
    dtpg = past["distance_to_perceived_goal"].values
    anrm = past["action_norm"].values
    ianrm = past["intended_action_norm"].values
    aper = past["gripper_aperture"].values
    ovz = past["object_velp_z"].values
    opz = past["object_pos_z"].values
    svis = past["safety_violation"].values
    isucc = past["is_success"].values
    rew = past["reward"].values
    obj_v = past[["object_velp_x", "object_velp_y", "object_velp_z"]].values
    grp_v = past[["grip_velp_x", "grip_velp_y", "grip_velp_z"]].values

    w_short = min(CAUSAL_WINDOW_SHORT, n_past)
    w_long = min(CAUSAL_WINDOW_LONG, n_past)

    dto_short = dto[-w_short:]
    dttg_short = dttg[-w_short:]
    dttg_long = dttg[-w_long:]
    anrm_short = anrm[-w_short:]
    aper_short = aper[-w_short:]
    ovz_short = ovz[-w_short:]
    opz_short = opz[-w_short:]

    # ── grasp/contact causal proxy (running streak, no forward lift check) ──
    streak_series = _grasp_streak_series(dto, obj_v, grp_v)
    contact_streak_now = int(streak_series[-1])
    grasp_kinematic_ever_sofar = float(np.any(streak_series >= GRASP_WINDOW))

    # ── action ratio (guard div-by-zero same as Phase 8) ─────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ianrm > 1e-6, anrm / ianrm, 1.0)

    goal_offset = np.abs(dtpg - dttg)

    feats = {
        # time reference — trivially causal, known at t
        "t_norm": t / (MAX_EPISODE_STEPS_PICKANDPLACE - 1),

        # gripper -> object distance
        "dto_now": float(dto[-1]),
        "dto_min_sofar": float(dto.min()),
        "dto_mean_sofar": float(dto.mean()),
        "dto_slope_short": _slope(dto_short),
        "reached_ever_sofar": float(np.any(dto < REACH_THRESHOLD)),

        # causal grasp/drop proxies (no forward lift confirmation — that
        # would require steps > t; see module docstring)
        "contact_streak_now": float(contact_streak_now),
        "grasp_kinematic_ever_sofar": grasp_kinematic_ever_sofar,
        "obj_z_now": float(opz[-1]),
        "obj_z_rise_short": float(opz[-1] - opz_short.min()),
        "separated_after_contact_now": float(
            grasp_kinematic_ever_sofar and dto[-1] > DROP_SEPARATION_THRESHOLD
        ),

        # object -> true goal distance
        "dttg_now": float(dttg[-1]),
        "dttg_min_sofar": float(dttg.min()),
        "dttg_max_sofar": float(dttg.max()),
        "dttg_range_short": float(dttg_short.max() - dttg_short.min()),
        "dttg_slope_long": _slope(dttg_long),
        "near_goal_frac_sofar": float(np.mean(dttg < 0.05)),

        # perceived vs true goal (goal-spoof signal)
        "dtpg_now": float(dtpg[-1]),
        "goal_offset_now": float(goal_offset[-1]),
        "goal_offset_max_sofar": float(goal_offset.max()),
        "goal_offset_mean_sofar": float(goal_offset.mean()),
        "dtpg_min_sofar": float(dtpg.min()),

        # action norms
        "anrm_now": float(anrm[-1]),
        "anrm_mean_sofar": float(anrm.mean()),
        "anrm_max_sofar": float(anrm.max()),
        "anrm_std_short": float(anrm_short.std()),
        "anrm_slope_short": _slope(anrm_short),
        "action_ratio_mean_sofar": float(np.nanmean(ratio)),
        "action_corrupted_frac_sofar": float(np.mean(np.abs(anrm - ianrm) > 0.10)),

        # gripper aperture
        "aper_now": float(aper[-1]),
        "aper_mean_sofar": float(aper.mean()),
        "aper_std_short": float(aper_short.std()),
        "aper_min_sofar": float(aper.min()),
        "aper_max_sofar": float(aper.max()),

        # object vertical velocity (lift signal)
        "ovz_max_sofar": float(ovz.max()),
        "ovz_mean_short": float(ovz_short.mean()),

        # safety
        "safety_violation_rate_sofar": float(svis.mean()),

        # success-so-far (causal: known at t whether episode has EVER
        # transiently succeeded up to now — not the same as knowing the
        # final/hindsight outcome, since is_success is non-sticky)
        "is_success_now": float(isucc[-1]),
        "is_success_ever_sofar": float(isucc.max()),
        "n_success_steps_sofar": int(np.sum(isucc > 0)),

        # reward
        "reward_sum_sofar": float(rew.sum()),
        "reward_mean_sofar": float(rew.mean()),
    }
    return feats


def build_causal_features(step_df: pd.DataFrame,
                          episode_col: str = "episode_idx") -> pd.DataFrame:
    """Batch causal-feature extraction across all episodes in a step log.

    Args:
        step_df: Concatenated step-log DataFrame (one model, sac_her method
                 already filtered by caller — mirrors Phase 8's convention).
        episode_col: Column identifying individual episodes.

    Returns:
        DataFrame with one row per (episode_idx, checkpoint_t): all causal
        feature columns plus "episode_idx" and "checkpoint_t".
    """
    checkpoints = list(range(CAUSAL_CHECKPOINT_START,
                              MAX_EPISODE_STEPS_PICKANDPLACE,
                              CAUSAL_CHECKPOINT_STRIDE))

    rows = []
    for ep_id, ep in step_df.groupby(episode_col):
        ep = ep.sort_values("timestep").reset_index(drop=True)
        if any(col not in ep.columns or ep[col].isna().all() for col in _REQUIRED_SPATIAL):
            continue  # FetchReach or malformed episode — not in scope
        n = len(ep)
        for t in checkpoints:
            if t >= n:
                continue
            row = _row_features(ep, t)
            row["episode_idx"] = ep_id
            row["checkpoint_t"] = t
            rows.append(row)

    return pd.DataFrame(rows)


def build_causal_features_online(step_df_upto_t: pd.DataFrame) -> dict:
    """Single-step causal feature extraction for online inference (Phase C).

    Args:
        step_df_upto_t: Step rows for ONE episode, sorted by timestep, from
                         step 0 up to and including "now". The last row is
                         treated as the current step.

    Returns:
        Feature dict (same keys as build_causal_features, minus
        episode_idx/checkpoint_t) for the current step.
    """
    ep = step_df_upto_t.sort_values("timestep").reset_index(drop=True)
    return _row_features(ep, len(ep) - 1)
