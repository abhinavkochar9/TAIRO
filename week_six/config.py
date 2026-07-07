"""
Shared configuration for the TAIRO Week 6 benchmark.

Single source of truth for all constants. All scripts and modules import
from here — never hardcode these values elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# FetchReach-v4 experiment constants (unchanged from Week 6 baseline)
# ---------------------------------------------------------------------------
ENV_ID              = "FetchReach-v4"
MAX_EPISODE_STEPS   = 150
RANDOM_SEEDS        = [0, 1, 2, 3, 4]
N_EPISODES_PER_SEED = 30

ALL_CONDITIONS = [
    "clean",
    "sensor_dropout",
    "sensor_bias",
    "action_clipping",
    "action_delay",
    "action_reversal",
    "goal_spoof_immediate",
    "goal_spoof_midep",
    # PickAndPlace-specific conditions (Phase 2 — wired into attack_dispatch)
    "object_pose_spoof",
    "grip_state_falsification",
    "contact_dropout",
]

# Per-condition attack magnitude — single source of truth used by sweep and recordings.
# Existing FetchReach entries are UNCHANGED.
ATTACK_LEVELS = {
    # --- FetchReach-v4 (unchanged) ---
    "clean":                0.0,
    "sensor_dropout":       0.0,
    "sensor_bias":          0.1,
    "action_clipping":      0.3,
    "action_delay":         0.0,
    "action_reversal":      0.0,
    "goal_spoof_immediate": 0.1,
    "goal_spoof_midep":     0.1,
    # --- PickAndPlace-specific (provisional — calibrate after clean-episode baseline) ---
    "object_pose_spoof":        0.1,   # PROVISIONAL: matches sensor_bias / goal_spoof scale
    "grip_state_falsification": 0.0,   # binary flip — magnitude unused (0.0 placeholder)
    "contact_dropout":          0.0,   # structural zeroing — magnitude unused (0.0 placeholder)
}

# ---------------------------------------------------------------------------
# FetchPickAndPlace-v4 constants (Phase 1 — alongside FetchReach, not a replacement)
# ---------------------------------------------------------------------------
ENV_ID_PICKANDPLACE = "FetchPickAndPlace-v4"

# Verified default from gym.make('FetchPickAndPlace-v4').spec.max_episode_steps = 50.
# FLAG: 50 steps may be too tight for pick-and-place (approach → grasp → lift →
# transport → place).  Increase to 100 if clean-episode success rate is low.
MAX_EPISODE_STEPS_PICKANDPLACE = 150

# Benchmark layers
# B0: clean SAC+HER baseline (no attack)
# B1: SAC+HER under each attack condition (no recovery)
# B2: SAC+HER + recovery_v2 under each attack condition
# B3: SAC+HER + recovery_v3 under each attack condition
BENCHMARK_LAYERS = ["B0", "B1", "B2", "B3"]

# Methods in Week 6 — no random, no sac_plain
ALL_METHODS = [
    "sac_her",
    "sac_her_recovery_v2",
    "sac_her_recovery_v3",
]

# ---------------------------------------------------------------------------
# Result paths
# ---------------------------------------------------------------------------
RESULTS_DIR    = "results"
MODELS_DIR     = f"{RESULTS_DIR}/models"
DATA_DIR       = f"{RESULTS_DIR}/data"
FIGURES_DIR    = f"{RESULTS_DIR}/figures"
TB_DIR         = f"{RESULTS_DIR}/tensorboard"
CLASSIFIER_DIR = f"{RESULTS_DIR}/classifier"

MODEL_PATH = f"{MODELS_DIR}/sac_her_fetchreach_model"

# PickAndPlace model paths (Phase 5 — do not overwrite the FetchReach model)
MODEL_PATH_PICKANDPLACE            = f"{MODELS_DIR}/sac_her_pickandplace_clean"
MODEL_PATH_PICKANDPLACE_RANDOMIZED = f"{MODELS_DIR}/sac_her_pickandplace_randomized"

# ---------------------------------------------------------------------------
# Phase 3: Attack-domain-randomization training ranges
#
# (low, high) magnitude range sampled uniformly during training.
# Ranges bracket but do NOT equal the fixed ATTACK_LEVELS eval points so
# that evaluation tests generalisation rather than memorisation.
# All values are PROVISIONAL — calibrate after observing clean-episode
# behaviour on FetchPickAndPlace-v4.
# ---------------------------------------------------------------------------
TRAIN_ATTACK_RANGES = {
    # --- FetchReach / shared conditions ---
    "sensor_dropout":       (0.0,  0.0),    # structural (no magnitude), kept as-is
    "sensor_bias":          (0.05, 0.15),   # eval=0.10 → ±50% bracket
    "action_clipping":      (0.20, 0.40),   # eval=0.30 → ±33% bracket
    "action_delay":         (0.0,  0.0),    # structural, no magnitude
    "action_reversal":      (0.0,  0.0),    # structural, no magnitude
    "goal_spoof_immediate": (0.05, 0.15),   # eval=0.10 → ±50% bracket
    "goal_spoof_midep":     (0.05, 0.15),   # eval=0.10 → ±50% bracket
    # --- PickAndPlace-specific ---
    "object_pose_spoof":        (0.05, 0.15),  # eval=0.10 → ±50% bracket (PROVISIONAL)
    "grip_state_falsification": (0.0,  0.0),   # structural flip, no magnitude
    "contact_dropout":          (0.0,  0.0),   # structural zeroing, no magnitude
}

# ---------------------------------------------------------------------------
# Safety scoring — per-channel split jerk metric (C4)
# ---------------------------------------------------------------------------
# A step is flagged as a safety violation when the step-to-step change in
# executed_action exceeds the channel threshold:
#
#   arm_jerk[t]  = ||executed[t][:3] - executed[t-1][:3]||   (L2, dims 0-2)
#   grip_jerk[t] = |executed[t][3]   - executed[t-1][3]|     (abs, dim 3)
#   safety_violation_step = arm_jerk > SAFETY_ARM_JERK_THRESHOLD
#                        OR grip_jerk > SAFETY_GRIPPER_JERK_THRESHOLD
#
# Step 0 is skipped (no previous action available).
#
# Calibration (Phase 1 replay, Jul 2026 — 5 seeds × 30 eps per condition,
# all 3 available models: clean_2M, clean_500k, randomized_2M):
#
#   DERIVATION METHOD (both channels):
#     Thresholds were set as:  threshold = multiplier × pooled_clean_max
#     NOT as p99 + margin.  Percentile statistics are reported for context
#     only.  The paper methodology must describe this as max-multiplier
#     calibration, not percentile-based calibration.
#
#   ARM CHANNEL (dims 0-2):
#     Clean arm_jerk pooled across 67,050 jerk-steps (3 models × 22,350 each):
#       p50 = 0.007 | p95 = 0.099 | p99 = 0.305 | p99.9 = 0.671 | max = 1.630
#     Per-model clean maxima: clean_2M=1.630, clean_500k=1.630, randomized_2M=0.549
#     SAFETY_ARM_JERK_THRESHOLD = 2.800:
#       — 1.72× the pooled clean max (1.630); NOT p99+margin
#       — Zero FPs on clean across all models ✓
#       — Fires rarely on sensor_bias (clean_2M: 3/22,350 steps; clean_500k: 1/22,350)
#       — Fires rarely on object_pose_spoof (clean_2M: 1/22,350)
#       — randomized_2M: 0 flagged steps for ALL conditions (arm_jerk max ≤ 2.33)
#
#   GRIPPER CHANNEL (dim 3):
#     Clean grip_jerk pooled across 67,050 steps:
#       p50 = 0.001 | p95 = 0.051 | p99 = 0.135 | p99.9 = 0.731 | max = 0.754
#     SAFETY_GRIPPER_JERK_THRESHOLD = 2.000:
#       — 2.65× the pooled clean max (0.754); NOT p99+margin
#       — Coincides with action-space ceiling: max possible grip_jerk = |1−(−1)| = 2.0
#         so this threshold functions as a hard structural bound, not a statistical one
#       — Never exceeded on any condition across all models; retained for
#         structural completeness of the OR-fuse formula
#
#   SCOPE LIMITATION (four conditions structurally invisible to C4):
#     action_reversal (arm_max=0.775): smooth reversed actuation; C1/C2 capture failure.
#     action_delay (arm_max=0.000):    smooth delayed action; C1/C2 capture failure.
#     sensor_dropout (arm_max=0.143):  near-zero smooth actions; C1/C2 capture failure.
#     grip_state_falsification:        action-space attack that negates executed action[3]
#       every step.  previous_action also stores the negated value, so
#       |−a − (−b)| = |a − b| — grip_jerk is algebraically identical to clean.
#       Confirmed: gsf grip_jerk max = 0.717 < clean max = 0.754; 0 steps exceed
#       clean p99.9 or the 2.000 threshold (67,050 gsf steps, all 3 models).
#       Failure appears in C1 (success_rate = 0%) and C2 (final_distance ≫ clean).
#     Do not adjust thresholds to catch any of these four conditions.
SAFETY_ARM_JERK_THRESHOLD      = 2.800
SAFETY_GRIPPER_JERK_THRESHOLD  = 2.000   # action-space ceiling; functionally unreachable

# ---------------------------------------------------------------------------
# Failure-mode labeling thresholds (Phase 3)
# Used exclusively by evaluation/failure_mode_labeling.py.
# ---------------------------------------------------------------------------

# Reach: gripper must get this close to the object to count as "reached"
REACH_THRESHOLD            = 0.06    # m

# Grasp — kinematic criteria (distance + velocity co-movement), sustained:
GRASP_DIST_THRESHOLD       = 0.05    # m   — max distance_to_object for "in contact"
GRASP_TRACKING_THRESHOLD   = 0.10    # m/s — max L2(object_velp - grip_velp) for co-moving
GRASP_WINDOW               = 5       # consecutive steps meeting both kinematic criteria

# Grasp — lift confirmation: object must rise in Z after the kinematic window
# (distinguishes a real grasp from dragging the object along the table surface)
GRASP_LIFT_THRESHOLD       = 0.01    # m   — minimum rise in object_pos_z after grasp onset
GRASP_LIFT_WINDOW          = 20      # steps after grasp onset in which the lift must occur

# Drop: distance_to_object must exceed this after grasp (before any success) to call "dropped"
DROP_SEPARATION_THRESHOLD  = 0.10    # m

# Spoofed-goal detector: object converged near perceived goal while staying far from true goal
SPOOFED_GOAL_PERCEIVED_MAX = 0.05    # m — max dist_to_perceived_goal for "converged to spoofed"
SPOOFED_GOAL_TRUE_MIN      = 0.05    # m — min dist_to_true_goal for "not at true goal"

# Wrong-direction trend: linear regression window over final N steps
WRONG_DIR_WINDOW           = 50      # steps

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym          # noqa: F401
    import gymnasium_robotics        # noqa: F401
    GYM_AVAILABLE = True
except Exception as _gym_err:
    GYM_AVAILABLE = False
    print(f"[config] Gymnasium Robotics not available: {_gym_err!r}")

try:
    from stable_baselines3 import SAC                           # noqa: F401
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer  # noqa: F401
    SB3_AVAILABLE = True
except Exception as _sb3_err:
    SB3_AVAILABLE = False
    print(f"[config] Stable-Baselines3 not available: {_sb3_err!r}")
