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
MAX_EPISODE_STEPS   = 50
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
MAX_EPISODE_STEPS_PICKANDPLACE = 50

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
