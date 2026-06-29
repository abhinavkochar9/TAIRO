"""
Shared configuration for the TAIRO Week 6 benchmark.

Single source of truth for all constants. All scripts and modules import
from here — never hardcode these values elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Experiment constants
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
]

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
