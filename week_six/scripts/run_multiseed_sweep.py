"""
TAIRO B0–B3 multi-seed benchmark sweep.

Iterates over all seeds × conditions × methods × recovery versions and writes
one row per episode (episode_results.csv) and one row per timestep (step_logs.csv)
to results/data/.

Benchmark layer assignment
--------------------------
B0 : condition == "clean", no recovery        (clean SAC+HER baseline)
B1 : all conditions,       no recovery        (attacked, base policies)
B2 : all conditions,       recovery_v2
B3 : all conditions,       recovery_v3        (current best)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import (
    RANDOM_SEEDS,
    ALL_CONDITIONS,
    N_EPISODES_PER_SEED,
    DATA_DIR,
    MODEL_PATH,
    SB3_AVAILABLE,
)
from envs.fetchreach_env import make_env
from evaluation.episode_runner import run_episode

_ATTACK_LEVELS = {
    "clean":                0.0,
    "sensor_dropout":       0.0,
    "sensor_bias":          0.1,
    "action_clipping":      0.3,
    "action_delay":         0.0,
    "action_reversal":      0.0,
    "goal_spoof_immediate": 0.1,
    "goal_spoof_midep":     0.1,
}

# (use_recovery, recovery_version) pairs that produce B1 / B2 / B3 rows.
_RECOVERY_CONFIGS = [
    (False, "none"),
    (True,  "v2"),
    (True,  "v3"),
]


def _layer(condition: str, use_recovery: bool, recovery_version: str) -> str:
    if not use_recovery:
        return "B0" if condition == "clean" else "B1"
    return "B2" if recovery_version == "v2" else "B3"


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    model = None
    if SB3_AVAILABLE:
        from stable_baselines3 import SAC
        print(f"[sweep] Loading SAC+HER model from {MODEL_PATH}")
        model = SAC.load(MODEL_PATH)
    else:
        print("[sweep] WARNING: SB3 not available — sac_her runs will be skipped.")

    episode_rows = []
    step_rows = []
    episode_idx = 0

    for seed in RANDOM_SEEDS:
        env = make_env(seed=seed)
        print(f"\n[sweep] seed={seed}")

        for condition in ALL_CONDITIONS:
            attack_level = _ATTACK_LEVELS[condition]

            for base_method in ["rule_based", "sac_her"]:
                if base_method == "sac_her" and model is None:
                    continue

                for use_recovery, recovery_version in _RECOVERY_CONFIGS:
                    method_label = (
                        f"{base_method}_recovery" if use_recovery else base_method
                    )
                    layer = _layer(condition, use_recovery, recovery_version)

                    for ep in range(N_EPISODES_PER_SEED):
                        result, step_df = run_episode(
                            env=env,
                            method=base_method,
                            condition=condition,
                            seed=seed + ep,
                            use_recovery=use_recovery,
                            recovery_version=recovery_version,
                        )

                        row = vars(result).copy()
                        row["method"]           = method_label
                        row["attack_level"]     = attack_level
                        row["episode_idx"]      = episode_idx
                        row["recovery_version"] = recovery_version
                        row["benchmark_layer"]  = layer
                        episode_rows.append(row)

                        step_df = step_df.copy()
                        step_df["episode_idx"]      = episode_idx
                        step_df["recovery_version"] = recovery_version
                        step_df["benchmark_layer"]  = layer
                        step_rows.append(step_df)

                        episode_idx += 1

        env.close()

    episode_df = pd.DataFrame(episode_rows)
    step_df_all = pd.concat(step_rows, ignore_index=True)

    ep_path   = os.path.join(DATA_DIR, "episode_results.csv")
    step_path = os.path.join(DATA_DIR, "step_logs.csv")
    episode_df.to_csv(ep_path, index=False)
    step_df_all.to_csv(step_path, index=False)

    print(f"\n[sweep] Done — {episode_idx} episodes, {len(step_df_all)} timesteps.")
    print(f"        {ep_path}")
    print(f"        {step_path}")


if __name__ == "__main__":
    main()
