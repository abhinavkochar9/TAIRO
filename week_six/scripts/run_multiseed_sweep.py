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
    ALL_METHODS,
    ATTACK_LEVELS,
    N_EPISODES_PER_SEED,
    DATA_DIR,
    MODEL_PATH,
    SB3_AVAILABLE,
)
from envs.fetchreach_env import make_env
from evaluation.episode_runner import run_episode

_RECOVERY_VERSION = {
    "sac_her":             "none",
    "sac_her_recovery_v2": "v2",
    "sac_her_recovery_v3": "v3",
}


def _layer(method: str, condition: str) -> str:
    if method == "sac_her":
        return "B0" if condition == "clean" else "B1"
    if method == "sac_her_recovery_v2":
        return "B2"
    return "B3"  # sac_her_recovery_v3


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create a temporary env just to satisfy HerReplayBuffer at load time
    _tmp_env = make_env(seed=0)
    model = None
    if SB3_AVAILABLE:
        from stable_baselines3 import SAC
        print(f"[sweep] Loading SAC+HER model from {MODEL_PATH}")
        model = SAC.load(MODEL_PATH, env=_tmp_env)
    else:
        print("[sweep] WARNING: SB3 not available — sac_her runs will be skipped.")
    _tmp_env.close()

    episode_rows = []
    step_rows = []
    episode_idx = 0

    for seed in RANDOM_SEEDS:
        env = make_env(seed=seed)
        print(f"\n[sweep] seed={seed}")

        for condition in ALL_CONDITIONS:
            attack_level = ATTACK_LEVELS[condition]

            for method in ALL_METHODS:
                if model is None:
                    continue

                recovery_version = _RECOVERY_VERSION[method]
                layer = _layer(method, condition)

                for ep in range(N_EPISODES_PER_SEED):
                    result, step_df = run_episode(
                        env=env,
                        method=method,
                        condition=condition,
                        seed=seed,
                        model=model,
                        attack_level=attack_level,
                        recovery_version=recovery_version,
                    )

                    row = vars(result).copy()
                    row["method"]           = method
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
