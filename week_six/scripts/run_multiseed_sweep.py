"""
TAIRO B0–B3 multi-seed benchmark sweep.

Iterates over seeds × conditions × methods × recovery versions and writes
one row per episode (episode_results.csv) and one row per timestep
(step_logs.csv) to results/data/.

Benchmark layer assignment
--------------------------
B0 : condition == "clean", no recovery        (clean SAC+HER baseline)
B1 : all conditions,       no recovery        (attacked, base policies)
B2 : all conditions,       recovery_v2
B3 : all conditions,       recovery_v3        (current best)

Output filenames
----------------
FetchReach (default):
    results/data/episode_results.csv          ← same as before (backward-compat)
    results/data/step_logs.csv

PickAndPlace:
    results/data/episode_results_pickandplace_clean.csv
    results/data/step_logs_pickandplace_clean.csv
    results/data/episode_results_pickandplace_randomized.csv   (--randomized)
    results/data/step_logs_pickandplace_randomized.csv

Usage examples
--------------
FetchReach (default — behaviour unchanged):
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py

PickAndPlace, default clean model:
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py --env pickandplace

PickAndPlace, specific model, 1 seed, 2 episodes (verification):
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py \\
        --env pickandplace \\
        --model-path results/models/sac_her_pickandplace_clean_500k \\
        --seeds 0 \\
        --n-episodes 2

PickAndPlace, domain-randomized model:
    conda run -n reu_robotics python3 scripts/run_multiseed_sweep.py \\
        --env pickandplace --randomized
"""

import argparse
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
    MAX_EPISODE_STEPS,
    MAX_EPISODE_STEPS_PICKANDPLACE,
    MODEL_PATH,
    MODEL_PATH_PICKANDPLACE,
    MODEL_PATH_PICKANDPLACE_RANDOMIZED,
    SB3_AVAILABLE,
)
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="TAIRO B0–B3 multi-seed benchmark sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        choices=["fetchreach", "pickandplace"],
        default="fetchreach",
        help="Environment to sweep (default: fetchreach).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Override the model file to load (no .zip extension). "
            "If omitted, uses MODEL_PATH for fetchreach or "
            "MODEL_PATH_PICKANDPLACE for pickandplace. "
            "Example: --model-path results/models/sac_her_pickandplace_clean_500k"
        ),
    )
    parser.add_argument(
        "--randomized",
        action="store_true",
        help=(
            "Use MODEL_PATH_PICKANDPLACE_RANDOMIZED as default model. "
            "Only meaningful with --env pickandplace. "
            "Ignored if --model-path is also set."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "Seeds to evaluate (space-separated). "
            "Default: all RANDOM_SEEDS from config.py. "
            "Example: --seeds 0 1 2"
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Episodes per (seed × condition × method). "
            "Default: N_EPISODES_PER_SEED from config.py. "
            "Example: --n-episodes 2"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Resolve environment, model path, step budget, and output tag.
    # FetchReach path is unchanged from the pre-flag version.
    # ------------------------------------------------------------------
    if args.env == "pickandplace":
        from envs.fetchpickandplace_env import make_env

        # --model-path takes precedence; --randomized selects the default.
        if args.model_path is not None:
            model_path = args.model_path
            # Derive a readable tag from the basename for the output filename.
            env_tag = os.path.splitext(os.path.basename(model_path))[0]
        elif args.randomized:
            model_path = MODEL_PATH_PICKANDPLACE_RANDOMIZED
            env_tag = "pickandplace_randomized"
        else:
            model_path = MODEL_PATH_PICKANDPLACE
            env_tag = "pickandplace_clean"

        max_steps = MAX_EPISODE_STEPS_PICKANDPLACE
        # PickAndPlace results go to separate files so FetchReach CSVs are never overwritten.
        ep_filename   = f"episode_results_{env_tag}.csv"
        step_filename = f"step_logs_{env_tag}.csv"

    else:  # fetchreach — fully backward-compatible
        from envs.fetchreach_env import make_env

        model_path = args.model_path if args.model_path is not None else MODEL_PATH
        max_steps  = MAX_EPISODE_STEPS
        env_tag    = "fetchreach"
        # Keep original filenames so existing analysis scripts are unaffected.
        ep_filename   = "episode_results.csv"
        step_filename = "step_logs.csv"

    seeds       = args.seeds      if args.seeds      is not None else RANDOM_SEEDS
    n_episodes  = args.n_episodes if args.n_episodes is not None else N_EPISODES_PER_SEED

    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    _tmp_env = make_env(seed=0)
    model = None
    if SB3_AVAILABLE:
        from stable_baselines3 import SAC
        print(f"[sweep] env={env_tag}  max_steps={max_steps}  model={model_path}")
        print(f"[sweep] seeds={seeds}  n_episodes={n_episodes}  "
              f"conditions={len(ALL_CONDITIONS)}  methods={len(ALL_METHODS)}")
        model = SAC.load(model_path, env=_tmp_env)
    else:
        print("[sweep] WARNING: SB3 not available — sac_her runs will be skipped.")
    _tmp_env.close()

    episode_rows = []
    step_rows    = []
    episode_idx  = 0

    # ------------------------------------------------------------------
    # Main sweep loop
    # ------------------------------------------------------------------
    for seed in seeds:
        env = make_env(seed=seed)
        print(f"\n[sweep] seed={seed}")

        for condition in ALL_CONDITIONS:
            attack_level = ATTACK_LEVELS[condition]

            for method in ALL_METHODS:
                if model is None:
                    continue

                recovery_version = _RECOVERY_VERSION[method]
                layer = _layer(method, condition)

                for _ep in range(n_episodes):
                    result, step_df = run_episode(
                        env=env,
                        method=method,
                        condition=condition,
                        seed=seed,
                        model=model,
                        attack_level=attack_level,
                        recovery_version=recovery_version,
                        max_steps=max_steps,
                    )

                    row = vars(result).copy()
                    row["method"]           = method
                    row["attack_level"]     = attack_level
                    row["episode_idx"]      = episode_idx
                    row["recovery_version"] = recovery_version
                    row["benchmark_layer"]  = layer
                    row["env"]              = env_tag
                    episode_rows.append(row)

                    step_df = step_df.copy()
                    step_df["episode_idx"]      = episode_idx
                    step_df["recovery_version"] = recovery_version
                    step_df["benchmark_layer"]  = layer
                    step_df["env"]              = env_tag
                    step_rows.append(step_df)

                    episode_idx += 1

        env.close()

    episode_df   = pd.DataFrame(episode_rows)
    step_df_all  = pd.concat(step_rows, ignore_index=True)

    ep_path   = os.path.join(DATA_DIR, ep_filename)
    step_path = os.path.join(DATA_DIR, step_filename)
    episode_df.to_csv(ep_path, index=False)
    step_df_all.to_csv(step_path, index=False)

    print(f"\n[sweep] Done — {episode_idx} episodes, {len(step_df_all)} timesteps.")
    print(f"        {ep_path}")
    print(f"        {step_path}")

    # Quick condition-coverage summary so the caller can verify all 11 ran.
    print("\n[sweep] Condition coverage (sac_her only):")
    sac_df = episode_df[episode_df["method"] == "sac_her"]
    for cond in ALL_CONDITIONS:
        n = int((sac_df["condition"] == cond).sum())
        print(f"        {cond:<30}  {n:>3} episodes")


if __name__ == "__main__":
    main()
