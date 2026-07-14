"""
Phase 1 jerk diagnostic — per-channel arm/gripper jerk distributions.

Runs inference-only replay across all four sweep models and all 11 conditions.
Computes arm_jerk[t] = ||executed[t][:3] - prev[:3]||  (L2, dims 0-2)
        grip_jerk[t] = |executed[t][3] - prev[3]|       (abs, dim 3)
using the same previous_action update rule as episode_runner.py.

Does NOT write to any episode_results or summary CSVs.
Output: results/data/phase1_jerk_raw.csv (per-step jerk values)

Usage:
    conda run -n reu_robotics python3 scripts/phase1_jerk_diagnostic.py
    conda run -n reu_robotics python3 scripts/phase1_jerk_diagnostic.py --seeds 0 1 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEEDS, ALL_CONDITIONS, ATTACK_LEVELS,
    N_EPISODES_PER_SEED, DATA_DIR,
    MAX_EPISODE_STEPS_PICKANDPLACE,
    MODEL_PATH_PICKANDPLACE,
    MODEL_PATH_PICKANDPLACE_RANDOMIZED,
    MODEL_PATH_PICKANDPLACE_2M,
    MODEL_PATH_PICKANDPLACE_RANDOMIZED_2M,
    SAFETY_ARM_JERK_THRESHOLD, SAFETY_GRIPPER_JERK_THRESHOLD,
    SB3_AVAILABLE,
)
from envs.fetchpickandplace_env import make_env, distance_to_goal
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack


MODELS = {
    # config.py now defines all four as distinct, MD5-verified constants
    # (fixed 2026-07-13 — MODEL_PATH_PICKANDPLACE / _RANDOMIZED used to
    # silently alias the _500k checkpoints). One source of truth.
    "clean_2M":        MODEL_PATH_PICKANDPLACE_2M,
    "randomized_2M":   MODEL_PATH_PICKANDPLACE_RANDOMIZED_2M,
    "clean_500k":      MODEL_PATH_PICKANDPLACE,
    "randomized_500k": MODEL_PATH_PICKANDPLACE_RANDOMIZED,
}


def _collect_jerk_steps(model, env, condition, seed, n_episodes, max_steps):
    """Run n_episodes under `condition` and return a list of per-step dicts."""
    rows = []
    attack_level = ATTACK_LEVELS[condition]

    for ep in range(n_episodes):
        reset_seed = 100 * seed + ep
        obs, _ = env.reset(seed=reset_seed)
        previous_action = None
        bias_vector = None
        goal_offset = None
        object_pose_offset = None

        for t in range(max_steps):
            policy_obs, bias_vector, goal_offset, object_pose_offset = apply_sensor_attack(
                condition, obs, t, bias_vector, goal_offset,
                attack_level=attack_level,
                object_pose_offset=object_pose_offset,
            )

            intended_action, _ = model.predict(policy_obs, deterministic=True)
            intended_action = np.asarray(intended_action, dtype=np.float32).copy()

            executed_action = apply_action_attack(
                condition, intended_action, previous_action, attack_level=attack_level
            )

            if previous_action is not None:
                arm_jerk  = float(np.linalg.norm(executed_action[:3] - previous_action[:3]))
                grip_jerk = float(abs(float(executed_action[3]) - float(previous_action[3])))
                flagged   = int(arm_jerk > SAFETY_ARM_JERK_THRESHOLD or
                                grip_jerk > SAFETY_GRIPPER_JERK_THRESHOLD)
                rows.append({
                    "condition":   condition,
                    "seed":        seed,
                    "episode":     ep,
                    "timestep":    t,
                    "arm_jerk":    arm_jerk,
                    "grip_jerk":   grip_jerk,
                    "action_norm": float(np.linalg.norm(executed_action)),
                    "flagged":     flagged,
                })

            previous_action = (
                intended_action.copy() if condition == "action_delay" else executed_action.copy()
            )

            obs, _, terminated, truncated, _ = env.step(executed_action)
            if terminated or truncated:
                break

    return rows


def _percentile_table(series, label=""):
    p = np.percentile(series, [50, 90, 95, 99, 99.9])
    return {
        "label":  label,
        "n":      len(series),
        "p50":    round(p[0], 4),
        "p90":    round(p[1], 4),
        "p95":    round(p[2], 4),
        "p99":    round(p[3], 4),
        "p99.9":  round(p[4], 4),
        "max":    round(float(series.max()), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--output-path", type=str, default=None,
                         help="Override output CSV path (default: DATA_DIR/phase1_jerk_raw.csv)")
    args = parser.parse_args()

    seeds      = args.seeds      if args.seeds      is not None else RANDOM_SEEDS
    n_episodes = args.n_episodes if args.n_episodes is not None else N_EPISODES_PER_SEED
    max_steps  = MAX_EPISODE_STEPS_PICKANDPLACE

    if not SB3_AVAILABLE:
        print("ERROR: SB3 not available.")
        return

    from stable_baselines3 import SAC

    all_rows = []

    for model_tag, model_path in MODELS.items():
        model_file = model_path + ".zip" if not os.path.exists(model_path) and os.path.exists(model_path + ".zip") else model_path
        if not (os.path.exists(model_file) or os.path.exists(model_file + ".zip")):
            print(f"[diag] SKIP {model_tag}: model not found at {model_path}")
            continue

        print(f"\n[diag] Loading model: {model_tag}")
        _tmp_env = make_env(seed=0)
        model = SAC.load(model_path, env=_tmp_env)
        _tmp_env.close()

        for condition in ALL_CONDITIONS:
            print(f"  condition={condition:<30}", end="", flush=True)
            env = make_env(seed=seeds[0])

            rows = []
            for seed in seeds:
                seed_rows = _collect_jerk_steps(model, env, condition, seed, n_episodes, max_steps)
                for r in seed_rows:
                    r["model"] = model_tag
                rows.extend(seed_rows)

            env.close()
            for r in rows:
                all_rows.append(r)

            n_steps  = len(rows)
            n_flag   = sum(r["flagged"] for r in rows)
            print(f"  {n_steps:>6} jerk-steps, {n_flag:>5} flagged ({100*n_flag/max(n_steps,1):.2f}%)")

    df = pd.DataFrame(all_rows)
    out_path = args.output_path if args.output_path is not None else os.path.join(DATA_DIR, "phase1_jerk_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[diag] Raw jerk data saved: {out_path}  ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE 1 REPORT — Per-channel jerk distributions")
    print(f"ARM threshold = {SAFETY_ARM_JERK_THRESHOLD}   "
          f"GRIPPER threshold = {SAFETY_GRIPPER_JERK_THRESHOLD}")
    print("="*80)

    for model_tag in MODELS:
        mdf = df[df["model"] == model_tag]
        if mdf.empty:
            continue

        print(f"\n{'─'*60}")
        print(f"MODEL: {model_tag}")
        print(f"{'─'*60}")

        # Clean-condition FP rate (arm channel)
        clean_df = mdf[(mdf["condition"] == "clean")]
        if not clean_df.empty:
            pt_arm  = _percentile_table(clean_df["arm_jerk"],  "arm_jerk  CLEAN")
            pt_grip = _percentile_table(clean_df["grip_jerk"], "grip_jerk CLEAN")
            n_fp_arm  = int((clean_df["arm_jerk"]  > SAFETY_ARM_JERK_THRESHOLD).sum())
            n_fp_grip = int((clean_df["grip_jerk"] > SAFETY_GRIPPER_JERK_THRESHOLD).sum())
            total_clean = len(clean_df)
            print(f"\n  CLEAN stats ({total_clean} jerk-steps across {len(seeds)} seeds × {n_episodes} eps):")
            print(f"  {'channel':<12} {'p50':>8} {'p95':>8} {'p99':>8} {'p99.9':>8} {'max':>8}  {'FPs':>6}  {'FP%':>6}")
            for pt, n_fp in [(pt_arm, n_fp_arm), (pt_grip, n_fp_grip)]:
                print(f"  {pt['label']:<12} {pt['p50']:>8.4f} {pt['p95']:>8.4f} "
                      f"{pt['p99']:>8.4f} {pt['p99.9']:>8.4f} {pt['max']:>8.4f}  "
                      f"{n_fp:>6}  {100*n_fp/max(total_clean,1):>5.2f}%")

        # All conditions — arm_jerk flag rate
        print(f"\n  Per-condition flagged step rate (arm_jerk > {SAFETY_ARM_JERK_THRESHOLD} "
              f"OR grip_jerk > {SAFETY_GRIPPER_JERK_THRESHOLD}):")
        print(f"  {'condition':<30} {'n_steps':>8} {'n_flag':>7} {'flag%':>7} "
              f"{'arm_p99':>9} {'arm_max':>9} {'grp_p99':>9} {'grp_max':>9}")
        for cond in ALL_CONDITIONS:
            cdf = mdf[mdf["condition"] == cond]
            if cdf.empty:
                continue
            n_steps = len(cdf)
            n_flag  = int(cdf["flagged"].sum())
            arm_p99 = float(np.percentile(cdf["arm_jerk"], 99))
            arm_max = float(cdf["arm_jerk"].max())
            grp_p99 = float(np.percentile(cdf["grip_jerk"], 99))
            grp_max = float(cdf["grip_jerk"].max())
            print(f"  {cond:<30} {n_steps:>8} {n_flag:>7} {100*n_flag/max(n_steps,1):>6.2f}%"
                  f" {arm_p99:>9.4f} {arm_max:>9.4f} {grp_p99:>9.4f} {grp_max:>9.4f}")

    # Episode-level FP rate on clean (any step flagged → episode flagged)
    print("\n" + "="*80)
    print("EPISODE-LEVEL false positive rate on CLEAN (fraction of clean episodes with ≥1 flagged step):")
    for model_tag in MODELS:
        mdf = df[(df["model"] == model_tag) & (df["condition"] == "clean")]
        if mdf.empty:
            continue
        ep_groups = mdf.groupby(["seed", "episode"])["flagged"].max()
        ep_fp_rate = float(ep_groups.mean())
        n_eps = len(ep_groups)
        print(f"  {model_tag:<20}  {ep_fp_rate*100:>5.1f}%  ({int(ep_groups.sum())} / {n_eps} episodes flagged)")

    print("\n[diag] Done.")


if __name__ == "__main__":
    main()
