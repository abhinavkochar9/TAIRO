"""
Record episodes of FetchReach-v4 for each (policy, condition) combination,
spread evenly across the canonical sweep seeds (config.RANDOM_SEEDS).

Videos are saved as .mp4 files under:
  results/videos/<condition>/<policy_name>/
  Named: {policy}_{condition}_seed{N}-episode-{M}.mp4

Seeding note
------------
In the full sweep, every episode within a seed block resets with the same
seed value (env.reset(seed=seed)), making all 30 repetitions per seed
bit-identical.  The 5 seeds (RANDOM_SEEDS) are where scenario diversity
comes from — each represents a different goal position.  Recording here
therefore samples eps_per_seed episodes per seed; they are identical to
each other within a seed but show different scenarios across seeds.

--n-episodes is the TOTAL episodes per (policy, condition) pair, divided
evenly across seeds.  Default 10 → 2 episodes per seed × 5 seeds.
If --n-episodes is not divisible by the number of seeds, the actual total
is rounded down to eps_per_seed * len(RANDOM_SEEDS).

Usage:
  conda run -n reu_robotics python3 scripts/record_videos.py
  conda run -n reu_robotics python3 scripts/record_videos.py \\
      --conditions sensor_dropout action_reversal \\
      --policies sac_her sac_her_recovery_v3 \\
      --n-episodes 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import SAC

from config import ALL_CONDITIONS, ATTACK_LEVELS, MAX_EPISODE_STEPS, RANDOM_SEEDS, RESULTS_DIR
from envs.fetchpickandplace_env import distance_to_goal, make_env
from policies.sac_her_policy import SACHerPolicy
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack
import recovery.recovery_v2 as _rv2
import recovery.recovery_v3 as _rv3

# ---------------------------------------------------------------------------
# Canonical lists
# ---------------------------------------------------------------------------

ALL_POLICIES = [
    "sac_her",
    "sac_her_recovery_v2",
    "sac_her_recovery_v3",
]

# Recovery policies skip the clean condition (consistent with the sweep)
_RECOVERY_POLICIES = {"sac_her_recovery_v2", "sac_her_recovery_v3"}


# ---------------------------------------------------------------------------
# Single (policy, condition) block
# ---------------------------------------------------------------------------

def record_pair(policy_name, condition, policy, n_episodes, output_dir):
    """Record episodes for one (policy_name, condition), spread across RANDOM_SEEDS.

    n_episodes is the total target; it is divided by len(RANDOM_SEEDS) to get
    eps_per_seed.  A separate RecordVideo env is created per seed so that video
    filenames embed the seed: {policy}_{condition}_seed{N}-episode-{M}.mp4.

    All episodes within a seed are identical (the sweep resets with the same seed
    each time), so eps_per_seed > 1 shows the same scenario repeatedly — useful
    for confirming consistency but not for adding scenario variety.
    """
    video_folder = os.path.join(output_dir, condition, policy_name)
    os.makedirs(video_folder, exist_ok=True)

    use_recovery = policy_name in _RECOVERY_POLICIES
    recovery_mod = (
        _rv2 if policy_name == "sac_her_recovery_v2" else
        _rv3 if policy_name == "sac_her_recovery_v3" else
        None
    )

    attack_level = ATTACK_LEVELS[condition]

    # Divide total episodes evenly across seeds; round down if not divisible.
    eps_per_seed = max(1, n_episodes // len(RANDOM_SEEDS))

    successes = []

    for seed in RANDOM_SEEDS:
        # Fresh env + wrapper per seed so filenames embed the seed number.
        base_env = make_env(seed=seed, rgb_mode=True)
        env = RecordEpisodeStatistics(base_env, buffer_length=eps_per_seed)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=f"{policy_name}_{condition}_seed{seed}",
            episode_trigger=lambda ep: True,
        )

        try:
            for within_seed_ep in range(eps_per_seed):
                obs, _info = env.reset(seed=seed)  # same seed each rep — matches sweep

                # Per-episode stateful attack state
                previous_action = None   # None is the step-0 sentinel for action_delay
                prev_obs = None
                bias_vector = None
                goal_offset = None
                object_pose_offset = None  # PickAndPlace: object_pose_spoof per-episode offset
                step_distances = []
                recovery_state = recovery_mod.RecoveryState() if use_recovery else None

                total_reward = 0.0

                for t in range(MAX_EPISODE_STEPS):
                    # Sensor attack: produce policy_obs from raw obs
                    policy_obs, bias_vector, goal_offset, object_pose_offset = apply_sensor_attack(
                        condition, obs, t, bias_vector, goal_offset, attack_level=attack_level,
                        object_pose_offset=object_pose_offset,
                    )

                    # Policy predicts from (possibly attacked) observation
                    intended_action = np.asarray(
                        policy(env, policy_obs), dtype=np.float32
                    )

                    # Action attack
                    executed_action = apply_action_attack(
                        condition, intended_action, previous_action, attack_level=attack_level,
                    )

                    # Recovery uses raw unattacked obs so it steers toward real goal
                    if use_recovery:
                        executed_action, _triggered = recovery_mod.maybe_apply_recovery(
                            obs=obs,
                            action=executed_action,
                            prev_obs=prev_obs,
                            prev_action=previous_action,
                            step_distances=step_distances,
                            step=t,
                            env=env,
                            state=recovery_state,
                        )

                    prev_obs = {k: np.asarray(v).copy() for k, v in obs.items()}
                    previous_action = (
                        intended_action.copy() if condition == "action_delay" else executed_action.copy()
                    )

                    obs, reward, terminated, truncated, info = env.step(executed_action)
                    step_distances.append(distance_to_goal(obs))
                    total_reward += float(reward)

                    if terminated or truncated:
                        break

                ep_success = bool(info.get("is_success", False))
                successes.append(ep_success)
                print(
                    f"  seed={seed} ep={within_seed_ep} | success={ep_success} | reward={total_reward:.2f}"
                )

        finally:
            env.close()

    rate = sum(successes) / len(successes) if successes else 0.0
    print(f"  Success rate: {rate:.1%} ({sum(successes)}/{len(successes)})\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record FetchReach-v4 episode videos for each (policy, condition)."
    )
    parser.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS,
        metavar="CONDITION",
        help="Conditions to record. Default: all 8.",
    )
    parser.add_argument(
        "--policies", nargs="+", default=ALL_POLICIES,
        metavar="POLICY",
        help="Policies to record. Default: all 3.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help=(
            "Total episodes per (policy, condition) pair, spread across seeds. "
            "Divided by len(RANDOM_SEEDS) to get episodes-per-seed; rounded down "
            "if not evenly divisible. Default 10 → 2 per seed × 5 seeds."
        ),
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(RESULTS_DIR, "videos"),
        help="Root output directory. Videos go in <output-dir>/<condition>/<policy>/",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(RESULTS_DIR, "models", "sac_her_pickandplace_clean_2M"),
        help="Path to the SAC+HER model zip (without .zip extension).",
    )
    args = parser.parse_args()

    # Load model once; reuse across all (policy, condition) pairs
    load_env = make_env(seed=0)
    model = SAC.load(args.model_path, env=load_env)
    load_env.close()
    policy = SACHerPolicy(model)

    for policy_name in args.policies:
        for condition in args.conditions:
            if condition == "clean" and policy_name in _RECOVERY_POLICIES:
                print(f"Skipping clean for {policy_name} (consistent with sweep).")
                continue

            print(f"\n=== {policy_name} | {condition} ===")
            record_pair(
                policy_name=policy_name,
                condition=condition,
                policy=policy,
                n_episodes=args.n_episodes,
                output_dir=args.output_dir,
            )

    print("Done. Videos saved to:", args.output_dir)


if __name__ == "__main__":
    main()
