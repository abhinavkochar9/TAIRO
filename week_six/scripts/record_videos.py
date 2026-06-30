"""
Record 10 episodes of FetchReach-v4 for each (policy, condition) combination.

Videos are saved as .mp4 files under:
  results/videos/<condition>/<policy_name>/

Usage:
  conda run -n reu_robotics python3 scripts/record_videos.py
  conda run -n reu_robotics python3 scripts/record_videos.py \
      --conditions sensor_dropout action_reversal \
      --policies sac_her sac_her_recovery_v3 \
      --n-episodes 5
"""

import argparse
import os
import sys

import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import SAC

from config import MAX_EPISODE_STEPS, RESULTS_DIR
from envs.fetchreach_env import distance_to_goal, make_env
from policies.sac_her_policy import SACHerPolicy
from attacks.action_attacks import manipulate_action
from attacks.sensor_attacks import apply_sensor_bias, apply_sensor_dropout, shift_target
import recovery.recovery_v2 as _rv2
import recovery.recovery_v3 as _rv3

# ---------------------------------------------------------------------------
# Canonical lists
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [
    "clean",
    "sensor_dropout",
    "sensor_bias",
    "goal_spoof_immediate",
    "goal_spoof_midep",
    "action_delay",
    "action_clipping",
    "action_reversal",
]

ALL_POLICIES = [
    "sac_her",
    "sac_her_recovery_v2",
    "sac_her_recovery_v3",
]

# Recovery policies skip the clean condition (consistent with the sweep)
_RECOVERY_POLICIES = {"sac_her_recovery_v2", "sac_her_recovery_v3"}


# ---------------------------------------------------------------------------
# Per-step attack helpers
# ---------------------------------------------------------------------------

def _apply_sensor_attack(condition, obs, t, bias_vector, goal_offset):
    """Return (policy_obs, bias_vector, goal_offset) after applying sensor attack."""
    policy_obs = {k: np.asarray(v).copy() for k, v in obs.items()}

    if condition == "sensor_dropout":
        policy_obs = apply_sensor_dropout(policy_obs, fields=["observation"])

    elif condition == "sensor_bias":
        policy_obs, bias_vector = apply_sensor_bias(
            policy_obs, magnitude=0.10, bias_vector=bias_vector
        )

    elif condition == "goal_spoof_immediate":
        policy_obs, returned_offset = shift_target(
            policy_obs, shift_scale=0.10, step=t,
            shift_step=None, goal_offset=goal_offset,
        )
        if returned_offset is not None:
            goal_offset = returned_offset

    elif condition == "goal_spoof_midep":
        policy_obs, returned_offset = shift_target(
            policy_obs, shift_scale=0.10, step=t,
            shift_step=20, goal_offset=goal_offset,
        )
        if returned_offset is not None:
            goal_offset = returned_offset

    return policy_obs, bias_vector, goal_offset


def _apply_action_attack(condition, intended_action, previous_action):
    """Return the executed action after applying any action-level attack."""
    if condition == "action_delay":
        return manipulate_action(
            intended_action, attack_type="action_delay",
            previous_action=previous_action,
        )
    elif condition == "action_clipping":
        return manipulate_action(
            intended_action, attack_type="action_clipping", clip_value=0.30
        )
    elif condition == "action_reversal":
        return manipulate_action(intended_action, attack_type="action_reverse")
    # clean / sensor attacks: action is unmodified
    return intended_action.copy()


# ---------------------------------------------------------------------------
# Single (policy, condition) block
# ---------------------------------------------------------------------------

def record_pair(policy_name, condition, policy, n_episodes, output_dir):
    """Record n_episodes for one (policy_name, condition) and print a summary."""
    video_folder = os.path.join(output_dir, condition, policy_name)
    os.makedirs(video_folder, exist_ok=True)

    use_recovery = policy_name in _RECOVERY_POLICIES
    recovery_mod = (
        _rv2 if policy_name == "sac_her_recovery_v2" else
        _rv3 if policy_name == "sac_her_recovery_v3" else
        None
    )

    # One env for the whole block; individual episodes use env.reset(seed=ep_idx)
    base_env = make_env(seed=0, rgb_mode=True)
    env = RecordEpisodeStatistics(base_env, buffer_length=n_episodes)
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=f"{policy_name}_{condition}",
        episode_trigger=lambda ep: True,
    )

    successes = []

    try:
        for ep_idx in range(n_episodes):
            obs, _info = env.reset(seed=ep_idx)

            # Per-episode stateful attack state
            previous_action = None   # None is the step-0 sentinel for action_delay
            prev_obs = None
            bias_vector = None
            goal_offset = None
            step_distances = []
            recovery_state = recovery_mod.RecoveryState() if use_recovery else None

            total_reward = 0.0
            ep_success = False

            for t in range(MAX_EPISODE_STEPS):
                # Sensor attack: produce policy_obs from raw obs
                policy_obs, bias_vector, goal_offset = _apply_sensor_attack(
                    condition, obs, t, bias_vector, goal_offset
                )

                # Policy predicts from (possibly attacked) observation
                intended_action = np.asarray(
                    policy(env, policy_obs), dtype=np.float32
                )

                # Action attack
                executed_action = _apply_action_attack(
                    condition, intended_action, previous_action
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
                f"  ep {ep_idx:2d} | success={ep_success} | reward={total_reward:.2f}"
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
        help="Number of episodes per (policy, condition) pair.",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(RESULTS_DIR, "videos"),
        help="Root output directory. Videos go in <output-dir>/<condition>/<policy>/",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(RESULTS_DIR, "models", "sac_her_fetchreach_model"),
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
