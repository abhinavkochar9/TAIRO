"""
Per-step attack dispatch for TAIRO benchmarks (FetchReach-v4 + FetchPickAndPlace-v4).

Provides two functions that encapsulate the sensor-level and action-level
attack logic shared between the sweep (evaluation/episode_runner.py) and
the video recorder (scripts/record_videos.py).

Why two functions instead of one
---------------------------------
The sensor attack must fire *before* the policy call (to produce the
corrupted policy_obs the policy sees), while the action attack fires
*after* (consuming intended_action produced by the policy).  Merging
both into a single post-policy function would require re-running the
sensor attack a second time, which breaks determinism for attacks that
sample a random offset on their first call (sensor_bias, goal_spoof*,
object_pose_spoof).

FetchReach vs PickAndPlace
--------------------------
FetchReach conditions: clean, sensor_dropout, sensor_bias, action_clipping,
  action_delay, action_reversal, goal_spoof_immediate, goal_spoof_midep.
PickAndPlace-specific: object_pose_spoof, grip_state_falsification,
  contact_dropout (Phase 2).
apply_sensor_attack / apply_action_attack handle all conditions; unknown
condition names fall through as no-ops so FetchReach sweeps are unaffected.

Do NOT modify attacks/sensor_attacks.py, attacks/action_attacks.py, or
the recovery modules — they are already shared and correctly imported.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from attacks.sensor_attacks import (
    add_sensor_noise,
    apply_sensor_bias,
    apply_sensor_dropout,
    shift_target,
    apply_object_pose_spoof,
    apply_contact_dropout,
)
from attacks.action_attacks import manipulate_action, ATTACK_GRIP_FALSIFY

# Onset step for goal_spoof_midep (matches the value previously defined
# in episode_runner.py and hardcoded as 20 in record_videos.py).
GOAL_SPOOF_MIDEP_STEP = 60


def apply_sensor_attack(
    condition: str,
    obs: Dict[str, np.ndarray],
    t: int,
    bias_vector: Optional[np.ndarray],
    goal_offset: Optional[np.ndarray],
    attack_level: float = 0.0,
    target_shift_step: int = 25,
    object_pose_offset: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply the sensor-level attack for *condition* and return the corrupted obs.

    Must be called BEFORE the policy, since the policy observes policy_obs.

    Args:
        condition:          Attack condition name (matches config.ALL_CONDITIONS).
        obs:                Raw observation dict from env.step() / env.reset().
        t:                  Current timestep within the episode (0-indexed).
        bias_vector:        Per-episode sensor bias; None on first step, reused after.
        goal_offset:        Per-episode goal offset; None until first activation.
        attack_level:       Magnitude parameter (e.g. noise_std, shift_scale).
                            Pass config.ATTACK_LEVELS[condition] at the call site.
        target_shift_step:  Onset step for the legacy "target_shift" condition.
        object_pose_offset: Per-episode object-pose spoof offset (PickAndPlace);
                            None on first step, reused after.

    Returns:
        (policy_obs, updated_bias_vector, updated_goal_offset, updated_object_pose_offset)
        policy_obs is a new dict; obs is never mutated.
        FetchReach conditions always return object_pose_offset=None.
    """
    policy_obs = obs  # default: policy sees raw obs; each attack fn copies internally

    if condition == "sensor_noise":
        policy_obs = add_sensor_noise(obs, noise_std=attack_level)

    elif condition == "sensor_dropout":
        policy_obs = apply_sensor_dropout(obs, fields=["observation"])

    elif condition == "sensor_bias":
        policy_obs, bias_vector = apply_sensor_bias(
            obs, magnitude=attack_level, bias_vector=bias_vector
        )

    elif condition == "goal_spoof_immediate":
        policy_obs, goal_offset = shift_target(
            obs, shift_scale=attack_level,
            step=t, shift_step=None, goal_offset=goal_offset,
        )

    elif condition == "goal_spoof_midep":
        policy_obs, new_offset = shift_target(
            obs, shift_scale=attack_level,
            step=t, shift_step=GOAL_SPOOF_MIDEP_STEP, goal_offset=goal_offset,
        )
        if new_offset is not None:
            goal_offset = new_offset

    # PickAndPlace-specific conditions (Phase 2)
    elif condition == "object_pose_spoof":
        policy_obs, object_pose_offset = apply_object_pose_spoof(
            obs, magnitude=attack_level, offset=object_pose_offset,
        )

    elif condition == "contact_dropout":
        policy_obs = apply_contact_dropout(obs)

    # Legacy condition — kept for backwards compatibility with Week 4 data.
    elif condition == "target_shift" and t >= target_shift_step:
        policy_obs, goal_offset = shift_target(
            obs, shift_scale=attack_level,
            step=t, shift_step=None, goal_offset=goal_offset,
        )

    return policy_obs, bias_vector, goal_offset, object_pose_offset


def apply_action_attack(
    condition: str,
    intended_action: np.ndarray,
    previous_action: Optional[np.ndarray],
    attack_level: float = 0.0,
) -> np.ndarray:
    """Apply the action-level attack for *condition* and return the executed action.

    Must be called AFTER the policy produces intended_action.
    The caller is responsible for updating previous_action AFTER recovery runs,
    using: intended_action.copy() if action_delay else executed_action.copy()

    Args:
        condition:       Attack condition name.
        intended_action: Action produced by the policy (pre-attack).
        previous_action: Stored action from prior step; None at step 0 (delay sentinel).
        attack_level:    Magnitude parameter (clip_value, noise_std, scale offset).

    Returns:
        executed_action (np.ndarray) — the action that will be sent to env.step().
    """
    executed_action = intended_action.copy()

    if condition == "action_noise":
        executed_action = manipulate_action(
            intended_action, "action_noise", noise_std=attack_level
        )
    elif condition == "action_scale":
        executed_action = manipulate_action(
            intended_action, "action_scale", scale=1.0 + attack_level
        )
    elif condition == "action_reversal":
        executed_action = manipulate_action(intended_action, "action_reverse")
    elif condition == "action_delay":
        executed_action = manipulate_action(
            intended_action, "action_delay", previous_action=previous_action
        )
    elif condition == "action_clipping":
        executed_action = manipulate_action(
            intended_action, "action_clipping", clip_value=attack_level
        )

    # PickAndPlace-specific (Phase 2) — negates only the gripper dim (action[3])
    elif condition == "grip_state_falsification":
        executed_action = manipulate_action(intended_action, ATTACK_GRIP_FALSIFY)

    return executed_action
