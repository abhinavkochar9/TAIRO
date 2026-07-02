"""
Episode runner for TAIRO FetchReach-v4 benchmarks.

Provides:
    EpisodeResult — dataclass holding per-episode summary metrics.
    run_episode   — runs one episode under a given method and attack condition.

Supported method strings
------------------------
    "rule_based"              Proportional reaching controller
    "sac" / "sac_her"         SB3 model (pass model= or policy_fn=)
    "recovery_aware_sac_her"  SB3 model with recovery damping enabled

Attack conditions (Week 5 additions marked with *)
---------------------------------------------------
    Observation-level:
        sensor_noise          Gaussian noise on all fields
        target_shift          Goal spoofing (immediate onset)
        sensor_dropout   *    Zeros obs["observation"] entirely
        sensor_bias      *    Constant per-dim offset on obs["observation"]
        goal_spoof_immediate* Goal shift from step 0
        goal_spoof_midep *    Goal shift from step 20 onward

    Action-level:
        action_noise, action_scale, action_reversal, action_delay

All attack functions, observation utilities, and recovery logic are
imported from their respective modules so this file contains only
orchestration logic.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MAX_EPISODE_STEPS, SAFETY_ACTION_NORM_THRESHOLD
from envs.fetchreach_env import distance_to_goal
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack
from policies.rule_based_policy import rule_based_reach_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_smoothness(actions: List[np.ndarray]) -> float:
    """Mean step-to-step action-difference norm. Lower = smoother control."""
    if len(actions) < 2:
        return 0.0
    diffs = [np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))]
    return float(np.mean(diffs))


def _action_magnitude(actions: List[np.ndarray]) -> float:
    """Mean action norm. Large values may indicate instability or attack amplification."""
    if not actions:
        return 0.0
    return float(np.mean([np.linalg.norm(a) for a in actions]))


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Per-episode summary record. One row in the episode-level CSV."""
    method: str
    condition: str
    seed: int
    attack_level: float
    total_reward: float
    success: float           # 1.0 if is_success was ever True, else 0.0
    final_distance: float    # distance_to_goal at last step
    episode_length: int
    action_smoothness: float
    action_magnitude: float
    safety_violation: float  # 1.0 if any step exceeded action norm threshold
    recovery_used: float     # 1.0 if recovery was triggered at any step


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,
    method: str,
    seed: int,
    condition: str = "clean",
    attack_level: float = 0.0,
    model=None,
    policy_fn: Optional[Callable] = None,
    use_recovery: bool = False,
    recovery_version: str = "v3",
    target_shift_step: int = 25,
    max_steps: int = MAX_EPISODE_STEPS,
) -> Tuple[EpisodeResult, pd.DataFrame]:
    """Run one episode and return a summary plus a step-level log DataFrame.

    Policy resolution order
    -----------------------
    1. ``policy_fn`` if provided (any callable ``(env, obs) -> action``).
    2. ``method`` string dispatch:
       - ``"rule_based"``  → rule_based_reach_policy
       - ``"sac"``, ``"sac_her"``, ``"sac_plain"`` → model.predict
    3. Fallback: rule_based_reach_policy.

    Args:
        env:               Gymnasium environment (already created).
        method:            Policy identifier string.
        seed:              Random seed passed to env.reset().
        condition:         Attack condition name (see module docstring).
        attack_level:      Float parameter for the attack (e.g., noise_std).
        model:             Trained SB3 model (required for sac / sac_her methods).
        policy_fn:         Optional callable that overrides ``method`` dispatch.
        use_recovery:      If True, apply TAIRO C5 recovery damping each step.
        target_shift_step: Step at which legacy target_shift activates.

    Returns:
        Tuple of (EpisodeResult, step_log_DataFrame).
    """
    # Recovery version is encoded in the method name — derive it here.
    _use_recovery = method in {"sac_her_recovery_v2", "sac_her_recovery_v3"}
    if _use_recovery:
        if method == "sac_her_recovery_v2":
            from recovery.recovery_v2 import maybe_apply_recovery, RecoveryState
        else:
            from recovery.recovery_v3 import maybe_apply_recovery, RecoveryState
        recovery_state = RecoveryState()
    else:
        recovery_state = None

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    actions: List[np.ndarray] = []
    step_logs: List[Dict] = []
    previous_action: Optional[np.ndarray] = None
    previous_obs: Optional[Dict] = None
    step_distances: List[float] = []
    any_recovery = False
    first_recovery_step: float = float("nan")

    # Per-episode constants sampled once for attacks that require a fixed offset.
    # object_pose_offset is only used by PickAndPlace object_pose_spoof; None here.
    bias_vector: Optional[np.ndarray] = None
    goal_offset: Optional[np.ndarray] = None
    object_pose_offset: Optional[np.ndarray] = None

    for t in range(max_steps):
        # -- Observation-level attacks ----------------------------------------
        policy_obs, bias_vector, goal_offset, object_pose_offset = apply_sensor_attack(
            condition, obs, t, bias_vector, goal_offset,
            attack_level=attack_level, target_shift_step=target_shift_step,
            object_pose_offset=object_pose_offset,
        )

        # -- Policy action -------------------------------------------------------
        if policy_fn is not None:
            action = policy_fn(env, policy_obs)
        elif method in {"sac_her", "sac_her_recovery_v2", "sac_her_recovery_v3"} and model is not None:
            action, _ = model.predict(policy_obs, deterministic=True)
        else:
            raise ValueError(f"run_episode: unknown method '{method}' or model is None")

        intended_action = np.asarray(action, dtype=np.float32).copy()

        # -- Action-level attacks ------------------------------------------------
        executed_action = apply_action_attack(
            condition, intended_action, previous_action, attack_level=attack_level
        )

        # -- Recovery (TAIRO C5) -------------------------------------------------
        recovery_triggered = False
        if _use_recovery:
            executed_action, recovery_triggered = maybe_apply_recovery(
                obs=obs,                      # raw unattacked obs — recovery steers to real goal
                action=executed_action,
                prev_obs=previous_obs,
                prev_action=previous_action,
                step_distances=step_distances,
                step=t,
                env=env,
                state=recovery_state,
            )
            if recovery_triggered:
                any_recovery = True
                if np.isnan(first_recovery_step):
                    first_recovery_step = float(t)

        previous_obs = obs
        # For action_delay, store the policy's intended action so the next step
        # replays it as a genuine 1-step lag. Storing executed_action would
        # perpetuate zeros forever (the confirmed bug).
        previous_action = (
            intended_action.copy() if condition == "action_delay" else executed_action.copy()
        )
        actions.append(executed_action.copy())

        # -- Environment step ----------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(executed_action)
        total_reward += float(reward)

        current_distance = distance_to_goal(obs)
        step_distances.append(current_distance)   # feed recovery trend detector
        is_success = float(info.get("is_success", 0.0))
        safety_violation_step = float(np.linalg.norm(executed_action) > SAFETY_ACTION_NORM_THRESHOLD)

        step_logs.append({
            "method": method,
            "condition": condition,
            "seed": seed,
            "attack_level": attack_level,
            "timestep": t,
            "reward": float(reward),
            "distance_to_goal": current_distance,
            "is_success": is_success,
            "action_norm": float(np.linalg.norm(executed_action)),
            "intended_action_norm": float(np.linalg.norm(intended_action)),
            "safety_violation": safety_violation_step,
            "recovery_triggered": float(recovery_triggered),
        })

        if terminated or truncated:
            break

    step_df = pd.DataFrame(step_logs)
    step_df["steps_to_recovery"] = first_recovery_step

    result = EpisodeResult(
        method=method,
        condition=condition,
        seed=seed,
        attack_level=float(attack_level),
        total_reward=float(total_reward),
        success=float(step_df["is_success"].max() if len(step_df) else 0.0),
        final_distance=float(step_df["distance_to_goal"].iloc[-1] if len(step_df) else float("nan")),
        episode_length=int(len(step_df)),
        action_smoothness=_action_smoothness(actions),
        action_magnitude=_action_magnitude(actions),
        safety_violation=float(step_df["safety_violation"].max() if len(step_df) else 0.0),
        recovery_used=float(any_recovery),
    )

    return result, step_df
