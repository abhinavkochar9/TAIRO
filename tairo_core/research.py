"""
Bridge between the webapp and the week_six research code.

Adds week_six/ to sys.path (its modules import each other as top-level
packages: `from attacks... import`, `from config import ...`), loads the
trained SAC+HER models, and exposes a step-by-step episode generator that
mirrors evaluation/episode_runner.py semantics exactly — including the
action_delay previous-action bookkeeping and the C4 split jerk metric —
while yielding per-step data so the UI can animate the rollout live.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
WEEK_SIX = REPO_ROOT / "week_six"
if str(WEEK_SIX) not in sys.path:
    sys.path.insert(0, str(WEEK_SIX))

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

# week_six imports (resolved via the sys.path shim above)
from config import (  # noqa: E402
    ALL_CONDITIONS,
    ATTACK_LEVELS,
    SAFETY_ARM_JERK_THRESHOLD,
    SAFETY_GRIPPER_JERK_THRESHOLD,
)
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack  # noqa: E402
from policies.rule_based_policy import rule_based_reach_policy  # noqa: E402

MODELS_DIR = WEEK_SIX / "results" / "models"
DATA_DIR = WEEK_SIX / "results" / "data"
FIGURES_DIR = WEEK_SIX / "results" / "figures"

# env id -> (max steps, policies available)
ENVS = {
    "FetchReach-v4": {
        "max_steps": 150,
        "models": {"SAC+HER": MODELS_DIR / "sac_her_fetchreach_model.zip"},
        "conditions": [
            "clean", "sensor_dropout", "sensor_bias", "action_clipping",
            "action_delay", "action_reversal", "goal_spoof_immediate",
            "goal_spoof_midep",
        ],
    },
    "FetchPickAndPlace-v4": {
        "max_steps": 150,
        "models": {
            "SAC+HER (clean-trained)": MODELS_DIR / "sac_her_pickandplace_clean.zip",
            "SAC+HER (attack-randomized)": MODELS_DIR / "sac_her_pickandplace_randomized.zip",
        },
        "conditions": ALL_CONDITIONS,
    },
}

RECOVERY_OPTIONS = ["none", "v2", "v3"]


@st.cache_resource(show_spinner="Loading SAC+HER model…")
def load_model(model_path: str, env_id: str, max_steps: int):
    """Load an SB3 SAC+HER model. HER models need an env at load time."""
    from stable_baselines3 import SAC

    env = gym.make(env_id, max_episode_steps=max_steps)
    model = SAC.load(model_path, env=env)
    return model


def make_env(env_id: str, max_steps: int):
    """Plain env (no gymnasium renderer — rendering goes through tairo_core.render)."""
    return gym.make(env_id, max_episode_steps=max_steps)


def distance_to_goal(obs) -> float:
    return float(np.linalg.norm(
        np.asarray(obs["achieved_goal"]) - np.asarray(obs["desired_goal"])
    ))


def run_episode_steps(
    env,
    policy: str,                    # "rule_based" or a key into ENVS[env_id]["models"]
    model,                          # loaded SB3 model or None for rule_based
    condition: str,
    attack_level: float,
    recovery: str,                  # "none" | "v2" | "v3"
    seed: int,
    max_steps: int,
    render_fn=None,                 # callable(data) -> rgb array, or None
    render_every: int = 1,
):
    """Generator: run one episode, yielding a dict per step.

    Mirrors evaluation/episode_runner.py: sensor attack before the policy,
    action attack after, recovery on the RAW obs, split C4 jerk on
    consecutive executed actions, and the action_delay previous-action rule
    (store intended, not executed, so the lag is genuine).
    """
    if recovery == "v2":
        from recovery.recovery_v2 import maybe_apply_recovery, RecoveryState
        recovery_state = RecoveryState()
    elif recovery == "v3":
        from recovery.recovery_v3 import maybe_apply_recovery, RecoveryState
        recovery_state = RecoveryState()
    else:
        recovery_state = None

    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    previous_action: Optional[np.ndarray] = None
    prev_executed: Optional[np.ndarray] = None
    previous_obs = None
    step_distances: list = []
    bias_vector = goal_offset = object_pose_offset = None

    for t in range(max_steps):
        # -- sensor attack (before the policy sees anything) ----------------
        policy_obs, bias_vector, goal_offset, object_pose_offset = apply_sensor_attack(
            condition, obs, t, bias_vector, goal_offset,
            attack_level=attack_level, object_pose_offset=object_pose_offset,
        )

        # -- policy ----------------------------------------------------------
        if policy == "rule_based" or model is None:
            intended = rule_based_reach_policy(env, policy_obs)
        else:
            intended, _ = model.predict(policy_obs, deterministic=True)
        intended = np.asarray(intended, dtype=np.float32).copy()

        # -- action attack ----------------------------------------------------
        executed = apply_action_attack(
            condition, intended, previous_action, attack_level=attack_level
        )

        # -- recovery (sees the RAW obs — steers to the real goal) ------------
        recovery_triggered = False
        if recovery_state is not None:
            executed, recovery_triggered = maybe_apply_recovery(
                obs=obs, action=executed, prev_obs=previous_obs,
                prev_action=previous_action, step_distances=step_distances,
                step=t, env=env, state=recovery_state,
            )

        # -- C4 split jerk on consecutive executed actions ---------------------
        if prev_executed is not None:
            arm_jerk = float(np.linalg.norm(executed[:3] - prev_executed[:3]))
            grip_jerk = float(abs(executed[3] - prev_executed[3]))
        else:
            arm_jerk = grip_jerk = 0.0
        safety_violation = (
            arm_jerk > SAFETY_ARM_JERK_THRESHOLD
            or grip_jerk > SAFETY_GRIPPER_JERK_THRESHOLD
        )

        previous_obs = obs
        prev_executed = executed.copy()
        previous_action = (
            intended.copy() if condition == "action_delay" else executed.copy()
        )

        # -- environment step ---------------------------------------------------
        obs, reward, terminated, truncated, info = env.step(executed)
        total_reward += float(reward)
        dist = distance_to_goal(obs)
        step_distances.append(dist)

        frame = None
        if render_fn is not None and (t % render_every == 0 or terminated or truncated):
            frame = render_fn(env.unwrapped.data)

        yield {
            "t": t,
            "frame": frame,
            "reward": float(reward),
            "total_reward": total_reward,
            "distance": dist,
            "is_success": float(info.get("is_success", 0.0)),
            "arm_jerk": arm_jerk,
            "grip_jerk": grip_jerk,
            "safety_violation": safety_violation,
            "recovery_triggered": bool(recovery_triggered),
            "intended_norm": float(np.linalg.norm(intended)),
            "executed_norm": float(np.linalg.norm(executed)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        if terminated or truncated:
            break
