"""
Ground-truth attack-category-aware Gymnasium wrapper for TAIRO PickAndPlace
training (ATTACK_AWARE_TRACK.md, Step 2).

Structurally mirrors training/attack_randomization_wrapper.py (same per-episode
sampling scheme, same apply_sensor_attack/apply_action_attack dispatch), with
one addition: the ground-truth 3-category flag (clean/action/sensor/goal, per
ATTACK_AWARE_TRACK.md §4a) for whichever condition is active this episode is
one-hot encoded and appended to obs["observation"] every step.

Do not conflate with the Phase 8/9 failure-mode classifier
(evaluation/failure_mode_labeling.py) — different track, see ATTACK_AWARE_TRACK.md.

Injection point (obs["observation"], not desired_goal/achieved_goal) is
confirmed safe against HER goal relabeling — see ATTACK_AWARE_TRACK.md §4b.

Usage:
    env = make_env(seed)
    env = AttackAwareWrapper(env, p_clean=ATTACK_AWARE_P_CLEAN, seed=42)
    model = SAC("MultiInputPolicy", env, ...)
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    ATTACK_CATEGORIES,
    ATTACK_CATEGORY_FLAG_DIM,
    ATTACK_CATEGORY_MAP,
    TRAIN_ATTACK_RANGES,
)

# All non-clean conditions that have training ranges defined (same set used
# by AttackRandomizationWrapper).
_NON_CLEAN_CONDITIONS = [c for c in TRAIN_ATTACK_RANGES if c != "clean"]

# Precompute one-hot rows once, indexed by category name.
_CATEGORY_ONEHOT = {
    cat: np.eye(ATTACK_CATEGORY_FLAG_DIM, dtype=np.float64)[i]
    for i, cat in enumerate(ATTACK_CATEGORIES)
}


class AttackAwareWrapper(gym.Wrapper):
    """Wraps a FetchPickAndPlace-v4 env with per-episode attack randomization
    AND a ground-truth attack-category flag appended to obs["observation"].

    Args:
        env:     Base Gymnasium environment.
        p_clean: Probability that any given episode is attack-free (default
                 from config.ATTACK_AWARE_P_CLEAN).
        seed:    Optional RNG seed for reproducibility.
    """

    def __init__(self, env: gym.Env, p_clean: float, seed: Optional[int] = None):
        super().__init__(env)
        self.p_clean = p_clean
        self._rng = np.random.default_rng(seed)

        # Episode-level attack state (reset each episode)
        self.current_condition: str = "clean"
        self.current_magnitude: float = 0.0
        self.current_category: str = "clean"
        self._step_t: int = 0
        self._bias_vector: Optional[np.ndarray] = None
        self._goal_offset: Optional[np.ndarray] = None
        self._object_pose_offset: Optional[np.ndarray] = None
        self._previous_action: Optional[np.ndarray] = None

        # Expand the "observation" Box to carry the one-hot category flag.
        # achieved_goal/desired_goal are untouched (ATTACK_AWARE_TRACK.md §4b).
        base_obs_space = env.observation_space["observation"]
        flag_dim = ATTACK_CATEGORY_FLAG_DIM
        new_low = np.concatenate([base_obs_space.low, np.zeros(flag_dim)])
        new_high = np.concatenate([base_obs_space.high, np.ones(flag_dim)])
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "observation": spaces.Box(
                low=new_low, high=new_high, dtype=base_obs_space.dtype
            ),
        })

    def _sample_episode_attack(self) -> None:
        """Sample condition + magnitude for the coming episode; derive category."""
        if self._rng.random() < self.p_clean:
            self.current_condition = "clean"
            self.current_magnitude = 0.0
        else:
            self.current_condition = str(
                self._rng.choice(_NON_CLEAN_CONDITIONS)
            )
            low, high = TRAIN_ATTACK_RANGES[self.current_condition]
            self.current_magnitude = float(self._rng.uniform(low, high))
        self.current_category = ATTACK_CATEGORY_MAP[self.current_condition]

    def _reset_episode_state(self) -> None:
        """Clear all per-episode attack state variables."""
        self._step_t = 0
        self._bias_vector = None
        self._goal_offset = None
        self._object_pose_offset = None
        self._previous_action = None

    def _append_category_flag(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Append the ground-truth one-hot category flag to obs["observation"]."""
        flag = _CATEGORY_ONEHOT[self.current_category]
        obs = dict(obs)
        obs["observation"] = np.concatenate([
            np.asarray(obs["observation"], dtype=np.float64).ravel(),
            flag,
        ])
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._sample_episode_attack()
        self._reset_episode_state()
        # Apply sensor attack to the initial observation (t=0), then flag it.
        obs, _ = self._apply_sensor(obs)
        obs = self._append_category_flag(obs)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        from evaluation.attack_dispatch import apply_action_attack

        # Apply action attack before sending to env.
        action = np.asarray(action, dtype=np.float32)
        executed_action = apply_action_attack(
            self.current_condition,
            action,
            self._previous_action,
            attack_level=self.current_magnitude,
        )

        # Update previous_action following the action_delay invariant:
        # store intended action so next step replays the genuine policy choice.
        if self.current_condition == "action_delay":
            self._previous_action = action.copy()
        else:
            self._previous_action = executed_action.copy()

        obs, reward, terminated, truncated, info = self.env.step(executed_action)
        self._step_t += 1

        # Apply sensor attack to the returned observation, then flag it.
        obs, _ = self._apply_sensor(obs)
        obs = self._append_category_flag(obs)
        return obs, reward, terminated, truncated, info

    def _apply_sensor(
        self, obs: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], None]:
        """Run apply_sensor_attack and update per-episode state variables."""
        from evaluation.attack_dispatch import apply_sensor_attack

        policy_obs, self._bias_vector, self._goal_offset, self._object_pose_offset = (
            apply_sensor_attack(
                self.current_condition,
                obs,
                self._step_t,
                self._bias_vector,
                self._goal_offset,
                attack_level=self.current_magnitude,
                object_pose_offset=self._object_pose_offset,
            )
        )
        return policy_obs, None
