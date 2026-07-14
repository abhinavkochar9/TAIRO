"""
Single-attack binary-flag Gymnasium wrapper for TAIRO PickAndPlace training
(ATTACK_AWARE_TRACK.md §7, Dr. Ho email 2026-07-12).

Distinct from training/attack_aware_wrapper.py's 3-category one-hot track:
this wrapper mixes clean episodes with exactly ONE fixed attack condition
(chosen at construction time from config.SINGLE_ATTACK_CONDITIONS) and
appends a scalar binary flag (0.0 clean / 1.0 attacked) to obs["observation"]
instead of a 4-dim one-hot category. Purpose: a per-attack learnability probe
— does the policy condition on the flag at all for a single, simple signal,
before trusting a negative result on the harder 3-category mixture.

Structurally mirrors attack_aware_wrapper.py's per-step attack-dispatch calls
(apply_sensor_attack / apply_action_attack via evaluation/attack_dispatch.py)
and the action_delay previous_action bookkeeping invariant exactly, so that
attack behavior is guaranteed identical to the rest of the benchmark.

Do not conflate with the Phase 8/9 failure-mode classifier
(evaluation/failure_mode_labeling.py) — different track, see ATTACK_AWARE_TRACK.md.

Injection point (obs["observation"], not desired_goal/achieved_goal) is the
same one confirmed safe against HER goal relabeling by the 3-category track
(ATTACK_AWARE_TRACK.md §4b) — no goal-stream condition is reachable through
this wrapper (see config.SINGLE_ATTACK_CONDITIONS), so that risk does not
even apply here, but the injection point is kept consistent regardless.

Usage:
    env = make_env(seed)
    env = SingleAttackWrapper(env, condition="action_reversal",
                               p_clean=SINGLE_ATTACK_P_CLEAN, seed=42)
    model = SAC("MultiInputPolicy", env, ...)
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    SINGLE_ATTACK_CONDITIONS,
    SINGLE_ATTACK_FLAG_DIM,
    TRAIN_ATTACK_RANGES,
)


class SingleAttackWrapper(gym.Wrapper):
    """Wraps a FetchPickAndPlace-v4 env with per-episode clean/attack sampling
    against a SINGLE fixed condition, plus a scalar ground-truth binary flag
    (0.0/1.0) appended to obs["observation"].

    Args:
        env:       Base Gymnasium environment.
        condition: Fixed attack condition for this wrapper instance. Must be
                   one of config.SINGLE_ATTACK_CONDITIONS.
        p_clean:   Probability that any given episode is attack-free.
        seed:      Optional RNG seed for reproducibility.
    """

    def __init__(
        self, env: gym.Env, condition: str, p_clean: float, seed: Optional[int] = None
    ):
        if condition not in SINGLE_ATTACK_CONDITIONS:
            raise ValueError(
                f"condition={condition!r} not in SINGLE_ATTACK_CONDITIONS="
                f"{SINGLE_ATTACK_CONDITIONS}"
            )
        super().__init__(env)
        self.condition = condition
        self.p_clean = p_clean
        self._rng = np.random.default_rng(seed)

        # Episode-level attack state (reset each episode)
        self.current_condition: str = "clean"
        self.current_magnitude: float = 0.0
        self._step_t: int = 0
        self._bias_vector: Optional[np.ndarray] = None
        self._goal_offset: Optional[np.ndarray] = None
        self._object_pose_offset: Optional[np.ndarray] = None
        self._previous_action: Optional[np.ndarray] = None

        # Expand the "observation" Box to carry the scalar binary flag.
        # achieved_goal/desired_goal are untouched (ATTACK_AWARE_TRACK.md §4b).
        base_obs_space = env.observation_space["observation"]
        flag_dim = SINGLE_ATTACK_FLAG_DIM
        new_low = np.concatenate([base_obs_space.low, np.zeros(flag_dim)])
        new_high = np.concatenate([base_obs_space.high, np.ones(flag_dim)])
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "observation": spaces.Box(
                low=new_low, high=new_high, dtype=base_obs_space.dtype
            ),
        })

    def _sample_episode_attack(self) -> None:
        """Sample clean-vs-attacked for the coming episode; attack is always
        self.condition when not clean."""
        if self._rng.random() < self.p_clean:
            self.current_condition = "clean"
            self.current_magnitude = 0.0
        else:
            self.current_condition = self.condition
            low, high = TRAIN_ATTACK_RANGES[self.condition]
            self.current_magnitude = float(self._rng.uniform(low, high))

    def _reset_episode_state(self) -> None:
        """Clear all per-episode attack state variables."""
        self._step_t = 0
        self._bias_vector = None
        self._goal_offset = None
        self._object_pose_offset = None
        self._previous_action = None

    def _append_flag(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Append the scalar ground-truth binary flag to obs["observation"]."""
        flag_value = 0.0 if self.current_condition == "clean" else 1.0
        obs = dict(obs)
        obs["observation"] = np.concatenate([
            np.asarray(obs["observation"], dtype=np.float64).ravel(),
            np.array([flag_value], dtype=np.float64),
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
        obs = self._append_flag(obs)
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
        obs = self._append_flag(obs)
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
