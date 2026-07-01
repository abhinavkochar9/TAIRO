"""
Attack-domain-randomization Gymnasium wrapper for TAIRO PickAndPlace training.

On each episode reset() this wrapper:
  1. With probability p_clean → clean episode (no attack).
  2. Otherwise, uniformly samples one of the 9 non-clean conditions and a
     magnitude within its TRAIN_ATTACK_RANGES entry.

Per-step, the wrapper intercepts observations (via step/reset returns) and
actions (via step inputs) to apply the sampled attack — using the same
apply_sensor_attack / apply_action_attack dispatch functions as evaluation,
so training and eval are guaranteed to use identical attack implementations.

Usage:
    env = make_env(seed)
    env = AttackRandomizationWrapper(env, p_clean=0.2, seed=42)
    model = SAC("MultiInputPolicy", env, ...)

The wrapper exposes the current episode's condition and magnitude as
attributes (.current_condition, .current_magnitude) for logging.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from config import TRAIN_ATTACK_RANGES, ATTACK_LEVELS

# All non-clean conditions that have training ranges defined.
_NON_CLEAN_CONDITIONS = [c for c in TRAIN_ATTACK_RANGES if c != "clean"]


class AttackRandomizationWrapper(gym.Wrapper):
    """Wraps a FetchPickAndPlace-v4 env with per-episode attack randomization.

    Args:
        env:     Base Gymnasium environment.
        p_clean: Probability that any given episode is attack-free (default 0.2).
        seed:    Optional RNG seed for reproducibility.
    """

    def __init__(self, env: gym.Env, p_clean: float = 0.2, seed: Optional[int] = None):
        super().__init__(env)
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

    def _sample_episode_attack(self) -> None:
        """Sample condition + magnitude for the coming episode."""
        if self._rng.random() < self.p_clean:
            self.current_condition = "clean"
            self.current_magnitude = 0.0
        else:
            self.current_condition = str(
                self._rng.choice(_NON_CLEAN_CONDITIONS)
            )
            low, high = TRAIN_ATTACK_RANGES[self.current_condition]
            self.current_magnitude = float(self._rng.uniform(low, high))

    def _reset_episode_state(self) -> None:
        """Clear all per-episode attack state variables."""
        self._step_t = 0
        self._bias_vector = None
        self._goal_offset = None
        self._object_pose_offset = None
        self._previous_action = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._sample_episode_attack()
        self._reset_episode_state()
        # Apply sensor attack to the initial observation (t=0).
        obs, _ = self._apply_sensor(obs)
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

        # Apply sensor attack to the returned observation.
        obs, _ = self._apply_sensor(obs)
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
