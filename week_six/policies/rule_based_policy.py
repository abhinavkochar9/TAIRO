"""
Rule-based proportional reaching controller.

Moves the end-effector toward the desired_goal using the vector difference
between achieved_goal and desired_goal, scaled by a proportional gain.

This controller is intentionally transparent and non-learned. It serves as
a strong interpretable baseline — above random but without any training.

FetchReach-v4 action space: [dx, dy, dz, gripper_ctrl] (4-D, clipped to ±1).
Gripper control is fixed at 0 since FetchReach does not require grasping.
"""

from typing import Dict
import numpy as np

# Default proportional gain. At gain=5 the controller drives ~1 m/s per
# 0.2 m of error, which reliably reaches the goal within 50 steps for
# clean observations. Reduce if actions saturate under attack.
DEFAULT_GAIN = 5.0


def rule_based_reach_policy(
    env,
    obs: Dict[str, np.ndarray],
    gain: float = DEFAULT_GAIN,
) -> np.ndarray:
    """Proportional controller: action = gain * (desired_goal - achieved_goal).

    Args:
        env:  Gymnasium environment (used for action_space bounds and shape).
        obs:  Goal-conditioned observation dict with keys
              ``achieved_goal`` and ``desired_goal``.
        gain: Proportional gain applied to the position error.

    Returns:
        Clipped action array of shape ``env.action_space.shape``.
    """
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    error = desired - achieved

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[:3] = gain * error[:3]   # position axes
    action[-1] = 0.0                # gripper — not needed for FetchReach

    return np.clip(action, env.action_space.low, env.action_space.high)
