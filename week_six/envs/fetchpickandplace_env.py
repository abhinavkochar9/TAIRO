"""
FetchPickAndPlace-v4 environment factory and observation utilities.

Mirrors the interface of envs/fetchreach_env.py (make_env / flatten_goal_obs /
distance_to_goal) so all sweep, recording, and training code can swap env
factories without logic changes.

Verified obs layout (25-dim observation vector, gymnasium-robotics):
  [0:3]   gripper_pos          — end-effector Cartesian position
  [3:6]   object_pos           — object Cartesian position  ← == achieved_goal
  [6:9]   obj_rel_pos          — object_pos - gripper_pos
  [9:11]  gripper_state        — finger widths (proprioception)
  [11:14] object_rot           — object Euler angles
  [14:17] object_velp          — object translational velocity
  [17:20] object_velr          — object rotational velocity
  [20:23] gripper_velp         — gripper translational velocity
  [23:25] gripper_fingers_vel  — finger velocities

Key difference from FetchReach-v4:
  • achieved_goal = object position (not gripper position)
  • observation is 25-dim (FetchReach is 10-dim)
  • action[3] is the gripper open/close command (same 4-dim action space)

Episode length flag: default max_episode_steps is 50 — identical to
FetchReach.  This may be too tight for pick-and-place (approach → grasp →
lift → transport → place).  MAX_EPISODE_STEPS_PICKANDPLACE is exposed in
config.py and passed to gym.make() here; increase it before training if
convergence is slow.
"""

from typing import Dict
import numpy as np
from config import ENV_ID_PICKANDPLACE, GYM_AVAILABLE, MAX_EPISODE_STEPS_PICKANDPLACE

# Slice indices for the 25-dim observation vector (verified against
# MujocoFetchPickAndPlaceEnv in gymnasium-robotics).
OBS_GRIPPER_POS   = slice(0, 3)
OBS_OBJECT_POS    = slice(3, 6)   # == achieved_goal
OBS_OBJ_REL_POS   = slice(6, 9)
OBS_GRIPPER_STATE = slice(9, 11)
OBS_OBJECT_ROT    = slice(11, 14)
OBS_OBJECT_VELP   = slice(14, 17)
OBS_OBJECT_VELR   = slice(17, 20)
OBS_GRIPPER_VELP  = slice(20, 23)
OBS_FINGERS_VEL   = slice(23, 25)

# Indices of object-tracking fields (position + relative pos + velocities).
# Used by apply_contact_dropout to zero only object-relevant data.
OBS_OBJECT_FIELDS_SLICES = [
    OBS_OBJECT_POS,
    OBS_OBJ_REL_POS,
    OBS_OBJECT_ROT,
    OBS_OBJECT_VELP,
    OBS_OBJECT_VELR,
]


def make_env(seed: int = 0, rgb_mode: bool = False):
    """Create and return a seeded FetchPickAndPlace-v4 Gymnasium environment.

    Args:
        seed:     Random seed passed to env.reset().
        rgb_mode: If True, passes render_mode="rgb_array" for RecordVideo.
    Returns:
        Gymnasium environment instance.
    Raises:
        RuntimeError: If gymnasium_robotics is not installed.
    """
    if not GYM_AVAILABLE:
        raise RuntimeError(
            "Gymnasium Robotics is not available. "
            "Install with: pip install gymnasium gymnasium-robotics mujoco"
        )
    import gymnasium as gym
    render_mode = "rgb_array" if rgb_mode else None
    env = gym.make(
        ENV_ID_PICKANDPLACE,
        max_episode_steps=MAX_EPISODE_STEPS_PICKANDPLACE,
        render_mode=render_mode,
    )
    # Do NOT call env.reset() here — SB3/DummyVecEnv handles the first reset
    # internally.  Callers that need a specific seed (eval, recording) call
    # env.reset(seed=seed) themselves after make_env() returns.
    return env


def flatten_goal_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate observation, achieved_goal, and desired_goal into one vector.

    Args:
        obs: Goal-conditioned observation dict from FetchPickAndPlace-v4.
    Returns:
        1-D float32 array of length 31 (25 + 3 + 3).
    """
    return np.concatenate([
        np.asarray(obs["observation"], dtype=np.float32).ravel(),
        np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
        np.asarray(obs["desired_goal"],  dtype=np.float32).ravel(),
    ])


def distance_to_goal(obs: Dict[str, np.ndarray]) -> float:
    """Euclidean distance between achieved_goal (object pos) and desired_goal.

    Args:
        obs: Goal-conditioned observation dict from FetchPickAndPlace-v4.
    Returns:
        Scalar distance in metres.
    """
    return float(np.linalg.norm(
        np.asarray(obs["achieved_goal"]) - np.asarray(obs["desired_goal"])
    ))
