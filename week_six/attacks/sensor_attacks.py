"""
Observation-level attack functions (Scenario 1 — Sensor Attacks).

add_sensor_noise        : Gaussian noise injected into every observation field.
shift_target            : Desired-goal spoofing / unexpected goal change.
                          Supports mid-episode onset (shift_step) and a
                          caller-supplied goal_offset held constant across steps.
apply_sensor_dropout    : Zeros out entire named fields — simulates camera or
                          proprioception feed going completely dead.
                          NOTE for PickAndPlace: zeroing obs["observation"] also
                          zeros object_pos (indices 3-5), obj_rel_pos (6-8), and
                          all velocity fields.  This is expected behaviour — the
                          full sensor bus is dead.
apply_sensor_bias       : Constant per-dimension offset on obs["observation"]
                          sampled once per episode — simulates a miscalibrated
                          sensor that always reads high or low by a fixed amount.
                          Shape-agnostic: works on both FetchReach (10-dim) and
                          FetchPickAndPlace (25-dim) observations.

--- PickAndPlace-specific attacks (Phase 2) ---
apply_object_pose_spoof : Corrupts only the object-position field (obs[3:6]) with
                          a constant per-episode offset.  Gripper / proprioceptive
                          fields are untouched.  Mirrors apply_sensor_bias's
                          sample-once pattern.  Also corrupts achieved_goal since
                          achieved_goal == object_pos in FetchPickAndPlace-v4.
apply_contact_dropout   : Zeros only the object-relative observation fields
                          (object_pos, obj_rel_pos, velocities), leaving gripper
                          fields intact.  Partial version of apply_sensor_dropout.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def add_sensor_noise(
    obs: Dict[str, np.ndarray],
    noise_std: float,
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise to every field of a goal-conditioned observation dict.

    Args:
        obs:       Raw observation dict from FetchReach-v4.
        noise_std: Standard deviation of the zero-mean Gaussian noise.

    Returns:
        New dict with independent noise applied to each array field.
    """
    attacked: Dict[str, np.ndarray] = {}
    for key, value in obs.items():
        arr = np.asarray(value).copy()
        attacked[key] = arr + np.random.normal(loc=0.0, scale=noise_std, size=arr.shape)
    return attacked


def shift_target(
    obs: Dict[str, np.ndarray],
    shift_scale: float = 0.03,
    step: int = 0,
    shift_step: Optional[int] = None,
    goal_offset: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Shift the desired_goal field to simulate Man-in-the-Middle goal spoofing.

    Supports two modes:
    - Immediate (shift_step=None): offset applied from step 0 onward.
    - Mid-episode (shift_step=N): obs returned unmodified before step N,
      offset applied from step N onward.

    The offset is sampled once per episode. Pass goal_offset on subsequent
    steps to hold it constant; the same vector is returned each call.

    Args:
        obs:         Raw observation dict from FetchReach-v4.
        shift_scale: Max per-axis uniform shift magnitude in metres.
        step:        Current timestep within the episode (0-indexed).
        shift_step:  Step at which the attack activates. None = always active.
        goal_offset: Pre-sampled offset vector. Sampled here if None.
        seed:        RNG seed used only when goal_offset must be sampled.

    Returns:
        Tuple of (attacked_obs, goal_offset_used).
        goal_offset_used is None when the attack has not yet activated.
    """
    attacked = {key: np.asarray(value).copy() for key, value in obs.items()}

    if shift_step is not None and step < shift_step:
        return attacked, None

    if goal_offset is None:
        rng = np.random.default_rng(seed)
        goal_offset = rng.uniform(
            -shift_scale, shift_scale, size=attacked["desired_goal"].shape
        ).astype(np.float32)

    attacked["desired_goal"] = attacked["desired_goal"] + goal_offset
    return attacked, goal_offset


def apply_sensor_dropout(
    obs: Dict[str, np.ndarray],
    fields: List[str],
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Zero out entire named fields of the observation dict.

    Simulates a sensor blackout — e.g. a camera or proprioception feed that
    goes completely dead, returning all zeros instead of real measurements.

    Args:
        obs:    Raw observation dict from FetchReach-v4.
        fields: List of dict keys to zero out, e.g. ["observation"].
        seed:   Unused; accepted for API consistency with other attacks.

    Returns:
        New dict with the specified fields replaced by zero arrays.
    """
    attacked = {key: np.asarray(value).copy() for key, value in obs.items()}
    for field in fields:
        if field in attacked:
            attacked[field] = np.zeros_like(attacked[field])
    return attacked


def apply_sensor_bias(
    obs: Dict[str, np.ndarray],
    magnitude: float,
    bias_vector: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Add a constant per-dimension offset to obs["observation"].

    The offset is sampled once per episode from Uniform[-magnitude, magnitude]
    and held constant across all steps. Pass bias_vector on subsequent steps
    to reuse the same vector without resampling.

    Simulates a miscalibrated sensor that consistently reads high or low by a
    fixed amount — a systematic rather than random error.

    Args:
        obs:         Raw observation dict from FetchReach-v4.
        magnitude:   Half-range of the uniform bias distribution.
        bias_vector: Pre-sampled bias. Sampled here if None.
        seed:        RNG seed used only when bias_vector must be sampled.

    Returns:
        Tuple of (attacked_obs, bias_vector_used).
    """
    attacked = {key: np.asarray(value).copy() for key, value in obs.items()}

    if bias_vector is None:
        rng = np.random.default_rng(seed)
        bias_vector = rng.uniform(
            -magnitude, magnitude, size=attacked["observation"].shape
        ).astype(np.float32)

    attacked["observation"] = attacked["observation"] + bias_vector
    return attacked, bias_vector


# ---------------------------------------------------------------------------
# PickAndPlace-specific sensor attacks (Phase 2)
# ---------------------------------------------------------------------------

# Slice constant for the 25-dim FetchPickAndPlace-v4 observation vector.
# Only object_pos is needed by name (apply_object_pose_spoof).
# contact_dropout uses literal [3:9] and [11:20] for readability.
# Verified against MujocoFetchPickAndPlaceEnv in gymnasium-robotics.
_OBJ_POS = slice(3, 6)   # object_pos — also == achieved_goal


def apply_object_pose_spoof(
    obs: Dict[str, np.ndarray],
    magnitude: float,
    offset: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Corrupt only object-position fields in obs["observation"] (indices 3-5).

    Simulates a compromised object-tracking sensor (e.g. vision pipeline
    reporting wrong object location) while leaving gripper proprioception
    intact.  The offset is sampled once per episode and held constant.

    IMPORTANT: In FetchPickAndPlace-v4, achieved_goal == object_pos.  This
    attack therefore also corrupts the achieved_goal key so the policy
    perceives a false grasp state.

    Args:
        obs:       Raw observation dict from FetchPickAndPlace-v4.
        magnitude: Half-range of uniform offset distribution (metres).
        offset:    Pre-sampled 3-dim offset; sampled here if None.
        seed:      RNG seed used only when offset must be sampled.

    Returns:
        Tuple of (attacked_obs, offset_used).
    """
    attacked = {key: np.asarray(value).copy() for key, value in obs.items()}

    if offset is None:
        rng = np.random.default_rng(seed)
        offset = rng.uniform(-magnitude, magnitude, size=(3,)).astype(np.float32)

    attacked["observation"][_OBJ_POS] = attacked["observation"][_OBJ_POS] + offset
    # achieved_goal == object_pos — corrupt it consistently so the policy sees
    # a coherent (though false) world state.
    attacked["achieved_goal"] = attacked["achieved_goal"] + offset
    return attacked, offset


def apply_contact_dropout(
    obs: Dict[str, np.ndarray],
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Zero only object-tracking fields in obs["observation"].

    Simulates loss of the external object sensor (camera / lidar) while
    keeping all gripper joint-encoder proprioception intact — the robot
    can no longer see the object, but can still feel whether its fingers
    closed on something.

    Zeroed ranges (contiguous, non-overlapping):
      [3:9]  — object_pos (3) + obj_rel_pos (3)
      [11:20] — object_rot (3) + object_velp (3) + object_velr (3)

    Preserved (gripper proprioception):
      [0:3]  — gripper_pos
      [9:11] — gripper_state (finger widths) ← joint-encoder, NOT camera
      [20:25] — gripper_velp (3) + gripper_fingers_vel (2)

    The gap at [9:11] is intentional: finger-width sensors are joint encoders
    on the gripper actuators, independent of any external object-tracking
    sensor.  Zeroing them would confuse "camera died" with "gripper broke."

    Args:
        obs:  Raw observation dict from FetchPickAndPlace-v4.
        seed: Unused; accepted for API consistency with other attack fns.

    Returns:
        New dict with object-tracking fields zeroed; obs is never mutated.
    """
    attacked = {key: np.asarray(value).copy() for key, value in obs.items()}
    attacked["observation"][3:9]  = 0.0   # object_pos + obj_rel_pos
    attacked["observation"][11:20] = 0.0  # object_rot + object_velp + object_velr
    return attacked
