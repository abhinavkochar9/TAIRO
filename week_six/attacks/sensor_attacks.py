"""
Observation-level attack functions (Scenario 1 — Sensor Attacks).

add_sensor_noise    : Gaussian noise injected into every observation field.
shift_target        : Desired-goal spoofing / unexpected goal change.
                      Supports mid-episode onset (shift_step) and a
                      caller-supplied goal_offset held constant across steps.
apply_sensor_dropout: Zeros out entire named fields — simulates camera or
                      proprioception feed going completely dead.
apply_sensor_bias   : Constant per-dimension offset on obs["observation"]
                      sampled once per episode — simulates a miscalibrated
                      sensor that always reads high or low by a fixed amount.
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
