"""
Recovery logic v2 (TAIRO C5 — Recovery capability).

Redesigned to eliminate the 100% false-positive rate of v1's action-divergence
and jerk signals on clean SAC+HER episodes (healthy policy naturally produces
large action deltas early in episodes).

Detection signals
-----------------
Signal 1 — Action norm saturation
    Healthy SAC+HER: action norm decays from ~1.67 to ~0.36 within the first
    5 steps as the controller converges. Under sensor_dropout and action_reverse
    the norm stays near 2.0 indefinitely because the controller never converges.
    Trigger: norm > ACTION_NORM_SATURATION for SATURATION_WINDOW consecutive steps.

    Motivated by: step log analysis showing clean episodes decay below 1.8 by
    step 3-4, while sensor_dropout/action_reverse stay above 1.9 for the full
    episode. action_delay and action_clipping also stay bounded and do not
    saturate at the ceiling.

Signal 2 — Insufficient progress by step PROGRESS_CHECK_STEP
    Clean episodes close >80% of initial distance within 10 steps.
    sensor_dropout makes zero progress (observation zeroed → policy sends the
    arm the wrong way). action_reverse drives the arm away from the goal
    (negative progress). Trigger: fires once at step == PROGRESS_CHECK_STEP
    if fractional progress < PROGRESS_THRESHOLD.

    Motivated by: clean step logs show distance_to_goal drops from ~0.22 to
    <0.05 by step 10. Dropout/reverse conditions show final_distance > 0.20
    across all 100 episodes with near-zero improvement through step 10.

Signal 3 — Distance trend with absolute floor (modified from v1)
    Detects sustained backward movement while still far from the goal.
    The floor (DISTANCE_FLOOR) prevents false positives on clean episodes where
    small oscillations around the goal (distance ~0.02–0.05) can appear as
    monotone increases over 5 steps.
    Trigger: distance strictly increasing for DISTANCE_TREND_WINDOW consecutive
    steps AND current distance > DISTANCE_FLOOR.

    Designed to catch: late-episode onset attacks (goal_spoof_midep activates at
    step 20 and causes sustained backward movement from a position that was
    already close to the spoofed goal but far from the real one).

Which attacks each signal is designed to catch
----------------------------------------------
  sensor_dropout      : Signal 1 (no convergence → saturated norm)
                        Signal 2 (zero progress by step 10)
  action_reverse      : Signal 1 (negated actions → opposite direction → no convergence)
                        Signal 2 (negative progress — arm moves away)
  goal_spoof_midep    : Signal 3 (late backward movement above floor)
  goal_spoof_immediate: Signal 3 (consistent backward movement above floor)
  sensor_bias         : Signal 3 (mild; fires on worst-case bias episodes)
  clean               : NO signals should fire (designed out)
  action_delay        : Possibly Signal 3 for very long delays; typically no trigger
  action_clipping     : Very unlikely to trigger any signal (constrained not corrupted)

Recovery response
-----------------
When ANY signal fires, the executed action is replaced by rule_based_reach_policy
applied to the RAW (unattacked) obs. This steers toward the true goal even when
goal observations are spoofed, because the rule-based controller reads
obs["achieved_goal"] and obs["desired_goal"] from the environment's true state,
not the attacked policy observation.

State management
----------------
RecoveryState must be instantiated once per episode and passed into every call
to maybe_apply_recovery. It tracks initial_distance and the rolling count of
consecutive high-norm steps. This avoids module-level mutable state and makes
the sweep and video scripts trivially parallelisable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from policies.rule_based_policy import rule_based_reach_policy
from envs.fetchreach_env import distance_to_goal

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
ACTION_NORM_SATURATION = 1.8   # norm threshold for Signal 1
SATURATION_WINDOW      = 3     # consecutive steps above threshold to trigger
PROGRESS_CHECK_STEP    = 10    # step at which Signal 2 fires (one-shot check)
PROGRESS_THRESHOLD     = 0.30  # fractional progress required by that step
DISTANCE_TREND_WINDOW  = 5     # consecutive increases required for Signal 3
DISTANCE_FLOOR         = 0.15  # minimum distance for Signal 3 to be relevant


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------

@dataclass
class RecoveryState:
    """Mutable state for one episode. Instantiate once per episode."""
    initial_distance: float = 0.0
    consecutive_saturation_steps: int = 0
    _progress_checked: bool = field(default=False, repr=False)

    def reset(self):
        self.initial_distance = 0.0
        self.consecutive_saturation_steps = 0
        self._progress_checked = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_apply_recovery(
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    prev_obs: Optional[Dict[str, np.ndarray]],
    prev_action: Optional[np.ndarray],
    step_distances: List[float],
    step: int,
    env,
    state: RecoveryState,
) -> Tuple[np.ndarray, bool]:
    """Detect attack-induced deviation and substitute rule-based control (v2).

    Args:
        obs:            Current (possibly attacked) observation dict.
                        Must contain ``achieved_goal`` and ``desired_goal``.
        action:         Current executed action (after any attack manipulation).
        prev_obs:       Observation from the previous step (None at step 0).
        prev_action:    Executed action from the previous step (None at step 0).
        step_distances: Running list of distance_to_goal values recorded at the
                        END of each prior step. Append to this list each step so
                        Signal 3 has a history.
        step:           Current timestep index (0-indexed).
        env:            Gymnasium environment (needed to call rule-based policy).
        state:          RecoveryState instance for this episode. Mutated in place.

    Returns:
        Tuple of:
            - final_action : ndarray — action to actually execute (original or
                             rule-based replacement).
            - triggered    : bool — True when recovery fired this step.
    """
    action = np.asarray(action, dtype=np.float32)

    # -- Initialise state on step 0 ------------------------------------------
    if step == 0:
        state.initial_distance = distance_to_goal(obs)
        state.consecutive_saturation_steps = 0
        state._progress_checked = False
        return action, False

    # -- Signal 1: action norm saturation -------------------------------------
    action_norm = float(np.linalg.norm(action))
    if action_norm > ACTION_NORM_SATURATION:
        state.consecutive_saturation_steps += 1
    else:
        state.consecutive_saturation_steps = 0

    saturation_triggered = state.consecutive_saturation_steps >= SATURATION_WINDOW

    # -- Signal 2: insufficient progress by step PROGRESS_CHECK_STEP ----------
    progress_triggered = False
    if step == PROGRESS_CHECK_STEP and not state._progress_checked:
        state._progress_checked = True
        current_dist = distance_to_goal(obs)
        if state.initial_distance > 1e-6:
            progress = (state.initial_distance - current_dist) / state.initial_distance
            if progress < PROGRESS_THRESHOLD:
                progress_triggered = True

    # -- Signal 3: distance trend with absolute floor -------------------------
    trend_triggered = False
    if len(step_distances) >= DISTANCE_TREND_WINDOW:
        window = step_distances[-DISTANCE_TREND_WINDOW:]
        current_dist_for_floor = step_distances[-1]
        monotone_increase = all(
            window[i] < window[i + 1] for i in range(DISTANCE_TREND_WINDOW - 1)
        )
        if monotone_increase and current_dist_for_floor > DISTANCE_FLOOR:
            trend_triggered = True

    triggered = saturation_triggered or progress_triggered or trend_triggered

    if triggered:
        recovery_action = rule_based_reach_policy(env, obs)
        return np.asarray(recovery_action, dtype=np.float32), True

    return action, False
