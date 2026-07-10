"""
Recovery logic v3 (TAIRO C5 — Recovery capability).

Detection signals are identical to v2 — thresholds are unchanged. The only
change from v2 is the recovery *response* behavior: instead of handing control
back to SAC+HER after every single recovery step, v3 holds recovery for a
sustained window and only exits early when the arm demonstrates genuine progress.

Why sustained recovery is needed
---------------------------------
v2 has an oscillation problem under directional attacks such as action_reverse.
The sequence each step is:
  1. v2 detection fires → rule-based replanning moves arm toward goal
  2. Next step: SAC+HER runs again → attack negates the action → arm moves away
  3. Detection fires again → repeat

The arm never makes net progress because the attack undoes each recovery step
before the arm can close distance. The underlying issue is that v2's detection
says "something is wrong" but then immediately trusts the policy again —
even though nothing about the attacking condition has changed.

Why the exit condition is progress-based rather than attack-specific
---------------------------------------------------------------------
The fix makes no assumptions about which attack is active or what mechanism is
causing failure. Instead it asks a single question that applies universally:
  "Is the arm actually getting closer to the goal while we are in recovery mode?"
If yes (PROGRESS_EXIT_WINDOW consecutive improving steps), we can safely return
control to the policy — the system is demonstrably converging. If no, we keep
the rule-based controller running until the countdown expires.

This means v3 will behave correctly whether the underlying cause is action
reversal, sensor dropout, or any future attack that produces persistent failure
— without needing to know which attack is active.

Sustained recovery behavior
---------------------------
When any detection signal fires:
  - recovery_steps_remaining is set to SUSTAINED_RECOVERY_STEPS (if not already
    counting down) and the rule-based action is applied.
  - Each subsequent step: if recovery_steps_remaining > 0, continue applying
    rule-based recovery, decrement the counter.
  - Early exit: if distance has decreased for PROGRESS_EXIT_WINDOW consecutive
    steps while in recovery, set recovery_steps_remaining = 0 immediately.
  - When recovery_steps_remaining reaches 0 and no signal fires, normal
    (policy) control resumes.

Detection signals (unchanged from v2)
--------------------------------------
Signal 1 — Action norm saturation:
    norm > ACTION_NORM_SATURATION for SATURATION_WINDOW consecutive steps.
Signal 2 — Insufficient progress by step PROGRESS_CHECK_STEP:
    fractional progress < PROGRESS_THRESHOLD at step PROGRESS_CHECK_STEP.
Signal 3 — Distance trend with absolute floor:
    distance strictly increasing for DISTANCE_TREND_WINDOW steps AND
    current distance > DISTANCE_FLOOR.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from policies.rule_based_policy import rule_based_reach_policy
from envs.fetchreach_env import distance_to_goal

# ---------------------------------------------------------------------------
# Detection thresholds — identical to v2, do not change
# ---------------------------------------------------------------------------
ACTION_NORM_SATURATION = 1.8
SATURATION_WINDOW      = 3
PROGRESS_CHECK_STEP    = 10
PROGRESS_THRESHOLD     = 0.30
DISTANCE_TREND_WINDOW  = 5
DISTANCE_FLOOR         = 0.15

# ---------------------------------------------------------------------------
# Sustained recovery constants
# ---------------------------------------------------------------------------
SUSTAINED_RECOVERY_STEPS = 10   # steps to hold recovery once triggered
PROGRESS_EXIT_WINDOW     = 3    # consecutive improving steps to exit early


# ---------------------------------------------------------------------------
# Per-episode state
# ---------------------------------------------------------------------------

@dataclass
class RecoveryState:
    """Mutable state for one episode. Instantiate once per episode."""
    initial_distance: float = 0.0
    consecutive_saturation_steps: int = 0
    recovery_steps_remaining: int = 0
    consecutive_improving_steps: int = 0
    _progress_checked: bool = field(default=False, repr=False)

    def reset(self):
        self.initial_distance = 0.0
        self.consecutive_saturation_steps = 0
        self.recovery_steps_remaining = 0
        self.consecutive_improving_steps = 0
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
    """Detect attack-induced deviation and apply sustained rule-based control (v3).

    Detection signals are identical to v2. The response differs: once triggered,
    recovery holds for SUSTAINED_RECOVERY_STEPS steps and only exits early if
    the arm shows PROGRESS_EXIT_WINDOW consecutive improving steps.

    Args:
        obs:            Current (possibly attacked) observation dict.
        action:         Current executed action (after any attack manipulation).
        prev_obs:       Observation from the previous step (None at step 0).
        prev_action:    Executed action from the previous step (None at step 0).
        step_distances: Running list of distance_to_goal values at end of each
                        prior step. Append after env.step() each iteration.
        step:           Current timestep index (0-indexed).
        env:            Gymnasium environment (needed to call rule-based policy).
        state:          RecoveryState instance for this episode. Mutated in place.

    Returns:
        Tuple of:
            - final_action : ndarray — action to execute (original or recovery).
            - triggered    : bool — True when recovery is active this step.
    """
    action = np.asarray(action, dtype=np.float32)

    # -- Initialise state on step 0 ------------------------------------------
    if step == 0:
        state.initial_distance = distance_to_goal(obs)
        state.consecutive_saturation_steps = 0
        state.recovery_steps_remaining = 0
        state.consecutive_improving_steps = 0
        state._progress_checked = False
        return action, False

    # -------------------------------------------------------------------------
    # Run v2 detection signals (thresholds unchanged)
    # -------------------------------------------------------------------------

    # Signal 1: action norm saturation
    action_norm = float(np.linalg.norm(action))
    if action_norm > ACTION_NORM_SATURATION:
        state.consecutive_saturation_steps += 1
    else:
        state.consecutive_saturation_steps = 0

    saturation_triggered = state.consecutive_saturation_steps >= SATURATION_WINDOW

    # Signal 2: insufficient progress by PROGRESS_CHECK_STEP
    progress_triggered = False
    if step == PROGRESS_CHECK_STEP and not state._progress_checked:
        state._progress_checked = True
        current_dist = distance_to_goal(obs)
        if state.initial_distance > 1e-6:
            progress = (state.initial_distance - current_dist) / state.initial_distance
            if progress < PROGRESS_THRESHOLD:
                progress_triggered = True

    # Signal 3: distance trend with absolute floor
    trend_triggered = False
    if len(step_distances) >= DISTANCE_TREND_WINDOW:
        window = step_distances[-DISTANCE_TREND_WINDOW:]
        current_dist_for_floor = step_distances[-1]
        monotone_increase = all(
            window[i] < window[i + 1] for i in range(DISTANCE_TREND_WINDOW - 1)
        )
        if monotone_increase and current_dist_for_floor > DISTANCE_FLOOR:
            trend_triggered = True

    any_signal = saturation_triggered or progress_triggered or trend_triggered

    # -------------------------------------------------------------------------
    # Sustained recovery response
    # -------------------------------------------------------------------------

    # Start countdown when a signal fires and we are not already in recovery
    if any_signal and state.recovery_steps_remaining == 0:
        state.recovery_steps_remaining = SUSTAINED_RECOVERY_STEPS

    in_recovery = any_signal or state.recovery_steps_remaining > 0

    if in_recovery:
        recovery_action = rule_based_reach_policy(env, obs)
        recovery_action = np.asarray(recovery_action, dtype=np.float32)

        # Track whether the arm is making progress during recovery
        if len(step_distances) >= 2:
            if step_distances[-1] < step_distances[-2]:
                state.consecutive_improving_steps += 1
            else:
                state.consecutive_improving_steps = 0

        # Early exit: arm is genuinely converging — hand control back now
        if state.consecutive_improving_steps >= PROGRESS_EXIT_WINDOW:
            state.recovery_steps_remaining = 0
            state.consecutive_improving_steps = 0
        elif state.recovery_steps_remaining > 0:
            state.recovery_steps_remaining -= 1

        return recovery_action, True

    return action, False
