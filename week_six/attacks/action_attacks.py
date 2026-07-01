"""
Action-level attack functions (Scenario 2 — Actuator / Command Attacks).

manipulate_action : Applies one of several action-space perturbations.

--- PickAndPlace-specific attacks (Phase 2) ---
attack_type="grip_state_falsification"
    Negates only action[3] (the gripper open/close command) while leaving
    the (dx, dy, dz) end-effector dims [0:3] untouched.  Simulates an
    adversary that intercepts and flips only the gripper command — causing
    the gripper to open when it should close and vice versa.
"""

from typing import Optional
import numpy as np

# Supported attack type identifiers
ATTACK_NONE              = "none"
ATTACK_NOISE             = "action_noise"
ATTACK_SCALE             = "action_scale"
ATTACK_REVERSE           = "action_reverse"
ATTACK_DELAY             = "action_delay"
ATTACK_CLIPPING          = "action_clipping"
ATTACK_GRIP_FALSIFY      = "grip_state_falsification"  # PickAndPlace-specific


def manipulate_action(
    action: np.ndarray,
    attack_type: str = ATTACK_NONE,
    noise_std: float = 0.05,
    scale: float = 1.0,
    previous_action: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Apply an action-level attack and return the clipped executed action.

    Attack types:
        ``none``                      — pass-through (no modification).
        ``action_noise``              — additive Gaussian noise with std ``noise_std``.
        ``action_scale``              — multiply action by ``scale``.
        ``action_reverse``            — negate action (adversarial flip).
        ``action_delay``              — replay ``previous_action``; zeros at step 0.
        ``action_clipping``           — clip each dim to [-clip_value, clip_value].
        ``grip_state_falsification``  — negate only action[3] (gripper dim);
                                        leaves (dx,dy,dz) untouched. PickAndPlace only.

    Actions are clipped to [-1, 1] after modification.

    Args:
        action:          Intended action from the policy.
        attack_type:     One of the string constants above.
        noise_std:       Std of Gaussian noise (used by ``action_noise``).
        scale:           Multiplication factor (used by ``action_scale``).
        previous_action: Last executed action (used by ``action_delay``).
        **kwargs:        Extra keyword args (e.g. clip_value for ``action_clipping``).

    Returns:
        Clipped executed action as a float32 array.
    """
    action = np.asarray(action, dtype=np.float32).copy()

    if attack_type == ATTACK_NONE:
        executed = action
    elif attack_type == ATTACK_NOISE:
        executed = action + np.random.normal(0.0, noise_std, size=action.shape)
    elif attack_type == ATTACK_SCALE:
        executed = scale * action
    elif attack_type == ATTACK_REVERSE:
        executed = -1.0 * action
    elif attack_type == ATTACK_DELAY:
        # Step-0 guard: no previous action means no movement (delay semantics).
        if previous_action is None:
            executed = np.zeros_like(action)
        else:
            executed = np.asarray(previous_action, dtype=np.float32).copy()
    elif attack_type == ATTACK_CLIPPING:
        clip_value = kwargs.get("clip_value", 0.3)
        executed = np.clip(action, -clip_value, clip_value)
    elif attack_type == ATTACK_GRIP_FALSIFY:
        # Negate only the gripper dimension (index 3); (dx,dy,dz) unchanged.
        executed = action.copy()
        executed[3] = -executed[3]
    else:
        executed = action

    return np.clip(executed, -1.0, 1.0)
