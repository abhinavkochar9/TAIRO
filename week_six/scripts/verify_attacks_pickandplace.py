#!/usr/bin/env python3
"""
Micro-sweep diagnostic for FetchPickAndPlace-v4 attack correctness.

For each of the 11 conditions in config.ALL_CONDITIONS, runs exactly
1 episode (seed=0) with the clean-trained PickAndPlace model and logs
the key attack signatures to verify each attack fires correctly.

Logged metrics per condition
-----------------------------
success          : True/False
final_dist       : distance_to_goal at last step
mean_action_norm : mean ||executed_action|| across episode
mean_grip_int    : mean intended action[3] (gripper) from policy
mean_grip_exec   : mean executed action[3] after attack applied
zero_frac@25     : fraction of the 25-dim obs["observation"] that is
                   exactly 0.0 at step 25 — verifies sensor/contact dropout
obj_pos_delta@25 : ||policy_obs["observation"][3:6] - obs["observation"][3:6]||
                   at step 25 — verifies object_pose_spoof offset
goal_shift@25    : ||policy_obs["desired_goal"] - obs["desired_goal"]||
                   at step 25 — verifies goal_spoof_* shift

Usage:
    conda run -n reu_robotics python3 scripts/verify_attacks_pickandplace.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import (
    ALL_CONDITIONS,
    ATTACK_LEVELS,
    MODEL_PATH_PICKANDPLACE,
    MAX_EPISODE_STEPS_PICKANDPLACE,
    SB3_AVAILABLE,
    GYM_AVAILABLE,
)
from envs.fetchpickandplace_env import make_env, distance_to_goal
from evaluation.attack_dispatch import apply_sensor_attack, apply_action_attack

# Step at which we snapshot the obs for structural diagnostics.
# Must be > GOAL_SPOOF_MIDEP_STEP (20) so mid-ep spoof is already active.
SNAPSHOT_STEP = 25


def run_diagnostic_episode(env, model, condition: str, seed: int = 0) -> dict:
    """Run one episode under *condition* and return a dict of diagnostic metrics."""
    attack_level = ATTACK_LEVELS[condition]

    obs, _info = env.reset(seed=seed)

    # Per-episode state variables (same conventions as episode_runner.py)
    bias_vector       = None
    goal_offset       = None
    object_pose_offset = None
    previous_action   = None

    # Accumulators
    action_norms_exec     = []
    gripper_intended_list = []
    gripper_executed_list = []
    total_reward          = 0.0
    success               = False

    # Snapshot at SNAPSHOT_STEP
    snap_zero_frac    = None   # fraction of obs["observation"] == 0.0
    snap_obj_delta    = None   # ||policy_obs[3:6] - obs[3:6]||  (object_pose_spoof)
    snap_goal_shift   = None   # ||policy_obs["desired_goal"] - obs["desired_goal"]||
    snap_ag_dg_dist   = None   # ||policy_obs["achieved_goal"] - policy_obs["desired_goal"]||

    for t in range(MAX_EPISODE_STEPS_PICKANDPLACE):
        # --- sensor attack fires BEFORE policy ---
        policy_obs, bias_vector, goal_offset, object_pose_offset = apply_sensor_attack(
            condition, obs, t, bias_vector, goal_offset,
            attack_level=attack_level,
            object_pose_offset=object_pose_offset,
        )

        # --- policy ---
        raw_action, _ = model.predict(policy_obs, deterministic=True)
        intended_action = np.asarray(raw_action, dtype=np.float32).copy()

        # --- action attack fires AFTER policy ---
        executed_action = apply_action_attack(
            condition, intended_action, previous_action,
            attack_level=attack_level,
        )

        # Accumulators
        action_norms_exec.append(float(np.linalg.norm(executed_action)))
        gripper_intended_list.append(float(intended_action[3]))
        gripper_executed_list.append(float(executed_action[3]))

        # Snapshot
        if t == SNAPSHOT_STEP:
            obs_arr      = policy_obs["observation"]
            raw_obs_arr  = obs["observation"]
            snap_zero_frac  = float(np.mean(obs_arr == 0.0))
            snap_obj_delta  = float(np.linalg.norm(obs_arr[3:6] - raw_obs_arr[3:6]))
            snap_goal_shift = float(np.linalg.norm(
                np.asarray(policy_obs["desired_goal"]) - np.asarray(obs["desired_goal"])
            ))
            snap_ag_dg_dist = float(np.linalg.norm(
                np.asarray(policy_obs["achieved_goal"]) - np.asarray(policy_obs["desired_goal"])
            ))

        # Update previous_action (action_delay sentinel logic from CLAUDE.md)
        previous_action = (
            intended_action.copy() if condition == "action_delay"
            else executed_action.copy()
        )

        obs, reward, terminated, truncated, _info = env.step(executed_action)
        total_reward += float(reward)
        if _info.get("is_success", False):
            success = True
        if terminated or truncated:
            break

    # Episode ended before SNAPSHOT_STEP (shouldn't happen at step 25 / 150-step budget)
    if snap_zero_frac is None:
        snap_zero_frac  = float("nan")
        snap_obj_delta  = float("nan")
        snap_goal_shift = float("nan")
        snap_ag_dg_dist = float("nan")

    return {
        "condition":       condition,
        "success":         success,
        "final_dist":      distance_to_goal(obs),
        "mean_act_norm":   float(np.mean(action_norms_exec))   if action_norms_exec else float("nan"),
        "mean_grip_int":   float(np.mean(gripper_intended_list)) if gripper_intended_list else float("nan"),
        "mean_grip_exec":  float(np.mean(gripper_executed_list)) if gripper_executed_list else float("nan"),
        "zero_frac@25":    snap_zero_frac,
        "obj_delta@25":    snap_obj_delta,
        "goal_shift@25":   snap_goal_shift,
        "ag_dg_dist@25":   snap_ag_dg_dist,
    }


# ---------------------------------------------------------------------------
# Expected-signature table (from CLAUDE.md investigation prompt)
# ---------------------------------------------------------------------------
# zero_frac thresholds (at step 25, mid-episode — some non-zeroed dims near 0
# in real episodes, so allow ±0.05 tolerance):
#   sensor_dropout:   ~1.0   (all 25 dims zeroed)
#   contact_dropout:  ~0.60  (15/25 dims: [3:9]+[11:20])
#   OLD broken mask:  ~0.68  (17/25 dims: above + [9:11] also zeroed)
#   others:           ~0.0   (no zeroing)
EXPECTED = {
    #                     zero_frac   obj_delta   goal_shift
    "clean":              (0.00,      0.00,       0.00),
    "sensor_dropout":     (1.00,      0.00,       0.00),
    "sensor_bias":        (0.00,      0.00,       0.00),
    "action_clipping":    (0.00,      0.00,       0.00),
    "action_delay":       (0.00,      0.00,       0.00),
    "action_reversal":    (0.00,      0.00,       0.00),
    "goal_spoof_immediate": (0.00,    0.00,       0.10),   # desired_goal shifted
    "goal_spoof_midep":   (0.00,      0.00,       0.10),   # shifted by step 25
    "object_pose_spoof":  (0.00,      0.10,       0.00),   # obs[3:6] shifted
    "grip_state_falsification": (0.00, 0.00,      0.00),   # action-only attack
    "contact_dropout":    (0.60,      0.00,       0.00),   # 15/25 zeros
}


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   nan  "
    return f"{v:8.4f}"


def flag_issues(rows: list, clean: dict) -> list[tuple]:
    """Return list of (condition, description) for suspicious results."""
    issues = []
    for row in rows:
        cond = row["condition"]
        if cond == "clean":
            continue

        # Generic: behavior indistinguishable from clean
        dist_diff   = abs(row["final_dist"]   - clean["final_dist"])
        zero_diff   = abs(row["zero_frac@25"] - clean["zero_frac@25"])
        obj_diff    = abs(row["obj_delta@25"] - clean["obj_delta@25"])
        goal_diff   = abs(row["goal_shift@25"] - clean["goal_shift@25"])

        all_same = (dist_diff < 0.005 and zero_diff < 0.01
                    and obj_diff < 0.005 and goal_diff < 0.005)
        if all_same and cond not in ("action_clipping",):
            # action_clipping may not change final_dist much for trained policy
            issues.append((cond, "ATTACK POSSIBLY NOT FIRING — all metrics match clean"))

        # Condition-specific
        if cond == "sensor_dropout":
            if row["zero_frac@25"] < 0.99:
                issues.append((cond,
                    f"zero_frac@25={row['zero_frac@25']:.3f} (expected ~1.0) — dropout not zeroing full obs"))

        elif cond == "contact_dropout":
            zf = row["zero_frac@25"]
            if zf > 0.65:
                issues.append((cond,
                    f"zero_frac@25={zf:.3f} > 0.65 — old broken mask still active? "
                    f"([9:11] gripper_state should NOT be zeroed, giving 15/25=0.60)"))
            elif zf < 0.50:
                issues.append((cond,
                    f"zero_frac@25={zf:.3f} < 0.50 — fewer zeros than expected (need [3:9]+[11:20])"))

        elif cond == "object_pose_spoof":
            d = row["obj_delta@25"]
            if d < 0.05:
                issues.append((cond,
                    f"obj_delta@25={d:.4f} — spoof offset appears too small; is apply_object_pose_spoof firing?"))

        elif cond in ("goal_spoof_immediate", "goal_spoof_midep"):
            gs = row["goal_shift@25"]
            if gs < 0.05:
                issues.append((cond,
                    f"goal_shift@25={gs:.4f} — desired_goal not shifted; is shift_target firing?"))

        elif cond == "grip_state_falsification":
            mi = row["mean_grip_int"]
            me = row["mean_grip_exec"]
            if not (np.isnan(mi) or np.isnan(me)):
                if abs(mi) > 0.05 and abs(me) > 0.05 and (mi * me > 0):
                    issues.append((cond,
                        f"mean_grip_int={mi:.3f} and mean_grip_exec={me:.3f} have SAME sign "
                        f"— action[3] negation not applied?"))
                if abs(mi - me) < 0.01:
                    issues.append((cond,
                        f"mean_grip_int≈mean_grip_exec={mi:.3f} — no difference detected"))

        elif cond == "sensor_bias":
            if row["zero_frac@25"] > 0.01:
                issues.append((cond,
                    "zero_frac@25 > 0 for sensor_bias — bias should offset not zero"))

    return issues


def main() -> None:
    if not GYM_AVAILABLE:
        print("[ERROR] Gymnasium Robotics not available — cannot run diagnostic.")
        sys.exit(1)
    if not SB3_AVAILABLE:
        print("[ERROR] Stable-Baselines3 not available — cannot run diagnostic.")
        sys.exit(1)

    from stable_baselines3 import SAC

    print(f"[verify] Loading clean PickAndPlace model: {MODEL_PATH_PICKANDPLACE}")
    _tmp_env = make_env(seed=0)
    model = SAC.load(MODEL_PATH_PICKANDPLACE, env=_tmp_env)
    _tmp_env.close()

    env = make_env(seed=0)
    print(f"[verify] Env: FetchPickAndPlace-v4  |  step budget: {MAX_EPISODE_STEPS_PICKANDPLACE}")
    print(f"[verify] Snapshot at step {SNAPSHOT_STEP}  |  seed=0  |  1 episode per condition\n")

    rows = []
    for condition in ALL_CONDITIONS:
        print(f"  running [{condition:<28}] ...", end="", flush=True)
        row = run_diagnostic_episode(env, model, condition, seed=0)
        rows.append(row)
        print(
            f"  success={str(row['success']):<5}  "
            f"final_dist={row['final_dist']:.4f}  "
            f"zero_frac@25={row['zero_frac@25']:.3f}"
        )
    env.close()

    clean_row = next(r for r in rows if r["condition"] == "clean")

    # -------------------------------------------------------------------------
    # Print diagnostic table
    # -------------------------------------------------------------------------
    SEP = "=" * 130
    sep = "-" * 130
    print(f"\n{SEP}")
    print("DIAGNOSTIC TABLE — FetchPickAndPlace-v4 Attack Verification (seed=0, 1 episode each)")
    print(SEP)
    hdr = (
        f"  {'Condition':<28}  {'Success':^7}  {'FinalDist':^10}  "
        f"{'MeanActNorm':^11}  {'GripInt':^8}  {'GripExec':^8}  "
        f"{'ZeroFrac@25':^11}  {'ObjDelta@25':^11}  {'GoalShift@25':^12}  {'AG-DG@25':^10}"
    )
    print(hdr)
    print(sep)

    for row in rows:
        cond = row["condition"]
        print(
            f"  {cond:<28}  "
            f"  {str(row['success']):<7}"
            f"  {row['final_dist']:>10.4f}"
            f"  {row['mean_act_norm']:>11.4f}"
            f"  {row['mean_grip_int']:>8.4f}"
            f"  {row['mean_grip_exec']:>8.4f}"
            f"  {row['zero_frac@25']:>11.4f}"
            f"  {row['obj_delta@25']:>11.4f}"
            f"  {row['goal_shift@25']:>12.4f}"
            f"  {row['ag_dg_dist@25']:>10.4f}"
        )

    # -------------------------------------------------------------------------
    # Expected-vs-actual summary
    # -------------------------------------------------------------------------
    print(f"\n{SEP}")
    print("EXPECTED vs ACTUAL — Key Attack Signatures")
    print(SEP)
    print(f"  {'Condition':<28}  {'zero_frac exp/act':^20}  {'obj_delta exp/act':^20}  {'goal_shift exp/act':^22}  Status")
    print(sep)

    for row in rows:
        cond = row["condition"]
        exp_zf, exp_od, exp_gs = EXPECTED.get(cond, (0.0, 0.0, 0.0))
        act_zf = row["zero_frac@25"]
        act_od = row["obj_delta@25"]
        act_gs = row["goal_shift@25"]

        zf_ok = abs(act_zf - exp_zf) < 0.12 or (exp_zf == 0.0 and act_zf < 0.15)
        od_ok = (exp_od == 0.0 and act_od < 0.02) or (exp_od > 0.0 and act_od > 0.05)
        gs_ok = (exp_gs == 0.0 and act_gs < 0.02) or (exp_gs > 0.0 and act_gs > 0.05)
        status = "OK" if (zf_ok and od_ok and gs_ok) else "** CHECK **"

        print(
            f"  {cond:<28}  "
            f"  {exp_zf:.2f} / {act_zf:.3f}        "
            f"  {exp_od:.2f} / {act_od:.4f}       "
            f"  {exp_gs:.2f} / {act_gs:.4f}          "
            f"  {status}"
        )

    # -------------------------------------------------------------------------
    # Flags
    # -------------------------------------------------------------------------
    print(f"\n{SEP}")
    issues = flag_issues(rows, clean_row)
    if issues:
        print(f"FLAGGED ISSUES ({len(issues)} total):")
        for cond, desc in issues:
            print(f"  [{cond}]  {desc}")
    else:
        print("No issues detected — all attack signatures match expectations.")
    print(SEP)


if __name__ == "__main__":
    main()
