# C4 Safety Metric — Paper Changelog

**Metric:** Per-channel arm/gripper split jerk (replaces whole-vector action-norm threshold)  
**Branch:** `week-six-pickandplace`  
**Calibration date:** July 2026

---

## Threshold values and derivation method

Both thresholds were derived by the same method: **`threshold = multiplier × pooled_clean_max`**.
This is **NOT** p99 + margin. Percentile statistics were collected for descriptive reporting only;
the threshold values themselves are multiples of the empirical clean maximum, chosen to give zero
false positives on clean episodes while remaining tight enough to fire on genuine anomalies.

| Channel | Threshold | Multiplier | Pooled clean max | Method note |
|---|---|---|---|---|
| Arm (dims 0–2) | **2.800** | 1.72× | 1.630 | Statistical max-multiplier calibration |
| Gripper (dim 3) | **2.000** | 2.65× | 0.754 | Max-multiplier; coincides with action-space ceiling `\|1−(−1)\|=2.0` — functions as a hard structural bound |

Both thresholds verified at **0 false positives** on 150 clean episodes per model
(clean_2M, clean_500k, randomized_2M — 450 total clean episodes).

**Paper methodology wording:** "Thresholds were set as a multiplier on the empirical clean-episode
maximum jerk, verified to produce zero false positives on clean episodes across all three
available models."  Do not describe either threshold as p99-based.

---

## Four-condition scope-limitation list

C4 measures actuator-level physical safety (sudden large direction changes in the executed
action). Four conditions are structurally invisible to C4:

| Condition | Mechanism | Where degradation appears |
|---|---|---|
| `action_reversal` | Full negation produces smooth reversed actuation; arm_jerk stays ≤ 0.78 on every step | C1 (success_rate = 0%) |
| `action_delay` | Replayed action is a valid smooth action; arm_jerk = 0.000 throughout episode | C1/C2 |
| `sensor_dropout` | Policy receives zero observation → outputs near-zero smooth actions; arm_jerk = 0.000 | C1/C2 |
| `grip_state_falsification` | Action-space attack: negates executed `action[3]` every step. `previous_action` stores the negated value, so `\|−a − (−b)\| = \|a − b\|` — grip_jerk is algebraically identical to clean. Empirical confirmation: gsf grip_jerk max = 0.717 < clean max = 0.754; 0 of 67,050 gsf steps exceed clean p99.9 or the 2.000 threshold. | C1 (success_rate = 0%), C2 (final_distance ≫ clean) |

All four should be listed in the paper's scope-limitation table with their mechanisms.

---

## Phase B: Recovery-transition jerk diagnostic

**Finding:** Recovery-controller hand-off (transition from base SAC policy to rule-based
controller) does produce arm-jerk spikes, but their contribution to total C4 violations is
**modest and not uniformly distributed**.

### Key numbers (pooled across 4 sweeps × 2 recovery methods, all conditions)

| Metric | Value |
|---|---|
| Total recovery-trigger activation events | 26,889 |
| Activation events that produce a safety violation (transition step) | 492 (1.83%) |
| All recovery-step violations (transition + steady-state) | 6,549 |
| Transition violations as fraction of all recovery violations | 492 / 6,549 = 7.51% |
| Transition violations as fraction of ALL violations (all methods) | 492 / 11,227 = 4.38% |
| Transition violation rate vs non-transition recovery steps | 1.83% vs 0.31% — 5.9× higher |

### Per-condition impact of excluding transition steps (recovery methods, pooled)

| Condition | Original violation rate | Excl. transition steps | Delta | Material? |
|---|---|---|---|---|
| `action_delay` | 0.125 | 0.100 | −0.025 | Yes (2.5pp) |
| `grip_state_falsification` | 0.100 | 0.075 | −0.025 | Yes (2.5pp) |
| `sensor_bias` | 0.123 | 0.113 | −0.009 | Borderline |
| All others | < 0.10 | ≈ same | < 0.002 | No |

The pooled delta underestimates localized impact. Isolating the clean_2M model:

| Condition / model / method | Original | Excl. transition | Delta |
|---|---|---|---|
| action_delay / clean_2M / v2 | 0.200 | **0.000** | **−0.200** |
| grip_state_falsification / clean_2M / v3 | 0.400 | 0.200 | −0.200 |

These 20pp swings are real — but they are isolated to the clean_2M model, which has the largest
arm_jerk range (max 1.630) and is therefore most susceptible to the √8 ≈ 2.83 transition spike.

### Root cause of steady-state violations under action_delay + v3

Beyond transition artifacts, `action_delay / clean_2M / v3` has 1,380 non-transition violation
steps (action_norm = 1.414 = √2). During v3's 10-step sustained recovery window, each step's
jerk is computed as `‖rule_based[:3] − previous_action[:3]‖` where `previous_action` stores the
SAC intended action (per the action_delay update rule), not the rule-based output. When the
delayed SAC action opposes the rule-based direction, jerk = √8 ≈ 2.83 fires every step of the
recovery window, not just at hand-off. This is an interaction between the delay update rule and
the sustained recovery window — not a threshold miscalibration.

### Three options for handling transition artifacts (proposed, not implemented)

1. **Leave as-is with a documented caveat** *(lowest friction)*: The transition violations are
   real arm-jerk events — the rule-based controller does produce a large direction change when
   taking over from an opposing SAC action. Whether that specific jerk represents physical danger
   or an acceptable control handoff is a scope judgment. The paper can note that a fraction
   (≤ 7.5%) of recovery-method C4 violations occur at controller-transition timesteps.

2. **Exclude the single transition timestep from jerk computation** *(moderate change)*: Mark
   `is_transition` in the step log (first `recovery_triggered=1` after `0`) and skip jerk
   computation at that step. This would lower the recovery methods' violation rates on clean_2M
   by up to 20pp for two conditions. Requires re-running `episode_runner.py` or a
   post-processing filter; does NOT require changing any threshold constant.

3. **Apply a separate, higher threshold for the single transition step** *(principled but
   complex)*: Allow `SAFETY_ARM_JERK_THRESHOLD_TRANSITION` = e.g. 3.5 for the first recovery
   step only, calibrated to the empirical transition-spike distribution. More defensible
   scientifically (the hand-off involves intentional direction change); adds implementation
   complexity and a new constant to calibrate.

**Recommendation for follow-up session:** Option 1 unless the paper reviewers object to
recovery methods scoring below their natural floor. Option 2 is the minimum-friction fix if a
correction is warranted. Option 3 is only worth implementing if the paper argues that transition
jerks are categorically distinct from steady-state jerks.

---

## Phase C: `prev_executed` bookkeeping fix (action_delay × recovery interaction)

**Bug:** `episode_runner.py` used a single `previous_action` variable for two distinct purposes:
(1) the action_delay replay buffer (must store `intended_action` so the next step replays the
policy's genuine intent with a one-step lag), and (2) the C4 jerk comparand (must store the
last *executed* action). For action_delay + recovery, these diverge: `previous_action` stores
the SAC intended action, but `executed_action` is the rule_based override. Every step of v3's
sustained recovery window computed `‖rule_based_t − intended_{t-1}‖` instead of
`‖rule_based_t − rule_based_{t-1}‖`. Since `intended_{t-1}` often points in the opposite
direction to `rule_based_t`, jerk ≈ √8 = 2.83 fired on every step of the 10-step window.

**Root cause:** `previous_action` served dual roles (delay buffer AND jerk comparand) that
coincide for all conditions except `action_delay + recovery`.

**Fix:** Added `prev_executed: Optional[np.ndarray] = None` — a dedicated variable that always
tracks the last executed action — and changed the jerk computation to use `prev_executed`
instead of `previous_action`. `previous_action` is unchanged and continues to store
`intended_action` for `action_delay`. The fix is entirely in `episode_runner.py` (7 lines;
no changes to recovery modules, attack modules, config, or thresholds).

**Verification (smoke test, clean_2M, action_delay):**

| Method | Pre-fix step violations (clean_2M, 150 eps) | Post-fix step violations | Episode rate change |
|---|---|---|---|
| `sac_her` (no recovery) | 0 (exact zero) | ~0 (smooth delay stream) | 0.000 → 0.000 |
| `sac_her_recovery_v2` | 30 (all at transition) | 30 (all at transition) | 0.200 → 0.200 |
| `sac_her_recovery_v3` | ~1,530 (150 transition + ~1,380 steady-state) | **150 (transition only)** | 0.400 → 0.400 |

Post-fix: `0 steady-state violations` remain in any condition for action_delay × v3. All
violation steps are now at recovery-transition timesteps (first step of each recovery window)
— confirmed by checking `is_transition` flag across all 67,500 action_delay × v3 step rows.

**Why episode-level rates are unchanged:** The transition jerk at recovery handoff
(`‖rule_based_t − intended_{t-2}‖` under the corrected prev_executed tracking, where
`intended_{t-2}` is the delayed SAC action from the previous step) is a *real* physical
arm-jerk event. The same episodes that triggered sustained violations before also trigger
a transition violation after — the episode is still flagged. The fix eliminates ~1,380
spurious step-level flags per 150 episodes but does not change which episodes are flagged.

**Before/after safety_score table (all 4 sweeps, action_delay only):**

| Sweep | Method | safety_score before | safety_score after |
|---|---|---|---|
| clean_2M | sac_her | 1.000 | 1.000 |
| clean_2M | sac_her_recovery_v2 | 0.800 | **0.800** (unchanged) |
| clean_2M | sac_her_recovery_v3 | 0.600 | **0.600** (unchanged) |
| clean_500k | all methods | 1.000 | 1.000 |
| randomized_2M | sac_her | 1.000 | 1.000 |
| randomized_2M | sac_her_recovery_v2 | 0.800 | **0.800** (unchanged) |
| randomized_2M | sac_her_recovery_v3 | 0.800 | **0.800** (unchanged) |
| randomized_500k | all methods | 1.000 | 1.000 |

**Paper implication:** The fix is semantically important (step-level logs are now physically
correct) but does not change any summary-table number. The C4 violation rates reported in
the paper are unchanged. A methodology note should state that jerk is computed from
consecutive executed actions (not intended-action pairs), and that recovery-transition
steps in the clean_2M model produce genuine arm-jerk spikes that register as C4 violations.

---

## Files modified by Phase A

| File | Change |
|---|---|
| `CLAUDE.md` §11 | Added `grip_state_falsification` as 4th row in scope-limitation table; expanded mechanism explanation; added threshold derivation method clarification (max-multiplier, NOT p99+margin) |
| `config.py` | Added derivation-method comment block above calibration statistics; expanded scope-limitation comment to list all four invisible conditions with mechanisms |
| `c4_paper_changelog.md` | Created (this file) |

## Files modified by Phase C

| File | Change |
|---|---|
| `evaluation/episode_runner.py` | Added `prev_executed` variable for C4 jerk comparand; changed jerk block to use `prev_executed` instead of `previous_action`; `previous_action` semantics unchanged |
| `CLAUDE.md` §11 | Updated `action_delay` scope-limitation text: "arm_jerk = 0.000" → "arm_jerk ≈ 0 (smooth delay stream)" for base policy; noted that recovery transitions produce real jerk spikes |
| `results/data/*.csv` | Regenerated action_delay rows for all 4 sweeps (episode_results, step_logs, summary); safety_score numbers unchanged — step-level violation attribution corrected |
