# ATTACK_AWARE_TRACK.md — Attack-Aware Policy (Dr. Ho Proposal)

**Status:** Design phase — Step 1 (diagnostic) not yet run.
**Branch:** To be developed on a separate branch from `main`/Week 6 trunk. Do not merge until
the existing 4 models, sweeps, and C4-fixed pipeline are confirmed unaffected.
**Relationship to CLAUDE.md:** This file is the single source of truth for this track. CLAUDE.md
carries only a pointer to it (see bottom of this file for the exact text added there). Do not
duplicate this content back into CLAUDE.md as it evolves — update here only.

---

## 1. Background — where this came from

Originating from Dr. Ho's email (2026-07-08), responding to an open architectural question about
his earlier observation-wrapper suggestion: ground-truth labels for what attack is active can't
exist at deployment, so how does an attack-aware policy get that signal at all?

Dr. Ho's answer, distilled to three points:

1. **Category scheme, staged rollout.** Start with a 3-category flag (action / sensor / goal —
   not clean/attacked binary, and not the full 11-condition taxonomy) added to the SAC+HER
   observation. Expect the 3-category version to underperform a clean-only baseline but outperform
   a policy with no attack awareness at all. The 11-label version may eventually learn more
   tailored responses, but is explicitly a later step, not this one.
2. **Training vs. deployment observation consistency.** If the policy is trained with an
   attack-category flag in its observation, deployment must supply that flag too — via a
   separate classifier (input: observation history + action history; output: clean vs.
   3-category vs. eventually 11-class), not the ground truth.
3. **Sequencing:** train the RL with **ground-truth** flags first (this establishes an upper
   bound before classifier noise enters the picture), while building the deployment classifier
   in parallel — not sequentially. Ground truth gets swapped for classifier predictions in a
   later phase to measure the resulting performance degradation.

Recommended order, per Dr. Ho: (1) train RL with ground-truth flags, (2) build the classifier
dataset and train it in parallel while (1) runs, (3) later, splice classifier predictions in for
(1) and measure the gap.

**Decision:** proceed as Dr. Ho describes it — he has more domain expertise here than the prior
smoke-test-first instinct this track started from. No smoke-test gate before the full run; see
Step 2 below.

---

## 2. Explicitly NOT the same as Phase 8/9 (findings.md / todo.md / CLAUDE.md §14)

This track introduces a **second, unrelated classifier**. Do not conflate the two:

| | Phase 8/9 (existing, findings.md/todo.md/CLAUDE.md §14) | This track (new) |
|---|---|---|
| Predicts | *How* an episode failed (7-class failure-mode taxonomy: `never_reached_object`, `wrong_direction`, etc.) | *What attack* is currently active (3-category, later 11-class) |
| Consumes | Full-episode aggregates (Phase 8, post-hoc) or rolling/causal features (Phase 9, not started) | Observation history + action history (rolling, causal — deployment-time) |
| Feeds | Recovery controller redesign (v2/v3 successor) | The SAC+HER policy itself, as an observation input |
| Ground truth source | Rule-based labeling function (`evaluation/failure_mode_labeling.py`) | Attack `condition` (already logged, already forbidden as a classifier *input* elsewhere — here it legitimately *is* the label target for this specific classifier) |
| Status | Phase 8 done (macro-F1 0.622); Phase 9 not started | Not started |

**Naming note:** to avoid the two threads getting numbered into the same sequence, this track
uses **Step 1 / Step 2 / Step 3**, never "Phase" — Phase numbering is reserved for the Phase
0–9 failure-mode classifier lineage.

One infrastructure note that *does* transfer: Phase 9's constraint of causal/rolling features
only (no full-episode aggregates like `final_distance`) applies equally to this track's
deployment classifier (Step 3) — same reasoning, same deployment-time causality requirement.

---

## 3. Relevant prior history (carried forward, not otherwise in CLAUDE.md)

**The p_clean=0.2 AttackRandomizationWrapper run is a confirmed negative result, and matters here
because this new run is a different architecture attempting a similar goal (a policy that
performs reasonably across both clean and attacked conditions).**

- Root cause (not documented elsewhere in this repo's `.md` files): at `p_clean=0.2`, 80% of
  training episodes were under attack, several of which corrupt gripper action or object-position
  observations directly. This left insufficient clean signal for SAC+HER to learn the full
  grasp-transport-place sequence at all. Result: `final_distance` pinned at ~0.337 m (the fixed
  spawn distance) across nearly all conditions, for both `randomized_500k` and `randomized_2M` —
  the object essentially never moved. No robustness trade-off signal anywhere; flat failure, not
  a graceful degradation curve.
- This was scoped and reported as a negative result / ablation rather than retrained at a
  different `p_clean`, given deadline pressure. A `p_clean=0.5` retrain remains a known,
  not-yet-executed option, currently in tension with mentor guidance to run small ablations
  before any further full-length training job (see `TAIRO_Week7_Action_Plan.docx`, Tier 2).
- **Why this matters for Step 2 (below):** the attack-aware architecture is *not* the same
  mechanism as `AttackRandomizationWrapper` — the policy here is told which category is active
  rather than having to infer robustness blind — so the old failure mode may not transfer
  directly. But the underlying risk (too little clean signal to learn the base task at all) is
  the same shape of risk, and should inform the `p_clean` choice made during Step 1, not be
  re-discovered from scratch.

**Oracle-privilege precedent (recovery controller, §10/CLAUDE.md, backburner item):** the Week 5
recovery controller queries the simulator's true goal directly to defeat goal-spoofing — a
privilege that doesn't exist in real deployment. This is already a disclosed limitation. The
ground-truth-flag training here (Step 1) is the same shape of privilege: real during training,
absent at deployment, and swapped for an imperfect classifier later (Step 3). Use the same
disclosure pattern in the paper — don't treat this as a new problem needing a new framing.

---

## 4. Design decisions for Step 1 (diagnostic) to resolve

These are open going into Step 1 — do not assume answers, report them explicitly:

### 4a. Category mapping — draft, needs confirmation

Dr. Ho's action/sensor/goal framing predates the 3 PickAndPlace-specific conditions. Draft
mapping below is a starting proposal, not a decision — Step 1 should confirm or revise it:

| Condition | Draft category | Rationale / ambiguity |
|---|---|---|
| `clean` | clean | — |
| `sensor_dropout` | sensor | obs zeroed |
| `sensor_bias` | sensor | obs offset |
| `action_clipping` | action | action clipped |
| `action_delay` | action | action replay |
| `action_reversal` | action | action negated |
| `goal_spoof_immediate` | goal | goal offset |
| `goal_spoof_midep` | goal | goal offset, delayed onset |
| `object_pose_spoof` | **sensor (proposed)** | corrupts perceived object position in obs — same channel as sensor_bias, not the goal. Flag for Step 1 to confirm this isn't better treated as its own category given it's PickAndPlace-specific and behaves differently from generic sensor_bias (per Phase 8 findings, it's the dominant source of the `wrong_direction`/`action_control_corruption` confusion). |
| `grip_state_falsification` | **action (proposed)** | negates executed `action[3]` — action-space by construction (CLAUDE.md §11 C4 scoping table already classifies it this way) |
| `contact_dropout` | **sensor (proposed)** | zeros contact/force sensor fields |

### 4b. Injection point

Must go into the `observation` key of the obs dict, **not** `achieved_goal`/`desired_goal`.
Reason: SB3's `HerReplayBuffer` is configured with `goal_selection_strategy="future"`
(CLAUDE.md §12) — HER relabels `desired_goal` during replay sampling, so anything placed there
would be silently corrupted by hindsight relabeling. Step 1 must confirm this holds in the actual
SB3 version in use before implementation, not just assume it from the general HER mechanism.

### 4c. Flag encoding

One-hot (4-dim: clean/action/sensor/goal) is the default recommendation — avoids implying false
ordinal distance between attack types to the critic. Step 1 should confirm or override.

### 4d. `p_clean`

No default assumed. Step 1 must recommend a value informed by §3 above, and state the reasoning
explicitly rather than silently picking one.

### 4e. `total_timesteps`

Dr. Ho described this as a "couple hours" run — closer to the 500k scale than 2M based on prior
wall-clock timing, but Step 1 should confirm against actual observed throughput on this hardware.

---

## 5. Track steps

### Step 1 — Diagnostic & design (NO training launch, NO file modifications)

Resolve all of §4a–4e above. Inspect actual `HerReplayBuffer`/obs-dict behavior rather than
assuming. Report a plain-language verdict — category mapping table, injection-point confirmation,
encoding choice, `p_clean`, `total_timesteps` — and stop. Do not proceed to Step 2 without review.

### Step 2 — Implementation & training

Only after Step 1 is reviewed and approved. Implement the ground-truth attack-category
observation wrapper per Step 1's confirmed design. New model artifact:
`results/models/sac_her_pickandplace_attackaware_3cat` (no `.zip` extension, matching existing
convention). Do not overwrite or modify `sac_her_pickandplace_clean` / `_randomized` or any
existing config.py constants — add new constants, don't repurpose old ones. Write TensorBoard
logs matching the existing convention (§12). Launch training on the separate branch. Report back
when complete; do not proceed to eval/sweep without explicit go-ahead.

### Step 3 — Deployment classifier dataset (can run in parallel with Step 2)

Build the training dataset for the deployment-time attack classifier: input = observation
history + action history (rolling window, causal only — same constraint as Phase 9, §2 above),
label = ground-truth attack category from Step 1's confirmed mapping. Reuse existing
`step_logs_*.csv` infrastructure — the 4,950-episode sweeps already carry `condition`, which maps
onto the category scheme, so no new episodes are required just for dataset construction. Dataset
construction only — do not train the classifier itself in this step. Report dataset size, class
balance, and any gaps before proceeding.

### Step 4 (future, not yet scoped) — Splice classifier predictions into Step 2's policy in place
of ground truth; measure performance degradation. Not started; depends on Step 2 and Step 3 both
being complete.

---

## 6. Open questions log

*(Update this section as Steps run — do not let answers live only in chat history.)*

- [ ] Category mapping for `object_pose_spoof`, `grip_state_falsification`, `contact_dropout` —
      draft in §4a, needs Step 1 confirmation.
- [ ] Whether `p_clean=0.5`-style risk (from §3) applies to this architecture or not — Step 1 to
      assess, not assume.
- [ ] Scope for this week vs. next: Step 2's eval sweep (comparable cost to the existing
      11-condition × 5-seed sweeps) and any resulting paper figures are not yet scheduled against
      the Week 7 action plan's Tiers 0–4 — needs a slot once Step 1's timeline estimate exists.

---

## Pointer text added to CLAUDE.md

Add under §1 (Project Summary), as a new short paragraph after the existing failure-mode
classifier paragraph:

> A third, independent workstream — an **attack-aware policy** (ground-truth attack-category
> flag added to the SAC+HER observation, per Dr. Ho's proposal) — is tracked entirely in
> `ATTACK_AWARE_TRACK.md`, developed on a separate branch. **Do not conflate with the Phase 9
> online failure-mode classifier in §14** — different classifier, different target, different
> consumer. See `ATTACK_AWARE_TRACK.md` for full context before doing any work on this track.

And one line added to §15 (Hard Rules for Claude Code), as rule 13:

> 13. **Attack-aware policy work (ground-truth attack-category flag, Dr. Ho's proposal) is
>     tracked in `ATTACK_AWARE_TRACK.md`, not here** — read it before touching anything on that
>     branch, and do not add its findings into CLAUDE.md directly.
