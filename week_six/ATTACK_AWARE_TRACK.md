# ATTACK_AWARE_TRACK.md — Attack-Aware Policy (Dr. Ho Proposal)

**Status:** Step 1 (diagnostic & design, §4) complete — pending human + mentor review before
Step 2 (implementation/training) begins. See §4 and §6 for resolved answers.
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

**Status: RESOLVED (Step 1 complete, 2026-07-08). All five sub-decisions below were confirmed
by direct inspection of this codebase/environment/hardware, not by assumption. Pending human +
mentor review before Step 2 begins.**

### 4a. Category mapping — CONFIRMED (draft mapping unchanged)

| Condition | Category | Rationale |
|---|---|---|
| `clean` | clean | — |
| `sensor_dropout` | sensor | obs zeroed |
| `sensor_bias` | sensor | obs offset |
| `action_clipping` | action | action clipped |
| `action_delay` | action | action replay |
| `action_reversal` | action | action negated |
| `goal_spoof_immediate` | goal | goal offset |
| `goal_spoof_midep` | goal | goal offset, delayed onset |
| `object_pose_spoof` | sensor | corrupts perceived object position in `obs["observation"]` — same channel as `sensor_bias` |
| `grip_state_falsification` | action | negates executed `action[3]` — action-space by construction (matches CLAUDE.md §11 C4 scoping table) |
| `contact_dropout` | sensor | zeros contact/force sensor fields in `obs["observation"]` |

Confirmed by applying one consistent criterion across all 11 conditions — **which channel does
the attack corrupt: the executed action stream, the observation/sensor stream, or the goal
stream?** All three PickAndPlace-specific conditions resolve unambiguously under this test:
`object_pose_spoof` and `contact_dropout` both corrupt fields inside `obs["observation"]`
(mechanistically identical to `sensor_bias`/`sensor_dropout`); `grip_state_falsification`
corrupts the executed action (mechanistically identical to `action_reversal`).

`object_pose_spoof`'s flagged ambiguity in the draft was about a **different problem**: Phase 8's
`wrong_direction`/`action_control_corruption` confusion is about a *failure-mode* classifier
(predicting *how* an episode failed, post-hoc, from full-episode trajectory shape) misattributing
labels for this condition specifically. That is orthogonal to this classifier's job (predicting
*which attack category is currently active*, causally, from obs/action history) and orthogonal to
the mechanistic channel question this mapping answers. Recommend not treating it as a reason to
add a bespoke 4th category — that would contradict Dr. Ho's explicit staged-rollout design (3
categories now, 11-class later per §1). If Step 2's eval shows `object_pose_spoof` is where the
3-category flag helps least, that is itself useful signal for prioritizing the future 11-class
step, not a reason to break the 3-category scheme now.

### 4b. Injection point — CONFIRMED: `observation` key, not `desired_goal`/`achieved_goal`

Checked directly against the installed SB3 version in `reu_robotics` (**stable_baselines3
2.8.0**), by reading
`stable_baselines3/her/her_replay_buffer.py::HerReplayBuffer._get_virtual_samples` (the method
that performs hindsight relabeling):

```python
# Get infos and obs
obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.observations.items()}
next_obs = {key: obs[batch_indices, env_indices, :] for key, obs in self.next_observations.items()}
...
# Sample and set new goals
new_goals = self._sample_goals(batch_indices, env_indices)
obs["desired_goal"] = new_goals
next_obs["desired_goal"] = new_goals
```

Relabeling **only ever overwrites the `desired_goal` key**, wholesale, with a goal sampled from a
different (future) timestep in the same episode. Every other key in the obs dict — including
`observation` — is copied through from the stored buffer untouched. This confirms both halves of
§4b's question directly from source, not from general HER theory:
- A flag placed in `observation` **survives** relabeling exactly as stored (it is never touched
  by `_get_virtual_samples`).
- A flag placed in `desired_goal` **would not** — it gets replaced in full by `new_goals`, which
  is a 3-dim `achieved_goal` sampled from a different step, with no mechanism to preserve a flag
  that was concatenated onto it.

Additionally confirmed empirically in this environment (`gymnasium` + `gymnasium_robotics`,
`FetchPickAndPlace-v4`): `achieved_goal`/`desired_goal` are shape `(3,)` and
`env.unwrapped.compute_reward(achieved_goal, desired_goal, info)` consumes that exact shape —
appending flag dims to `achieved_goal` would also break virtual-transition reward computation in
`_get_virtual_samples` (`self.env.env_method("compute_reward", next_obs["achieved_goal"],
obs["desired_goal"], infos, ...)`), independent of the relabeling-survival question. `observation`
is the only viable injection point on both grounds.

`obs["observation"]` for `FetchPickAndPlace-v4` is confirmed shape `(25,) float64`; a 4-dim
one-hot flag appended makes it `(29,)`.

### 4c. Flag encoding — CONFIRMED: one-hot, 4-dim (clean/action/sensor/goal)

No override. One-hot avoids implying false ordinal distance between attack categories to the
critic (an integer/ordinal encoding would imply, e.g., "goal" is "closer" to "sensor" than to
"action," which is meaningless here). 4-dim (not 3-dim + implicit all-zero-for-clean) keeps
`clean` an explicit, addressable state rather than an implicit default, consistent with how the
other categorical/structural flags in this codebase are handled explicitly rather than inferred.

### 4d. `p_clean` — CONFIRMED: 0.5

**Anchored in a previously-unreported existing run, not just reasoning from §3.**
`results/models/sac_her_pickandplace_randomized_p50_2M` (`results/data/train_log_p50_2M.txt`,
`p_clean=0.5`, `AttackRandomizationWrapper`, 2M timesteps) has already been trained and is not yet
documented in `findings.md`/`todo.md`. Its rollout `success_rate` curve (sampled from the log):

```
timesteps    success_rate
    600      0.00
 240,600     0.03
 480,600     0.07
 960,600     0.01
1,248,600    0.04
1,296,600    0.09
1,440,600    0.12
1,680,600    0.19
1,824,600    0.28
1,968,600    0.33   (final ≈0.38)
```

This is a real, if slow and noisy, learning curve — not the flat pinned-at-spawn-distance failure
the `p_clean=0.2` run produced (§3). That is direct evidence in this exact codebase that
`p_clean=0.5` clears the "insufficient clean signal" failure mode that killed `0.2`.

Whether that risk transfers to the attack-aware architecture: **partially, and probably less
severely.** The mechanism is different — `AttackRandomizationWrapper` forces the policy to learn
one blind, averaged-robustness behavior across all conditions with no signal for which is active;
the attack-aware architecture tells the policy which category is active, which should reduce
(not eliminate) the burden, since the policy can in principle specialize per-category rather than
average across them blindly. This is a reasonable hypothesis but **not yet evidence** — no
attack-aware run exists yet to confirm the flag actually buys back clean-signal budget. Given
that, and given no run at any other `p_clean` value exists in this codebase, recommend anchoring
Step 2's ground-truth run at the one value with direct, non-flat evidence (`0.5`) rather than
speculatively going lower on an unverified hypothesis. A follow-up ablation at lower `p_clean`
(e.g. 0.3–0.4) to test whether the flag's expected advantage allows a lower value is reasonable
future work, but out of scope for this run.

### 4e. `total_timesteps` — CONFIRMED: 2,000,000 (overrides Dr. Ho's "couple hours" framing — flagged for explicit review)

**Measured wall-clock throughput, this hardware, this exact training setup**
(`train_log_p50_2M.txt`, same env/replay-buffer/network configuration expected for the
attack-aware run — only a 4-dim obs concatenation differs, negligible compute overhead):

| Timesteps | Wall-clock (measured) |
|---|---|
| 500,000 | ~1.07 hr |
| 1,000,000 | ~2.25 hr |
| 1,500,000 | ~3.44 hr |
| 2,000,000 | ~4.91 hr |

Dr. Ho's "couple hours" estimate maps almost exactly to **1,000,000** timesteps by throughput
alone — not the "closer to 500k" framing in the original draft. But throughput is only half the
picture. **The success-rate curve in §4d above shows this specific training setup (PickAndPlace,
partial attack exposure) stays in the single digits through ~1.2M timesteps and only starts
ramping up after ~1.3–1.4M**, reaching 20–33% by 2M. Stopping at 500k or 1M — i.e., at "couple
hours" — would almost certainly land inside that flat, low-signal region, which is
indistinguishable from reproducing the §3 negative result again, this time misattributed to the
new architecture rather than to an undertrained run. That would also directly defeat the run's
stated purpose (§1, point 3: "establishes an upper bound before classifier noise enters the
picture") — a truncated run gives a meaningless, artificially pessimistic upper bound.

**Recommendation: 2,000,000 timesteps (~4.9 hr), explicitly diverging from the "couple hours"
estimate.** This is a real cost increase over Dr. Ho's framing and is flagged here specifically
for human/mentor sign-off rather than silently resolved — it is a genuine time/resource
trade-off, not a technical judgment call. It is consistent with the existing repo convention that
2M, not 500k, is the timestep scale at which PickAndPlace models become reliable at all (§1: only
`clean_2M` — not `clean_500k` — achieves reliable clean-condition success).

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

- [x] Category mapping for `object_pose_spoof`, `grip_state_falsification`, `contact_dropout` —
      **confirmed as drafted** (§4a): all three resolve unambiguously under the "which channel is
      corrupted" test. `object_pose_spoof`'s Phase 8 ambiguity is a different (failure-mode
      classifier) problem, not a reason to add a 4th category now.
- [x] Whether `p_clean=0.5`-style risk (from §3) applies to this architecture or not — **partially,
      probably less severely** (§4d), but unverified until an attack-aware run actually exists.
      Anchored `p_clean=0.5` on a previously-undocumented existing run
      (`sac_her_pickandplace_randomized_p50_2M`) that shows a real, non-flat learning curve —
      not on reasoning alone.
- [x] `total_timesteps` — **2,000,000 recommended**, diverging from Dr. Ho's "couple hours"
      framing (§4e). Measured throughput on this hardware puts "couple hours" at ~1M steps, but
      the comparable existing run's success-rate curve stays flat/low through ~1.2–1.4M steps —
      stopping early risks reproducing the §3 negative result and undermines the run's stated
      purpose of establishing a real upper bound. **Flagged explicitly for mentor sign-off** since
      it's a resource trade-off, not just a technical call.
- [x] Injection point — **confirmed via direct SB3 2.8.0 source inspection** (§4b), not assumed:
      `HerReplayBuffer._get_virtual_samples` only overwrites `desired_goal` during relabeling;
      `observation` passes through untouched. `achieved_goal` also ruled out independently
      (reward computation requires its `(3,)` shape).
- [x] Flag encoding — **confirmed one-hot, 4-dim** (§4c), no override.
- [ ] Scope for this week vs. next: Step 2's eval sweep (comparable cost to the existing
      11-condition × 5-seed sweeps) and any resulting paper figures are not yet scheduled against
      the Week 7 action plan's Tiers 0–4 — needs a slot once Step 1's timeline estimate exists.
      (Now that `total_timesteps=2M` (~4.9 hr) is recommended, this scheduling question is more
      pressing than when the file assumed a "couple hours" run.)

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
