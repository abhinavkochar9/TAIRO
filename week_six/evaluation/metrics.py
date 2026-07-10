"""
TAIRO metric computation — summary statistics and trustworthiness scores.

Metric fields produced by add_trustworthiness_scores()
-------------------------------------------------------
reliability_score      — Does the robot succeed under its operating condition?
robustness_score       — How well does performance hold under disturbances?
cyber_resilience_score — How well does the robot resist adversarial attacks?
safety_score           — Does the robot avoid unsafe actuator behaviour?
recovery_score         — Does the robot improve success rate over no-recovery?

Composite scores
----------------
trustworthiness_score_equal    : equal weights — 0.20 × each component.
                                 Useful as a model-free baseline comparison.

trustworthiness_score_weighted : argued weights from the Week 3 TAIRO
                                 framework C1–C5 priorities:
                                   reliability      0.10
                                   robustness       0.20
                                   cyber_resilience 0.25
                                   safety           0.15
                                   recovery         0.30

trustworthiness_score          : alias for trustworthiness_score_weighted
                                 (kept for backward compatibility with Week 4
                                 pipelines that reference this column).

Weight rationale (Week 3 framework component weights, argued ordering)
----------------------------------------------------------------------
C1 Perception & State Understanding      → reliability_score      0.10
C2 Uncertainty & Failure Detection       → robustness_score       0.20
C3 Cybersecurity-Aware Reasoning         → cyber_resilience_score 0.25
C4 RL-Based Adaptation                   → safety_score           0.15
C5 Failure Recovery & Safety Control     → recovery_score         0.30
"""

from typing import Optional

import numpy as np
import pandas as pd

from config import SAFETY_ARM_JERK_THRESHOLD, SAFETY_GRIPPER_JERK_THRESHOLD  # noqa: F401 — re-exported for reference

# ---------------------------------------------------------------------------
# Equal weights — 0.20 × each component
# ---------------------------------------------------------------------------
WEIGHT_EQUAL = 0.20

# ---------------------------------------------------------------------------
# Argued weights — Week 3 TAIRO framework C1–C5 priorities (must sum 1.0)
# ---------------------------------------------------------------------------
WEIGHT_RELIABILITY       = 0.10
WEIGHT_ROBUSTNESS        = 0.20
WEIGHT_CYBER_RESILIENCE  = 0.25
WEIGHT_SAFETY            = 0.15
WEIGHT_RECOVERY          = 0.30

assert abs(5 * WEIGHT_EQUAL - 1.0) < 1e-9, "Equal weights must sum to 1.0"
assert abs(
    WEIGHT_RELIABILITY
    + WEIGHT_ROBUSTNESS
    + WEIGHT_CYBER_RESILIENCE
    + WEIGHT_SAFETY
    + WEIGHT_RECOVERY
    - 1.0
) < 1e-9, "TAIRO argued weights must sum to 1.0"


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate episode-level records into per-(method, condition, attack_level) rows.

    Args:
        df: Episode-level DataFrame produced by run_episode / run_benchmark.

    Returns:
        Summary DataFrame with one row per (method, condition, attack_level)
        group, containing mean values for all key metrics.
    """
    group_cols = ["method", "condition", "attack_level"]
    summary = (
        df.groupby(group_cols)
        .agg(
            success_rate=("success", "mean"),
            avg_reward=("total_reward", "mean"),
            final_distance=("final_distance", "mean"),
            avg_episode_length=("episode_length", "mean"),
            action_smoothness=("action_smoothness", "mean"),
            action_magnitude=("action_magnitude", "mean"),
            safety_violation_rate=("safety_violation", "mean"),
            recovery_rate=("recovery_used", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return summary


def add_trustworthiness_scores(
    summary: pd.DataFrame,
    baseline_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute TAIRO C1–C5 sub-scores and two composite trustworthiness scores.

    Sub-score definitions
    ---------------------
    C1 Reliability        = success_rate (direct, clipped to [0, 1])

    C2 Robustness         = 0.6 * (1 - normalised_distance)
                          + 0.4 * (1 - normalised_smoothness)
                          Captures both how close the robot gets and how
                          smoothly it moves (attack-induced jitter degrades
                          smoothness).

    C3 Cyber Resilience   = success_rate
                          Reuses reliability because attack resilience is
                          directly observable as maintained task success.
                          (clean condition receives resilience = reliability)

    C4 Safety             = 1 - safety_violation_rate

    C5 Recovery           = max(0, this_method_success_rate - no_recovery_success_rate)
                          where no_recovery_success_rate is the sac_her (B1)
                          success rate for the same (condition, attack_level).
                          Clean condition is set to 1.0 by convention — nothing
                          to recover from.  Base sac_her rows receive 0.0.
                          ``recovery_rate`` (trigger frequency) is preserved as
                          its own column but is NOT used as the score.

    Composite scores
    ----------------
    trustworthiness_score_equal    : 0.20 × (C1 + C2 + C3 + C4 + C5)
    trustworthiness_score_weighted : argued weights (see module docstring)
    trustworthiness_score          : alias for trustworthiness_score_weighted

    Args:
        summary:          Output of summarize_results().
        baseline_summary: Optional summary DataFrame containing sac_her B1 rows
                          used as the C5 no-recovery baseline.  Pass the B1
                          summarize_results() output when scoring B2/B3 layers
                          so the cross-layer sac_her success rates are available.
                          If None, sac_her rows within ``summary`` itself are
                          used (correct for B0/B1 single-layer calls).

    Returns:
        summary with C1–C5 sub-score columns, both composite columns, and the
        backward-compatible ``trustworthiness_score`` alias appended.
    """
    out = summary.copy()

    # -- C1: Reliability (Perception & State Understanding) --------------------
    out["reliability_score"] = out["success_rate"].clip(0.0, 1.0)

    # -- C2: Robustness (Uncertainty & Failure Detection) ----------------------
    max_dist   = max(out["final_distance"].max(), 1e-9)
    max_smooth = max(out["action_smoothness"].max(), 1e-9)

    distance_score   = 1.0 - (out["final_distance"]    / max_dist)
    smoothness_score = 1.0 - (out["action_smoothness"] / max_smooth)
    out["robustness_score"] = (0.6 * distance_score + 0.4 * smoothness_score).clip(0.0, 1.0)

    # -- C3: Cyber Resilience (Cybersecurity-Aware Reasoning) ------------------
    out["cyber_resilience_score"] = out["success_rate"].clip(0.0, 1.0)

    # -- C4: Safety (RL-Based Adaptation) --------------------------------------
    out["safety_score"] = (1.0 - out["safety_violation_rate"]).clip(0.0, 1.0)

    # -- C5: Recovery (Failure Recovery & Safety Control) ----------------------
    # Formula: max(0, this_method_success - no_recovery_baseline_success)
    # Baseline is sac_her success rate for the same (condition, attack_level).
    # Use baseline_summary if provided (needed when scoring B2/B3 without B1
    # sac_her rows present in `summary`); otherwise fall back to sac_her rows
    # within `summary` itself.
    ref = baseline_summary if baseline_summary is not None else out
    no_recovery_sr = (
        ref[ref["method"] == "sac_her"]
        .set_index(["condition", "attack_level"])["success_rate"]
        .to_dict()
    )

    def _recovery_score(row: pd.Series) -> float:
        if row["condition"] == "clean":
            return 1.0
        if row["method"] == "sac_her":
            return 0.0
        key = (row["condition"], row["attack_level"])
        baseline = no_recovery_sr.get(key, 0.0)
        return max(0.0, float(row["success_rate"]) - baseline)

    out["recovery_score"] = out.apply(_recovery_score, axis=1)

    # -- Composite: equal weights (0.20 × each) --------------------------------
    out["trustworthiness_score_equal"] = WEIGHT_EQUAL * (
        out["reliability_score"]
        + out["robustness_score"]
        + out["cyber_resilience_score"]
        + out["safety_score"]
        + out["recovery_score"]
    )

    # -- Composite: argued weights (Week 3 framework priorities) ---------------
    out["trustworthiness_score_weighted"] = (
        WEIGHT_RELIABILITY      * out["reliability_score"]
        + WEIGHT_ROBUSTNESS     * out["robustness_score"]
        + WEIGHT_CYBER_RESILIENCE * out["cyber_resilience_score"]
        + WEIGHT_SAFETY         * out["safety_score"]
        + WEIGHT_RECOVERY       * out["recovery_score"]
    )

    # Backward-compatible alias — always matches the argued/weighted formula
    out["trustworthiness_score"] = out["trustworthiness_score_weighted"]

    return out
