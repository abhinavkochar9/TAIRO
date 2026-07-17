# RECOVERED FROM git commit 53a40ac (failure-mode-classifier branch), original path week_six/scripts/build_paper_trustworthiness_table.py — REFERENCE ONLY, do not run/import/restore. The Eq.2-vs-metrics.py reconciliation decision is still open (see findings.md).
"""
Build the paper-exact composite trustworthiness table (Eq. 2, Section III-G)
for Section V-D of the TAIRO paper.

This is a standalone, read-only computation, independent of
evaluation/metrics.py's C1-C5 pipeline. metrics.py's C2/C3/C4 formulas do not
match the paper's S2/S3/S4 definitions (see
results/investigations/paper_trustworthiness_table_findings.md for the
formula-by-formula comparison), so this script reimplements S1-S5 and T
exactly as specified in the paper rather than reusing add_trustworthiness_scores().

Eq. 2:
    T = 0.10*S1 + 0.25*S2 + 0.25*S3 + 0.05*S4 + 0.35*S5

Reads:  results/data/episode_results_sac_her_pickandplace_<model>.csv
        results/data/step_logs_sac_her_pickandplace_<model>.csv
Writes: results/data/paper_trustworthiness_table.csv

Scope: sac_her (no-recovery) rows only, PickAndPlace, all 4 models. No
recovery-method episode data currently exists for PickAndPlace (confirmed:
episode_results 'method' column is 'sac_her' only in every file), so S5 is
computed as 0.0 for every row rather than omitted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import DATA_DIR

# ---------------------------------------------------------------------------
# Eq. 2 weights (Section III-G) — must sum to 1.0
# ---------------------------------------------------------------------------
WEIGHT_S1_RELIABILITY  = 0.10
WEIGHT_S2_ROBUSTNESS   = 0.25
WEIGHT_S3_RESILIENCE   = 0.25
WEIGHT_S4_ADAPTATION   = 0.05
WEIGHT_S5_RECOVERY     = 0.35

assert abs(
    WEIGHT_S1_RELIABILITY
    + WEIGHT_S2_ROBUSTNESS
    + WEIGHT_S3_RESILIENCE
    + WEIGHT_S4_ADAPTATION
    + WEIGHT_S5_RECOVERY
    - 1.0
) < 1e-9, "Eq. 2 weights must sum to 1.0"

# Epsilon guarding the S3 and S4 denominators (Section III-G ratio formulas)
EPS = 1e-6

MODELS = ["clean_500k", "clean_2M", "randomized_500k", "randomized_2M"]


def _episode_frame(model: str) -> pd.DataFrame:
    """Build a per-episode DataFrame with all raw ingredients for S1/S2/S4."""
    ep = pd.read_csv(os.path.join(DATA_DIR, f"episode_results_sac_her_pickandplace_{model}.csv"))
    sl = pd.read_csv(os.path.join(DATA_DIR, f"step_logs_sac_her_pickandplace_{model}.csv"))

    key = ["condition", "seed", "attack_level", "episode_idx"]
    d_start = (
        sl[sl["timestep"] == 0][key + ["distance_to_goal"]]
        .rename(columns={"distance_to_goal": "d_start"})
    )
    ep = ep.merge(d_start, on=key, how="left", validate="one_to_one")
    assert ep["d_start"].notna().all(), f"{model}: missing step-0 distance for some episodes"
    return ep


def _compute_episode_subscores(ep: pd.DataFrame) -> pd.DataFrame:
    """Attach per-episode S2 and S4 terms (S1 is just the 'success' column)."""
    out = ep.copy()

    # S2 (Robustness) = clip(1 - action_smoothness, 0, 1)
    # action_smoothness (episode_results) is already (1/L)*sum_t ||a_t - a_{t-1}||
    # (see evaluation/episode_runner.py:_action_smoothness) — exactly the raw
    # quantity Eq. 2's S2 formula needs, no renormalization.
    out["s2_ep"] = (1.0 - out["action_smoothness"]).clip(0.0, 1.0)

    # S4 (Adaptation) = clip(max(0, (d_start - d_final) / (d_start + eps)), 0, 1)
    d_final = out["final_distance"]
    progress = (out["d_start"] - d_final) / (out["d_start"] + EPS)
    out["s4_ep"] = progress.clip(lower=0.0).clip(0.0, 1.0)

    return out


def _build_model_table(model: str) -> pd.DataFrame:
    ep = _episode_frame(model)
    ep = _compute_episode_subscores(ep)

    grouped = ep.groupby("condition").agg(
        S1=("success", "mean"),
        S2=("s2_ep", "mean"),
        S4=("s4_ep", "mean"),
        n_episodes=("success", "count"),
    ).reset_index()

    # S3 (Cyber Resilience) = clip(success_rate_attack / (success_rate_clean + eps), 0, 1)
    # Ratio to the SAME model's clean-condition success rate.
    clean_sr = grouped.loc[grouped["condition"] == "clean", "S1"].iloc[0]
    grouped["S3"] = (grouped["S1"] / (clean_sr + EPS)).clip(0.0, 1.0)

    # S5 (Recovery) — no recovery-method PickAndPlace episode data currently
    # exists (episode_results 'method' column is 'sac_her' only). Computed
    # explicitly as 0.0 rather than omitted; see module docstring.
    grouped["S5"] = 0.0

    grouped["T"] = (
        WEIGHT_S1_RELIABILITY * grouped["S1"]
        + WEIGHT_S2_ROBUSTNESS * grouped["S2"]
        + WEIGHT_S3_RESILIENCE * grouped["S3"]
        + WEIGHT_S4_ADAPTATION * grouped["S4"]
        + WEIGHT_S5_RECOVERY * grouped["S5"]
    )

    grouped.insert(0, "model", model)
    return grouped[["model", "condition", "S1", "S2", "S3", "S4", "S5", "T", "n_episodes"]]


def _sanity_check(table: pd.DataFrame) -> list:
    """Return a list of human-readable warnings for any check that fails."""
    warnings = []

    # S1 must exactly match success_rate in the existing per-model summary CSVs.
    for model in MODELS:
        summary_path = os.path.join(DATA_DIR, f"sac_her_pickandplace_{model}_summary.csv")
        summary = pd.read_csv(summary_path)
        summary = summary[summary["method"] == "sac_her"][["condition", "success_rate"]]
        merged = table[table["model"] == model].merge(summary, on="condition", how="left")
        mismatch = merged[(merged["S1"] - merged["success_rate"]).abs() > 1e-9]
        if len(mismatch):
            warnings.append(
                f"{model}: S1 does not match summary CSV success_rate for conditions "
                f"{mismatch['condition'].tolist()}"
            )

    # Clean condition sanity: ~1.0 for clean_2M, ~0.0 for the others (established finding).
    clean_rows = table[table["condition"] == "clean"].set_index("model")["S1"]
    if not np.isclose(clean_rows.get("clean_2M", -1), 1.0, atol=1e-6):
        warnings.append(f"clean_2M clean-condition S1 = {clean_rows.get('clean_2M')}, expected ~1.0")
    for model in ["clean_500k", "randomized_500k", "randomized_2M"]:
        if not np.isclose(clean_rows.get(model, -1), 0.0, atol=1e-6):
            warnings.append(f"{model} clean-condition S1 = {clean_rows.get(model)}, expected ~0.0")

    # Range check.
    for col in ["S1", "S2", "S3", "S4", "S5"]:
        out_of_range = table[(table[col] < -1e-9) | (table[col] > 1.0 + 1e-9)]
        if len(out_of_range):
            warnings.append(f"{col} out of [0,1] for rows: {out_of_range[['model', 'condition']].values.tolist()}")

    return warnings


def main() -> None:
    tables = [_build_model_table(model) for model in MODELS]
    full_table = pd.concat(tables, ignore_index=True)

    output_path = os.path.join(DATA_DIR, "paper_trustworthiness_table.csv")
    full_table.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(full_table)} rows)")

    warnings = _sanity_check(full_table)
    if warnings:
        print("\nSANITY CHECK WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nAll sanity checks passed.")

    print("\n" + full_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
