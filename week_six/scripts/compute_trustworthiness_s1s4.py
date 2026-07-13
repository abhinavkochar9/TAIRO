"""Standalone S1-S4 trustworthiness sub-score computation (paper Eq. 1, Sec. III.G).

Self-contained: does NOT import or modify evaluation/metrics.py, which uses
different (audited-as-incorrect-vs-the-paper) ingredients for S2-S4. Reads the
post-C4-fix PickAndPlace episode_results_*.csv / step_logs_*.csv files and
computes, per trained model, aggregated across all 11 attack conditions:

    S1 Reliability      = success_rate
    S2 Robustness       = 1 - mean_t ||a_t - a_{t-1}||   (raw, episode-level;
                           no dataset-relative normalization)
    S3 Cyber Resilience = success_rate_attacked / (success_rate_clean + eps)
    S4 Adaptation        = max(0, (d_start - d_final) / (d_start + eps))

All four are clipped to [0, 1] at the per-episode (S1/S2/S4) or per-seed-ratio
(S3) level, then reported as mean +/- std across the 5 evaluation seeds.

S5 (Recovery) and T (the full composite) are explicitly OUT OF SCOPE: there is
no PickAndPlace recovery-controller data yet, so S5 cannot be computed.

Usage:
    /opt/miniconda3/envs/reu_robotics/bin/python3 scripts/compute_trustworthiness_s1s4.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import DATA_DIR

EPS = 1e-6

MODELS = ["clean_2M", "clean_500k", "randomized_2M", "randomized_500k"]

MODEL_LABELS = {
    "clean_2M": "clean\\_2M",
    "clean_500k": "clean\\_500k",
    "randomized_2M": "randomized\\_2M",
    "randomized_500k": "randomized\\_500k",
}


def _load_model(model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_path = f"{DATA_DIR}/episode_results_sac_her_pickandplace_{model}.csv"
    step_path = f"{DATA_DIR}/step_logs_sac_her_pickandplace_{model}.csv"
    episodes = pd.read_csv(episode_path)
    steps = pd.read_csv(step_path)
    return episodes, steps


def _join_d_start(model: str, episodes: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """Attach d_start (distance_to_goal at timestep==0) via the (model, episode_idx) key.

    episode_idx is confirmed unique within a single model's CSV pair (globally
    incrementing across condition x seed x episode), so within one model's
    files episode_idx alone is a valid join key; the "model" half of the key
    is satisfied by processing one model's file pair at a time.
    """
    t0 = steps[steps["timestep"] == 0]
    dupes = t0["episode_idx"].duplicated().sum()
    if dupes:
        raise ValueError(f"[{model}] {dupes} duplicate episode_idx at timestep==0 in step_logs")

    d_start = t0.set_index("episode_idx")["distance_to_goal"].rename("d_start")
    merged = episodes.join(d_start, on="episode_idx")

    missing = merged["d_start"].isna()
    if missing.any():
        print(f"  WARNING [{model}]: {missing.sum()} episodes have no timestep==0 row in "
              f"step_logs and could not be joined for d_start — their S4 will be NaN. "
              f"episode_idx values: {sorted(merged.loc[missing, 'episode_idx'].tolist())}")
    return merged


def compute_scores(model: str) -> dict:
    episodes, steps = _load_model(model)
    episodes = _join_d_start(model, episodes, steps)

    # Per-episode S2 and S4, each clipped to [0, 1] individually.
    episodes["s2_episode"] = (1.0 - episodes["action_smoothness"]).clip(0.0, 1.0)
    s4_raw = (episodes["d_start"] - episodes["final_distance"]) / (episodes["d_start"] + EPS)
    episodes["s4_episode"] = s4_raw.clip(lower=0.0, upper=1.0)

    seed_rows = []
    for seed, seed_df in episodes.groupby("seed"):
        clean_df = seed_df[seed_df["condition"] == "clean"]
        attacked_df = seed_df[seed_df["condition"] != "clean"]

        s1 = seed_df["success"].mean()

        s_clean = clean_df["success"].mean()
        s_attack = attacked_df["success"].mean()
        s3 = float(np.clip(s_attack / (s_clean + EPS), 0.0, 1.0))

        seed_rows.append({
            "seed": seed,
            "S1": s1,
            "S2": seed_df["s2_episode"].mean(),
            "S3": s3,
            "S4": seed_df["s4_episode"].mean(),
        })

    seed_scores = pd.DataFrame(seed_rows)

    dropped = int(episodes["d_start"].isna().sum())

    return {
        "model": model,
        "S1_mean": seed_scores["S1"].mean(), "S1_std": seed_scores["S1"].std(),
        "S2_mean": seed_scores["S2"].mean(), "S2_std": seed_scores["S2"].std(),
        "S3_mean": seed_scores["S3"].mean(), "S3_std": seed_scores["S3"].std(),
        "S4_mean": seed_scores["S4"].mean(), "S4_std": seed_scores["S4"].std(),
        "n_episodes": len(episodes),
        "n_dropped_d_start": dropped,
        "d_start_min": episodes["d_start"].min(),
        "d_start_max": episodes["d_start"].max(),
        "d_start_nunique": episodes["d_start"].nunique(),
        "action_smoothness_min": episodes["action_smoothness"].min(),
        "action_smoothness_max": episodes["action_smoothness"].max(),
    }


def run_sanity_checks(results: list[dict]) -> list[str]:
    problems = []
    for r in results:
        m = r["model"]

        if np.isclose(r["S3_mean"], r["S1_mean"], atol=1e-9):
            problems.append(f"[{m}] S3_mean == S1_mean ({r['S3_mean']:.6f}) — investigate.")

        if r["S4_mean"] < -1e-9:
            problems.append(f"[{m}] S4_mean is negative ({r['S4_mean']:.6f}).")

        if r["d_start_max"] <= 0:
            problems.append(f"[{m}] d_start values are all <= 0.")

        if r["n_dropped_d_start"] > 0:
            problems.append(
                f"[{m}] {r['n_dropped_d_start']} episodes dropped from S4 due to missing d_start."
            )

    return problems


def to_latex(results: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{S1--S4 Trustworthiness Sub-Scores --- SAC+HER, "
                 r"FetchPickAndPlace-v4, No Recovery. Mean $\pm$ std across 5 "
                 r"evaluation seeds, aggregated over all 11 attack conditions "
                 r"(clean + 10 attacked) per model.}")
    lines.append(r"\label{tab:s1s4}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & $S_1$ & $S_2$ & $S_3$ & $S_4$ \\")
    lines.append(r"\midrule")
    for r in results:
        label = MODEL_LABELS[r["model"]]
        lines.append(
            f"{label} & "
            f"{r['S1_mean']:.3f} $\\pm$ {r['S1_std']:.3f} & "
            f"{r['S2_mean']:.3f} $\\pm$ {r['S2_std']:.3f} & "
            f"{r['S3_mean']:.3f} $\\pm$ {r['S3_std']:.3f} & "
            f"{r['S4_mean']:.3f} $\\pm$ {r['S4_std']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    results = [compute_scores(m) for m in MODELS]

    print("=" * 70)
    print("Per-model diagnostics")
    print("=" * 70)
    for r in results:
        print(f"\n[{r['model']}]")
        print(f"  n_episodes             = {r['n_episodes']}")
        print(f"  n_dropped (S4 d_start)  = {r['n_dropped_d_start']}")
        print(f"  d_start range           = [{r['d_start_min']:.6f}, {r['d_start_max']:.6f}] "
              f"({r['d_start_nunique']} unique values)")
        print(f"  action_smoothness range = [{r['action_smoothness_min']:.6f}, "
              f"{r['action_smoothness_max']:.6f}]")
        print(f"  S1 = {r['S1_mean']:.4f} +/- {r['S1_std']:.4f}")
        print(f"  S2 = {r['S2_mean']:.4f} +/- {r['S2_std']:.4f}")
        print(f"  S3 = {r['S3_mean']:.4f} +/- {r['S3_std']:.4f}")
        print(f"  S4 = {r['S4_mean']:.4f} +/- {r['S4_std']:.4f}")

    print("\n" + "=" * 70)
    print("Sanity checks")
    print("=" * 70)
    problems = run_sanity_checks(results)
    if problems:
        print("FAILED:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All checks passed (S3 != S1, S2/S4 well-behaved, no dropped episodes).")

    print("\n" + "=" * 70)
    print("LaTeX table")
    print("=" * 70)
    print(to_latex(results))


if __name__ == "__main__":
    main()
