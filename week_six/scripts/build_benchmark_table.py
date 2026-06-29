"""
Build the B0–B3 mean ± std benchmark table.

Reads:  results/data/episode_results.csv
Writes: results/data/summary.csv  — one row per (benchmark_layer, method, condition)
        with mean metrics and C1–C5 trustworthiness scores.

Also prints a formatted success_rate and final_distance table to stdout so
the mentor-facing results are visible without opening the CSV.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DATA_DIR, BENCHMARK_LAYERS
from evaluation.metrics import summarize_results, add_trustworthiness_scores


def _fmt(series: pd.Series) -> str:
    """Return 'mean ± std' string for a numeric series."""
    return f"{series.mean():.3f} ± {series.std():.3f}"


def main() -> None:
    ep_path = os.path.join(DATA_DIR, "episode_results.csv")
    df = pd.read_csv(ep_path)
    print(f"[table] Loaded {len(df)} episodes from {ep_path}")

    # Build summary per benchmark layer so trustworthiness scores are computed
    # within-layer (each layer has its own normalisation context).
    summary_parts = []
    for layer in BENCHMARK_LAYERS:
        layer_df = df[df["benchmark_layer"] == layer]
        if layer_df.empty:
            print(f"[table] WARNING: no data for layer {layer}, skipping.")
            continue
        summary = summarize_results(layer_df)
        summary = add_trustworthiness_scores(summary)
        summary.insert(0, "benchmark_layer", layer)
        summary_parts.append(summary)

    summary_df = pd.concat(summary_parts, ignore_index=True)

    out_path = os.path.join(DATA_DIR, "summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"[table] Wrote {len(summary_df)} rows → {out_path}")

    # --- Formatted mean ± std table across seeds (printed to stdout) ----------
    print("\n=== B0–B3 Benchmark Table: Success Rate & Distance (mean ± std) ===\n")
    group_cols = ["benchmark_layer", "method", "condition"]
    table_rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        layer, method, condition = keys
        table_rows.append({
            "layer":          layer,
            "method":         method,
            "condition":      condition,
            "success_rate":   _fmt(g["success"]),
            "final_distance": _fmt(g["final_distance"]),
            "n_episodes":     len(g),
        })

    display_df = pd.DataFrame(table_rows)
    # Order by layer then condition to match B0→B3 narrative in the paper.
    layer_order = {l: i for i, l in enumerate(BENCHMARK_LAYERS)}
    display_df["_layer_key"] = display_df["layer"].map(layer_order)
    display_df = display_df.sort_values(["_layer_key", "condition", "method"]).drop(
        columns=["_layer_key"]
    )
    print(display_df.to_string(index=False))

    # --- Trustworthiness score summary ----------------------------------------
    print("\n=== Weighted Trustworthiness Score (C1–C5) per Method × Layer ===\n")
    ts_cols = [
        "benchmark_layer", "method", "condition",
        "reliability_score", "robustness_score",
        "cyber_resilience_score", "safety_score", "recovery_score",
        "trustworthiness_score_weighted",
    ]
    available = [c for c in ts_cols if c in summary_df.columns]
    print(summary_df[available].sort_values(
        ["benchmark_layer", "condition", "method"]
    ).to_string(index=False))


if __name__ == "__main__":
    main()
