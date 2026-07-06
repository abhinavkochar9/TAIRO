"""
Build the B0–B3 mean ± std benchmark table.

Reads:  results/data/episode_results.csv  (default, backward-compatible)
        or any path supplied via --input-file / --input-path
Writes: results/data/summary.csv  (default)
        or a stem-derived name matching the input, e.g.
        episode_results_sac_her_pickandplace_clean_2M.csv
        → results/data/sac_her_pickandplace_clean_2M_summary.csv
        Override with --output-file.

Also prints a formatted success_rate and final_distance table to stdout so
the mentor-facing results are visible without opening the CSV.

Usage
-----
FetchReach (original default, unchanged):
    conda run -n reu_robotics python3 scripts/build_benchmark_table.py

PickAndPlace — specific file:
    conda run -n reu_robotics python3 scripts/build_benchmark_table.py \\
        --input-file results/data/episode_results_sac_her_pickandplace_clean_2M.csv

With explicit output path:
    conda run -n reu_robotics python3 scripts/build_benchmark_table.py \\
        --input-file results/data/episode_results_sac_her_pickandplace_clean_2M.csv \\
        --output-file results/data/my_summary.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import DATA_DIR, BENCHMARK_LAYERS
from evaluation.metrics import summarize_results, add_trustworthiness_scores

_DEFAULT_INPUT = os.path.join(DATA_DIR, "episode_results.csv")


def _fmt(series: pd.Series) -> str:
    """Return 'mean ± std' string for a numeric series."""
    return f"{series.mean():.3f} ± {series.std():.3f}"


def _derive_output_path(input_path: str) -> str:
    """
    Derive a variant-specific summary filename from the input stem so that
    running the script for multiple model variants doesn't clobber one file.

    episode_results_sac_her_pickandplace_clean_2M.csv
      → results/data/sac_her_pickandplace_clean_2M_summary.csv

    episode_results.csv (legacy FetchReach default)
      → results/data/summary.csv  (backward-compatible)
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]  # strip .csv
    prefix = "episode_results_"
    if stem == "episode_results":
        return os.path.join(DATA_DIR, "summary.csv")
    if stem.startswith(prefix):
        variant = stem[len(prefix):]
        return os.path.join(DATA_DIR, f"{variant}_summary.csv")
    return os.path.join(DATA_DIR, f"{stem}_summary.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build B0–B3 mean ± std benchmark table from a sweep CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-file", "--input-path",
        dest="input_file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the episode_results CSV to process. "
            f"Defaults to {_DEFAULT_INPUT} (FetchReach backward-compat)."
        ),
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path for the output summary CSV. "
            "Defaults to a name derived from the input filename."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    ep_path = args.input_file if args.input_file is not None else _DEFAULT_INPUT
    df = pd.read_csv(ep_path)
    print(f"[table] Loaded {len(df)} episodes from {ep_path}")

    # Build summary per benchmark layer so trustworthiness scores are computed
    # within-layer (each layer has its own normalisation context).
    # B1 is pre-computed first and passed as the C5 no-recovery baseline to
    # B2 and B3, which lack sac_her rows of their own.
    b1_df = df[df["benchmark_layer"] == "B1"]
    b1_baseline = summarize_results(b1_df) if not b1_df.empty else None

    summary_parts = []
    for layer in BENCHMARK_LAYERS:
        layer_df = df[df["benchmark_layer"] == layer]
        if layer_df.empty:
            print(f"[table] WARNING: no data for layer {layer}, skipping.")
            continue
        summary = summarize_results(layer_df)
        # B2/B3 need the B1 sac_her baseline for the C5 recovery_score formula.
        # B0/B1 contain sac_her rows themselves so no external baseline is needed.
        baseline = b1_baseline if layer in {"B2", "B3"} else None
        summary = add_trustworthiness_scores(summary, baseline_summary=baseline)
        summary.insert(0, "benchmark_layer", layer)
        summary_parts.append(summary)

    summary_df = pd.concat(summary_parts, ignore_index=True)

    out_path = (
        args.output_file
        if args.output_file is not None
        else _derive_output_path(ep_path)
    )
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
