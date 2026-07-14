"""
Phase 7 — Generate failure-mode labels at scale.

Runs label_batch() over the full Phase 6 step logs (all 4 models,
sac_her only, 6,600 episodes total) and writes per-episode label CSVs.

Output
------
    results/data/labels_sac_her_pickandplace_<model>.csv
        Columns: episode_idx, failure_mode, condition, seed, success,
                 final_distance, first_success_step

    results/data/labels_sac_her_pickandplace_all.csv
        Same, concatenated across all 4 models with a `model` column.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from evaluation.failure_mode_labeling import label_batch, ALL_LABELS
from config import DATA_DIR as _DEFAULT_DATA_DIR

MODELS = ["clean_2M", "clean_500k", "randomized_2M", "randomized_500k"]

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-dir", type=str, default=None,
                      help="Override input/output data directory (default: DATA_DIR from config.py)")
_args = _parser.parse_args()
DATA_DIR = _args.data_dir if _args.data_dir is not None else _DEFAULT_DATA_DIR

os.makedirs(DATA_DIR, exist_ok=True)

all_frames = []

for model in MODELS:
    step_path = os.path.join(DATA_DIR, f"step_logs_sac_her_pickandplace_{model}.csv")
    ep_path   = os.path.join(DATA_DIR, f"episode_results_sac_her_pickandplace_{model}.csv")

    print(f"\n[phase7] Labeling {model} ...")
    step_df = pd.read_csv(step_path)
    ep_df   = pd.read_csv(ep_path)

    # Filter to sac_her only (resweep already is, but guard against any extras)
    step_df = step_df[step_df["method"] == "sac_her"].copy()
    ep_df   = ep_df[ep_df["method"] == "sac_her"].copy()

    labels_df = label_batch(step_df, episode_col="episode_idx")

    # Merge with episode metadata
    meta = ep_df[["episode_idx", "condition", "seed", "success",
                  "final_distance", "first_success_step"]].copy()
    merged = labels_df.merge(meta, on="episode_idx", how="left")
    merged["model"] = model

    out_path = os.path.join(DATA_DIR, f"labels_sac_her_pickandplace_{model}.csv")
    merged.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({len(merged)} episodes)")

    # Label distribution for this model
    dist = merged.groupby(["condition", "failure_mode"]).size().unstack(
        fill_value=0).reindex(columns=ALL_LABELS, fill_value=0)
    print(dist.to_string())

    all_frames.append(merged)

# Combined across all models
combined = pd.concat(all_frames, ignore_index=True)
combined_path = os.path.join(DATA_DIR, "labels_sac_her_pickandplace_all.csv")
combined.to_csv(combined_path, index=False)
print(f"\n[phase7] Combined → {combined_path}  ({len(combined)} total episodes)")

# ── Aggregate distribution table ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("AGGREGATE LABEL DISTRIBUTION (all 4 models combined)")
print("=" * 80)
agg = combined.groupby(["condition", "failure_mode"]).size().unstack(
    fill_value=0).reindex(columns=ALL_LABELS, fill_value=0)
agg["TOTAL"] = agg.sum(axis=1)
print(agg.to_string())

print("\n" + "=" * 80)
print("LABEL TOTALS ACROSS ALL CONDITIONS")
print("=" * 80)
totals = combined["failure_mode"].value_counts().reindex(ALL_LABELS, fill_value=0)
for label, count in totals.items():
    pct = 100 * count / len(combined)
    print(f"  {label:<35}  {count:>5}  ({pct:.1f}%)")

print("\n" + "=" * 80)
print("RARE / MISSING LABELS (< 1% of episodes)")
print("=" * 80)
rare = totals[totals < 0.01 * len(combined)]
if rare.empty:
    print("  None")
else:
    for label, count in rare.items():
        print(f"  {label:<35}  {count:>5}")

# ── Class imbalance per condition ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("DOMINANT LABEL PER CONDITION (most common across all models)")
print("=" * 80)
for cond in combined["condition"].unique():
    sub = combined[combined["condition"] == cond]["failure_mode"]
    dominant = sub.value_counts().idxmax()
    pct = 100 * sub.value_counts().max() / len(sub)
    print(f"  {cond:<30}  {dominant}  ({pct:.0f}%)")
