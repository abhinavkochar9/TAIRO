"""
Phase 9, Phase A — build the causal/rolling feature matrix and run the
sanity check (NaN/inf, feature ranges, checkpoint coverage). Does NOT train
anything — that is Phase B, gated on a separate go-ahead.

Reuses existing step_logs_*.csv (no new episode data, no re-run of the
sweep). Mirrors scripts/train_failure_classifier.py's loading/merge
structure but calls evaluation.causal_features.build_causal_features
instead of the full-episode aggregate builder.

Output
------
    results/classifier/causal_feature_matrix.csv
        one row per (episode_idx, checkpoint_t); labeled with the same
        Phase 8.5 failure_mode (hindsight label) via the labels_*.csv files.
    Console: feature list, per-feature NaN/inf counts, per-feature
        min/max/mean, row counts.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import DATA_DIR as _DEFAULT_DATA_DIR, CLASSIFIER_DIR as _DEFAULT_CLASSIFIER_DIR
from evaluation.causal_features import build_causal_features, FORBIDDEN_FEATURES

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-dir", type=str, default=None)
_parser.add_argument("--classifier-dir", type=str, default=None)
_args = _parser.parse_args()
DATA_DIR = _args.data_dir if _args.data_dir is not None else _DEFAULT_DATA_DIR
CLASSIFIER_DIR = _args.classifier_dir if _args.classifier_dir is not None else _DEFAULT_CLASSIFIER_DIR

os.makedirs(CLASSIFIER_DIR, exist_ok=True)

MODELS = ["clean_2M", "clean_500k", "randomized_2M", "randomized_500k"]

print("[phase9-A] Loading step logs and building CAUSAL feature matrix ...")
feat_frames = []

for model in MODELS:
    step_path = os.path.join(DATA_DIR, f"step_logs_sac_her_pickandplace_{model}.csv")
    label_path = os.path.join(DATA_DIR, f"labels_sac_her_pickandplace_{model}.csv")
    step_df = pd.read_csv(step_path)
    label_df = pd.read_csv(label_path)

    step_df = step_df[step_df["method"] == "sac_her"].copy()

    feats = build_causal_features(step_df)
    feats = feats.merge(
        label_df[["episode_idx", "failure_mode", "condition", "seed", "model"]],
        on="episode_idx", how="left",
    )
    feat_frames.append(feats)
    n_eps = feats["episode_idx"].nunique()
    print(f"  {model}: {n_eps} episodes -> {len(feats)} checkpoint-rows, {feats.shape[1]} cols")

feat_df = pd.concat(feat_frames, ignore_index=True)
print(f"\n  Combined: {len(feat_df)} rows across {feat_df['episode_idx'].nunique()} episodes\n")

feat_df.to_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"), index=False)

NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "condition", "seed", "model",
                    *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]

assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"

print("=" * 70)
print(f"FEATURE LIST ({len(FEATURE_COLS)} features)")
print("=" * 70)
for f in FEATURE_COLS:
    print(f"  {f}")

print()
print("=" * 70)
print("CHECKPOINT COVERAGE")
print("=" * 70)
print(feat_df["checkpoint_t"].value_counts().sort_index())

print()
print("=" * 70)
print("SANITY CHECK — NaN / inf counts per feature")
print("=" * 70)
X = feat_df[FEATURE_COLS]
nan_counts = X.isna().sum()
inf_counts = X.apply(lambda col: np.isinf(col.values).sum())
bad = pd.DataFrame({"nan": nan_counts, "inf": inf_counts})
bad = bad[(bad["nan"] > 0) | (bad["inf"] > 0)]
if len(bad) == 0:
    print("  No NaN or inf values in any feature. Clean.")
else:
    print(bad)

print()
print("=" * 70)
print("SANITY CHECK — feature ranges (min / mean / max)")
print("=" * 70)
desc = X.describe().T[["min", "mean", "max"]]
print(desc.to_string(float_format=lambda v: f"{v:.4f}"))

print()
print("=" * 70)
print("LABEL DISTRIBUTION (checkpoint-rows, hindsight label repeated per row)")
print("=" * 70)
print(feat_df["failure_mode"].value_counts())

print("\n[phase9-A] Done. Matrix saved -> "
      f"{os.path.join(CLASSIFIER_DIR, 'causal_feature_matrix.csv')}")
