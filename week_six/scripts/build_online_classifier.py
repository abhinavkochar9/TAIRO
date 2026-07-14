"""
Phase 9, Phase C — save the trained online/causal failure-mode classifier
and measure per-step online inference latency. Does NOT wire anything into
recovery/recovery_v2.py or recovery_v3.py — that integration is a separate,
not-yet-scoped prompt. This phase only confirms the classifier CAN run
online.

Model choice (open question flagged at the end of Phase B, resolved here):
trained on ALL 14 checkpoints pooled (not excluding near-terminal ones).
Rationale: a deployed recovery controller queries this model at every step
of a live rollout, including late ones — excluding late checkpoints from
training would make the saved model worse at exactly the steps it will
actually see in production. The Phase B caveat (late checkpoints look
almost like Phase 8.5's post-hoc full-episode features) is a caveat about
how to INTERPRET the pooled evaluation metric, not a reason to exclude that
data from training the deployed model.

Same train/test discipline as Phase B: seeds 0-3 train, seed 4 held out
(the saved model is NOT retrained on seed 4 — consistent with Phase 8.5).

Output
------
    results/classifier/online_failure_classifier.pkl
        {model, feature_cols, label_order, train_seeds, test_seed,
         checkpoint_policy, window_short, window_long}
    Console: latency numbers (feature extraction / predict / total), split
        out separately, plus percentiles across a real multi-episode,
        multi-step online simulation (not the precomputed batch matrix).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import DATA_DIR, CLASSIFIER_DIR, CAUSAL_WINDOW_SHORT, CAUSAL_WINDOW_LONG
from evaluation.failure_mode_labeling import ALL_LABELS
from evaluation.causal_features import (
    FORBIDDEN_FEATURES,
    build_causal_features_online,
)

TRAIN_SEEDS = [0, 1, 2, 3]
TEST_SEED = 4
MODELS = ["clean_2M", "clean_500k", "randomized_2M", "randomized_500k"]

# ── 1. Train the deployed model on the full causal feature matrix ─────────
print("[phase9-C] Loading causal feature matrix ...")
feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"))

NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "condition", "seed", "model",
                    *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"

train_df = feat_df[feat_df["seed"].isin(TRAIN_SEEDS)].copy()
print(f"[phase9-C] Training on {len(train_df)} rows (seeds {TRAIN_SEEDS}, all 14 checkpoints pooled)")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(train_df[FEATURE_COLS].values, train_df["failure_mode"].values)

# n_jobs is a runtime-only setting on a fitted sklearn estimator (does not
# change the trees or predictions) — training benefits from n_jobs=-1
# parallelism across 73,920 rows, but single-row online predict() pays
# thread-pool dispatch overhead on every call. Measured below: n_jobs=-1
# gives ~28ms/predict, n_jobs=1 gives ~9.6ms/predict for the SAME model.
# Save with n_jobs=1 since online serving is always single-row.
rf.n_jobs = 1

out = {
    "model": rf,
    "feature_cols": FEATURE_COLS,
    "label_order": ALL_LABELS,
    "train_seeds": TRAIN_SEEDS,
    "test_seed": TEST_SEED,
    "checkpoint_policy": "trained on all 14 per-episode checkpoints pooled "
                        "(t=19,29,...,149) — see module docstring for why "
                        "late checkpoints were NOT excluded",
    "window_short": CAUSAL_WINDOW_SHORT,
    "window_long": CAUSAL_WINDOW_LONG,
    "inference_n_jobs": 1,  # see comment above — deliberately not -1
}
pkl_path = os.path.join(CLASSIFIER_DIR, "online_failure_classifier.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(out, f)
print(f"[phase9-C] Model saved -> {pkl_path}  (n_jobs=1 for online serving)\n")

# ── 2. Latency measurement — real single-episode, single-step simulation ──
# Load raw step logs (not the precomputed batch matrix) to simulate genuine
# online use: growing a per-episode history buffer one step at a time and
# calling the same online extraction path Phase C is meant to validate.
print("[phase9-C] Building per-step online simulation for latency measurement ...")

sim_episodes = []  # list of per-episode step_df (sorted by timestep), test seed only
for model in MODELS:
    step_path = os.path.join(DATA_DIR, f"step_logs_sac_her_pickandplace_{model}.csv")
    step_df = pd.read_csv(step_path)
    step_df = step_df[(step_df["method"] == "sac_her") & (step_df["seed"] == TEST_SEED)].copy()
    # 5 episodes per model -> 20 episodes total, 150 steps each -> 3,000 single-step calls
    ep_ids = sorted(step_df["episode_idx"].unique())[:5]
    for ep_id in ep_ids:
        ep = step_df[step_df["episode_idx"] == ep_id].sort_values("timestep").reset_index(drop=True)
        sim_episodes.append(ep)

n_episodes = len(sim_episodes)
n_steps = sum(len(ep) for ep in sim_episodes)
print(f"[phase9-C] Simulating {n_episodes} episodes x ~150 steps = {n_steps} single-step calls\n")

def _stats(arr, label):
    print(f"  {label:<28} mean={arr.mean():.3f}ms  median={np.median(arr):.3f}ms  "
          f"p95={np.percentile(arr, 95):.3f}ms  p99={np.percentile(arr, 99):.3f}ms  "
          f"max={arr.max():.3f}ms")


def _run_latency_pass(model, label):
    extract_times, predict_times, total_times = [], [], []

    # Warm-up (avoid first-call overhead skewing percentiles)
    _warmup_ep = sim_episodes[0]
    for t in range(5, 15):
        feats = build_causal_features_online(_warmup_ep.iloc[: t + 1])
        x = np.array([[feats[c] for c in FEATURE_COLS]])
        model.predict(x)

    for ep in sim_episodes:
        for t in range(len(ep)):
            history = ep.iloc[: t + 1]  # only steps [0, t] — online-available history

            t0 = time.perf_counter()
            feats = build_causal_features_online(history)
            t1 = time.perf_counter()

            x = np.array([[feats[c] for c in FEATURE_COLS]])
            model.predict(x)
            t2 = time.perf_counter()

            extract_times.append((t1 - t0) * 1000)
            predict_times.append((t2 - t1) * 1000)
            total_times.append((t2 - t0) * 1000)

    extract_times, predict_times, total_times = map(np.array, (extract_times, predict_times, total_times))
    print("=" * 70)
    print(f"PER-STEP ONLINE INFERENCE LATENCY — {label}  (n={len(total_times)} single-step "
          f"calls, single episode/single step, not batched)")
    print("=" * 70)
    _stats(extract_times, "Feature extraction")
    _stats(predict_times, "RF predict (1 row)")
    _stats(total_times, "Total (extract+predict)")
    print(f"  -> {total_times.mean():.3f}ms mean = {total_times.mean()/50*100:.1f}% of a "
          f"50ms (20Hz) control-loop budget\n")
    return total_times


# Two passes on the SAME trained trees — only n_jobs differs — to make the
# parallelization-overhead finding explicit rather than silently shipping
# whichever setting training happened to leave behind.
import copy
rf_parallel = copy.deepcopy(rf)
rf_parallel.n_jobs = -1

_run_latency_pass(rf_parallel, "n_jobs=-1 (sklearn training default)")
saved_total_times = _run_latency_pass(rf, "n_jobs=1 (SAVED CONFIG — used in online_failure_classifier.pkl)")

print("\n[phase9-C] Done. Files:")
print(f"  Model:      {pkl_path}")
print(f"  Extraction: evaluation/causal_features.py (build_causal_features_online)")
