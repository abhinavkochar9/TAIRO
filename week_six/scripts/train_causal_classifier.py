"""
Phase 9, Phase B — train + evaluate the causal/online failure-mode
classifier against the Phase 8.5 post-hoc baseline (acc=0.996, macro-F1=
0.779). Same RF setup as scripts/train_failure_classifier.py (300 trees,
class_weight="balanced", seed 0-3 train / seed 4 test), applied to
results/classifier/causal_feature_matrix.csv (built by
scripts/build_causal_feature_matrix.py, Phase A) instead of the full-episode
aggregate feature_matrix.csv.

Does NOT save a model yet — Phase B is evaluation-only per the prompt; the
model is saved in Phase C once latency is also confirmed.

Unlike Phase 8.5 (one row per episode), this matrix has one row per
(episode_idx, checkpoint_t) — 14 checkpoints/episode, same hindsight label
repeated across all checkpoints of an episode. Reported both pooled
(all checkpoints together, directly comparable to Phase 8.5's per-episode
numbers) and broken out by checkpoint_t (to show the accuracy-over-time
curve, which matters for how early a recovery controller could act).
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                              f1_score, balanced_accuracy_score)

from config import CLASSIFIER_DIR as _DEFAULT_CLASSIFIER_DIR
from evaluation.failure_mode_labeling import ALL_LABELS
from evaluation.causal_features import FORBIDDEN_FEATURES

_parser = argparse.ArgumentParser()
_parser.add_argument("--classifier-dir", type=str, default=None)
_parser.add_argument("--baseline-acc", type=float, default=0.996,
                      help="Phase 8.5 post-hoc accuracy to diff against (default: 0.996)")
_parser.add_argument("--baseline-f1", type=float, default=0.779,
                      help="Phase 8.5 post-hoc macro-F1 to diff against (default: 0.779)")
_args = _parser.parse_args()
CLASSIFIER_DIR = _args.classifier_dir if _args.classifier_dir is not None else _DEFAULT_CLASSIFIER_DIR
_BASELINE_ACC = _args.baseline_acc
_BASELINE_F1 = _args.baseline_f1

TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]

feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "causal_feature_matrix.csv"))

NON_FEATURE_COLS = {"episode_idx", "checkpoint_t", "failure_mode", "condition", "seed", "model",
                    *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"

print(f"[phase9-B] Causal feature matrix: {len(feat_df)} rows "
      f"({feat_df['episode_idx'].nunique()} episode_idx values, not globally "
      f"unique across models — same convention as Phase 8.5)")
print(f"[phase9-B] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

train_df = feat_df[feat_df["seed"].isin(TRAIN_SEEDS)].copy()
test_df = feat_df[feat_df["seed"] == TEST_SEED].copy()

X_train, y_train = train_df[FEATURE_COLS].values, train_df["failure_mode"].values
X_test, y_test = test_df[FEATURE_COLS].values, test_df["failure_mode"].values

print(f"[phase9-B] Train: {len(X_train)} rows (seeds {TRAIN_SEEDS})")
print(f"[phase9-B] Test:  {len(X_test)} rows  (seed {TEST_SEED})\n")

# ── Leakage check ─────────────────────────────────────────────────────────
print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)
train_ep = set(train_df["episode_idx"])
test_ep = set(test_df["episode_idx"])
ep_overlap = train_ep & test_ep
print(f"  episode_idx overlap between train/test: {len(ep_overlap)} (expect 0)")
assert len(ep_overlap) == 0, f"LEAKAGE: {len(ep_overlap)} episode_idx values appear in both train and test"
# Rows are (episode_idx, checkpoint_t) pairs here, not one row per episode —
# also confirm no single episode contributes checkpoint rows to both splits.
train_pairs = set(zip(train_df["episode_idx"], train_df["checkpoint_t"]))
test_pairs = set(zip(test_df["episode_idx"], test_df["checkpoint_t"]))
print(f"  (episode_idx, checkpoint_t) row overlap: {len(train_pairs & test_pairs)} (expect 0)")
print()

# ── Class counts per split ────────────────────────────────────────────────
print("=" * 70)
print("CLASS COUNTS PER SPLIT (row-level, i.e. per checkpoint, not per episode)")
print("=" * 70)
train_counts = train_df["failure_mode"].value_counts().reindex(ALL_LABELS, fill_value=0)
test_counts = test_df["failure_mode"].value_counts().reindex(ALL_LABELS, fill_value=0)
counts_df = pd.DataFrame({"train": train_counts, "test": test_counts})
counts_df["train_pct"] = (100 * counts_df["train"] / counts_df["train"].sum()).round(2)
counts_df["test_pct"] = (100 * counts_df["test"] / counts_df["test"].sum()).round(2)
print(counts_df.to_string())
print()

# ── Majority-class baseline ──────────────────────────────────────────────
majority_label = train_df["failure_mode"].value_counts().idxmax()
y_majority = np.full(len(y_test), majority_label)
print("=" * 70)
print(f"BASELINE — majority class (always predicts '{majority_label}')")
print("=" * 70)
print(f"  Accuracy: {accuracy_score(y_test, y_majority):.3f}")
print(f"  Macro F1: {f1_score(y_test, y_majority, average='macro', zero_division=0):.3f}\n")

# ── Random Forest (pooled — all checkpoints together) ────────────────────
print("=" * 70)
print("RANDOM FOREST — CAUSAL FEATURES, POOLED ACROSS ALL CHECKPOINTS")
print("  (n_estimators=300, class_weight='balanced', seed=42 — same as Phase 8.5)")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(f"  Accuracy: {acc:.3f}   (baseline: {_BASELINE_ACC:.3f}, delta={acc-_BASELINE_ACC:+.3f})")
print(f"  Macro F1: {macro_f1:.3f}   (baseline: {_BASELINE_F1:.3f}, delta={macro_f1-_BASELINE_F1:+.3f})\n")

print("Per-class report:")
print(classification_report(y_test, y_pred, labels=ALL_LABELS, zero_division=0, digits=3))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred, labels=ALL_LABELS)
cm_df = pd.DataFrame(cm, index=[f"T:{l[:12]}" for l in ALL_LABELS],
                              columns=[f"P:{l[:12]}" for l in ALL_LABELS])
print(cm_df.to_string())
print()

# ── Balanced accuracy + bootstrap CI for macro-F1 ────────────────────────
print("=" * 70)
print("BALANCED ACCURACY + MACRO-F1 95% CI (bootstrap, n=2000 resamples)")
print("=" * 70)
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"  Balanced accuracy: {bal_acc:.3f}")
rng = np.random.default_rng(42)
n_test = len(y_test)
y_test_arr = np.asarray(y_test)
y_pred_arr = np.asarray(y_pred)
boot_f1 = []
for _ in range(2000):
    idx = rng.integers(0, n_test, n_test)
    boot_f1.append(f1_score(y_test_arr[idx], y_pred_arr[idx], average="macro", zero_division=0))
boot_f1 = np.array(boot_f1)
ci_lo, ci_hi = np.percentile(boot_f1, [2.5, 97.5])
print(f"  Macro-F1: {macro_f1:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
print("  NOTE: bootstrap resamples individual (episode_idx, checkpoint_t) rows, not whole "
      "episodes — rows from the same episode are correlated (14 checkpoints share one label), "
      "so this CI is narrower than a true episode-level CI. Treat as indicative, not exact.")
print()

# ── Feature importances ──────────────────────────────────────────────────
print("=" * 70)
print("FEATURE IMPORTANCES (causal model, top 20)")
print("=" * 70)
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(20).items():
    bar = "█" * int(val * 300)
    print(f"  {feat:<30}  {val:.4f}  {bar}")
print()

# ── Accuracy / macro-F1 as a function of checkpoint_t ────────────────────
print("=" * 70)
print("ACCURACY / MACRO-F1 BY CHECKPOINT (how early can this classifier act?)")
print("=" * 70)
test_df = test_df.copy()
test_df["pred"] = y_pred
for t, sub in test_df.groupby("checkpoint_t"):
    a = accuracy_score(sub["failure_mode"], sub["pred"])
    f = f1_score(sub["failure_mode"], sub["pred"], average="macro", zero_division=0)
    print(f"  t={t:>3}  n={len(sub):>5}  acc={a:.3f}  macro-F1={f:.3f}")
print()

# ── Per-class F1 by checkpoint, divergent_transport only (headline label) ─
print("=" * 70)
print("divergent_transport F1 BY CHECKPOINT")
print("=" * 70)
for t, sub in test_df.groupby("checkpoint_t"):
    f1_dt = f1_score(sub["failure_mode"], sub["pred"],
                      labels=["divergent_transport"], average="macro", zero_division=0)
    print(f"  t={t:>3}  divergent_transport F1={f1_dt:.3f}")
print()

# ── Held-out-condition generalization check (same conditions as Phase 8.5) ─
print("=" * 70)
print("GENERALIZATION CHECK — trained on 9 conditions, tested on 2 held-out")
print("  Held-out: object_pose_spoof, sensor_bias  (same as Phase 8.5 check)")
print("=" * 70)

HOLDOUT_CONDITIONS = ["object_pose_spoof", "sensor_bias"]
gen_train = feat_df[~feat_df["condition"].isin(HOLDOUT_CONDITIONS)].copy()
gen_test = feat_df[feat_df["condition"].isin(HOLDOUT_CONDITIONS)].copy()

rf_gen = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_gen.fit(gen_train[FEATURE_COLS].values, gen_train["failure_mode"].values)
y_pred_gen = rf_gen.predict(gen_test[FEATURE_COLS].values)
y_true_gen = gen_test["failure_mode"].values

acc_gen = accuracy_score(y_true_gen, y_pred_gen)
f1_gen = f1_score(y_true_gen, y_pred_gen, average="macro", zero_division=0)
print(f"  Train: {len(gen_train)} rows ({feat_df['condition'].nunique() - len(HOLDOUT_CONDITIONS)} conditions)")
print(f"  Test:  {len(gen_test)} rows  ({HOLDOUT_CONDITIONS})")
print(f"  Accuracy: {acc_gen:.3f}")
print(f"  Macro F1: {f1_gen:.3f}\n")

print("Per-class report (held-out conditions only):")
present_labels = [l for l in ALL_LABELS if l in y_true_gen]
print(classification_report(y_true_gen, y_pred_gen, labels=present_labels,
                             zero_division=0, digits=3))

f1_dt_gen = f1_score(y_true_gen, y_pred_gen, labels=["divergent_transport"],
                      average="macro", zero_division=0)
print(f"  divergent_transport F1 (held-out conditions): {f1_dt_gen:.3f}")
print(f"  (Phase 8.5 post-hoc equivalent: 0.016 — merge fixed seed-split "
      f"confusion, not the missing-condition generalization gap)")
print()

print("[phase9-B] Done. No model saved (Phase B is evaluation-only).")
