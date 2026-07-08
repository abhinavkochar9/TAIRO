"""
Diagnostic-only check (isolated task, NOT part of Phase 8/9 pipeline).

Question: are `wrong_direction` and `action_control_corruption` separable
using the existing Phase 8 feature set, or is the classifier confusion
between them (F1 0.452 / 0.100 on seed-4 test) evidence they're not
distinguishable trajectory-shapes at all?

Reads results/classifier/feature_matrix.csv (read-only, not regenerated).
Does not modify failure_mode_labeling.py, episode_runner.py, config.py,
or the Phase 8 classifier artifacts.

STEP 1: baseline separability with the existing 34 features.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import CLASSIFIER_DIR

TARGET_LABELS = ["wrong_direction", "action_control_corruption"]
TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]

# ── Load and filter ───────────────────────────────────────────────────────
feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "feature_matrix.csv"))
sub = feat_df[feat_df["failure_mode"].isin(TARGET_LABELS)].copy()

print("=" * 70)
print("STEP 1 — Baseline separability (existing 34 features)")
print("=" * 70)
print(f"Filtered rows: {len(sub)}  (expected 71 + 57 = 128)")
counts = sub["failure_mode"].value_counts()
print(counts.to_string())
assert len(sub) == 128, f"Expected 128 episodes, got {len(sub)} — STOP, discrepancy."
assert counts.get("wrong_direction", 0) == 71, f"wrong_direction count mismatch: {counts.get('wrong_direction', 0)}"
assert counts.get("action_control_corruption", 0) == 57, f"action_control_corruption count mismatch: {counts.get('action_control_corruption', 0)}"
print("Count check PASSED (71 + 57 = 128).\n")

# ── Feature columns — exclude condition/attack_level, assert programmatically ─
NON_FEATURE_COLS = {"episode_idx", "failure_mode", "condition", "seed", "model"}
FORBIDDEN_FEATURES = {"condition", "attack_level", "method"}
FEATURE_COLS = [c for c in sub.columns if c not in NON_FEATURE_COLS]
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"
print(f"Feature set: {len(FEATURE_COLS)} features (condition/attack_level excluded, asserted).\n")

# ── Per-seed class counts ─────────────────────────────────────────────────
print("-" * 70)
print("Per-seed class counts")
print("-" * 70)
seed_counts = sub.groupby(["seed", "failure_mode"]).size().unstack(fill_value=0)
print(seed_counts.to_string())
print()

train_df = sub[sub["seed"].isin(TRAIN_SEEDS)].copy()
test_df = sub[sub["seed"] == TEST_SEED].copy()
print(f"Seed-based split: train={len(train_df)} (seeds 0-3), test={len(test_df)} (seed 4)")
print(f"  Test class counts: {test_df['failure_mode'].value_counts().to_dict()}\n")

X_train = train_df[FEATURE_COLS].values
y_train = train_df["failure_mode"].values
X_test = test_df[FEATURE_COLS].values
y_test = test_df["failure_mode"].values

# ── Majority baseline (seed split) ────────────────────────────────────────
majority_label = train_df["failure_mode"].value_counts().idxmax()
y_majority = np.full(len(y_test), majority_label)
maj_acc = accuracy_score(y_test, y_majority)
maj_f1 = f1_score(y_test, y_majority, average="macro", zero_division=0)

print("=" * 70)
print("SEED-BASED SPLIT (train seeds 0-3, test seed 4)")
print("=" * 70)
print(f"Majority-class baseline (always '{majority_label}'): acc={maj_acc:.3f}  macro-F1={maj_f1:.3f}\n")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
print(f"Random Forest: acc={acc:.3f}  macro-F1={f1_macro:.3f}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, labels=TARGET_LABELS, zero_division=0, digits=3))

print("Top-10 feature importances (seed-split RF):")
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(10).items():
    bar = "█" * int(val * 300)
    print(f"  {feat:<25}  {val:.4f}  {bar}")
print()

# ── Stratified k-fold cross-check ─────────────────────────────────────────
print("=" * 70)
print("STRATIFIED K-FOLD CROSS-CHECK (5-fold, all 128 episodes, ignoring seed)")
print("=" * 70)

X_all = sub[FEATURE_COLS].values
y_all = sub["failure_mode"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs, fold_f1s = [], []
all_y_true, all_y_pred = [], []
maj_fold_accs, maj_fold_f1s = [], []

for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    fold_majority = pd.Series(y_tr).value_counts().idxmax()
    y_maj_pred = np.full(len(y_te), fold_majority)
    maj_fold_accs.append(accuracy_score(y_te, y_maj_pred))
    maj_fold_f1s.append(f1_score(y_te, y_maj_pred, average="macro", zero_division=0))

    rf_fold = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_fold.fit(X_tr, y_tr)
    y_pred_fold = rf_fold.predict(X_te)

    fold_accs.append(accuracy_score(y_te, y_pred_fold))
    fold_f1s.append(f1_score(y_te, y_pred_fold, average="macro", zero_division=0))
    all_y_true.extend(y_te)
    all_y_pred.extend(y_pred_fold)

print(f"Majority baseline across folds: mean acc={np.mean(maj_fold_accs):.3f}  mean macro-F1={np.mean(maj_fold_f1s):.3f}")
print(f"Random Forest across folds:     mean acc={np.mean(fold_accs):.3f} (std {np.std(fold_accs):.3f})"
      f"  mean macro-F1={np.mean(fold_f1s):.3f} (std {np.std(fold_f1s):.3f})\n")

print("Pooled per-class report (concatenated across 5 held-out folds):")
print(classification_report(all_y_true, all_y_pred, labels=TARGET_LABELS, zero_division=0, digits=3))

print("Top-10 feature importances (full-data RF, for reference):")
rf_full = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_full.fit(X_all, y_all)
imp_full = pd.Series(rf_full.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp_full.head(10).items():
    bar = "█" * int(val * 300)
    print(f"  {feat:<25}  {val:.4f}  {bar}")
print()

print("[diag] Step 1 complete.")
