"""
Diagnostic-only check, Step 2 (continuation of diag_wd_acc_separability.py).

Step 1 was inconclusive: stratified 5-fold showed real signal (macro-F1 0.647
vs 0.357 baseline) but the seed-based split (train 0-3, test seed=4, n=25)
scored *below* the majority baseline (0.278 vs 0.359) — small-test-set noise
vs. genuine non-separability could not be distinguished from the existing
34 full-episode-aggregate features alone.

This script adds a small number of TAIL-WINDOW trajectory-shape features,
computed only for the 128 wrong_direction / action_control_corruption
episodes, from the raw step logs. Does NOT touch or regenerate the full
6,600-episode feature matrix, and does NOT modify failure_mode_labeling.py,
episode_runner.py, config.py, or the Phase 8 classifier artifacts.

Feature choice rationale
-------------------------
The label boundary between wrong_direction and action_control_corruption is
`_is_diverging(dttg)` in failure_mode_labeling.py: the sign of the linear
slope of `distance_to_true_goal` over the final WRONG_DIR_WINDOW=50 steps.
Both labels are only reached *after* the grasp gate (never_reached_object,
reached_but_failed_grasp) and the no-drop gate (grasped_but_dropped) have
already passed — meaning distance_to_object is resolved early (object is
held) and stays low/flat for both classes for most of the episode. It carries
little separating signal for this specific pair. distance_to_true_goal is
where the actual behavioral difference between the two classes lives, so the
tail-window erraticism feature below uses distance_to_true_goal, not
distance_to_object.

Four new features (kept small — 128 rows, overfitting risk is real):
  tail_dttg_std       — std of distance_to_true_goal over final TAIL_WINDOW
                         steps. Erraticism/oscillation signal, distinct from
                         the existing full-episode dttg_slope/dttg_range.
  tail_dttg_slope     — slope of distance_to_true_goal over final
                         TAIL_WINDOW=30 steps (shorter horizon than the
                         label's own WRONG_DIR_WINDOW=50) — a sharper-focus
                         version of the signal the label itself is built on.
  tail_action_div     — mean |action_norm - intended_action_norm| over the
                         tail window. Motivated by the label docstring:
                         action_control_corruption = "elevated action-norm /
                         safety-violation signal" — existing features only
                         have full-episode action_corrupted_frac/action_ratio.
  tail_anrm_std        — std of action_norm over the tail window. Jitteriness
                         signal distinct from full-episode std_anrm; captures
                         unstable actuation late in the episode specifically.
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

from config import CLASSIFIER_DIR, DATA_DIR

TARGET_LABELS = ["wrong_direction", "action_control_corruption"]
TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]
TAIL_WINDOW = 30   # within the requested 20-30 step range

# ── Load Step 1 subset ─────────────────────────────────────────────────────
feat_df = pd.read_csv(os.path.join(CLASSIFIER_DIR, "feature_matrix.csv"))
sub = feat_df[feat_df["failure_mode"].isin(TARGET_LABELS)].copy()
assert len(sub) == 128
print(f"[step2] Base subset: {len(sub)} episodes across models: {sorted(sub['model'].unique())}\n")

# ── Compute tail-window features from raw step logs, per model ─────────────
tail_rows = []
for model in sorted(sub["model"].unique()):
    ep_ids = set(sub.loc[sub["model"] == model, "episode_idx"].tolist())
    step_path = os.path.join(DATA_DIR, f"step_logs_sac_her_pickandplace_{model}.csv")
    step_df = pd.read_csv(step_path)
    step_df = step_df[(step_df["method"] == "sac_her") & (step_df["episode_idx"].isin(ep_ids))]

    for ep_id, ep in step_df.groupby("episode_idx"):
        ep = ep.sort_values("timestep")
        dttg = ep["distance_to_true_goal"].values
        anrm = ep["action_norm"].values
        ianrm = ep["intended_action_norm"].values

        tail_dttg = dttg[-TAIL_WINDOW:]
        tail_anrm = anrm[-TAIL_WINDOW:]
        tail_ianrm = ianrm[-TAIL_WINDOW:]

        ts = np.arange(len(tail_dttg), dtype=float)
        tail_dttg_slope = float(np.polyfit(ts, tail_dttg, 1)[0]) if len(tail_dttg) >= 2 else 0.0

        tail_rows.append({
            "episode_idx": ep_id,
            "model": model,
            "tail_dttg_std": float(np.std(tail_dttg)),
            "tail_dttg_slope": tail_dttg_slope,
            "tail_action_div": float(np.mean(np.abs(tail_anrm - tail_ianrm))),
            "tail_anrm_std": float(np.std(tail_anrm)),
        })

tail_df = pd.DataFrame(tail_rows)
print(f"[step2] Tail features computed for {len(tail_df)} episodes (window={TAIL_WINDOW} steps).\n")

sub2 = sub.merge(tail_df, on=["episode_idx", "model"], how="left")
assert sub2["tail_dttg_std"].isna().sum() == 0, "Tail feature merge failed for some episodes."
assert len(sub2) == 128

# ── Sanity check: does dttg tail-erraticism separate the classes better than dto would? ─
print("-" * 70)
print("Sanity check: tail-window std by class (justifies dttg over dto choice)")
print("-" * 70)
print(sub2.groupby("failure_mode")[["tail_dttg_std", "tail_dttg_slope", "tail_action_div", "tail_anrm_std"]].mean().to_string())
print()

NEW_FEATURES = ["tail_dttg_std", "tail_dttg_slope", "tail_action_div", "tail_anrm_std"]
OLD_NON_FEATURE_COLS = {"episode_idx", "failure_mode", "condition", "seed", "model"}
FORBIDDEN_FEATURES = {"condition", "attack_level", "method"}
OLD_FEATURE_COLS = [c for c in sub.columns if c not in OLD_NON_FEATURE_COLS]
NEW_FEATURE_COLS = OLD_FEATURE_COLS + NEW_FEATURES
assert not any(f in NEW_FEATURE_COLS for f in FORBIDDEN_FEATURES)
print(f"[step2] Feature set: {len(OLD_FEATURE_COLS)} existing + {len(NEW_FEATURES)} new = {len(NEW_FEATURE_COLS)} total\n")


def run_seed_split(df, feature_cols, label):
    train_df = df[df["seed"].isin(TRAIN_SEEDS)]
    test_df = df[df["seed"] == TEST_SEED]
    X_train, y_train = train_df[feature_cols].values, train_df["failure_mode"].values
    X_test, y_test = test_df[feature_cols].values, test_df["failure_mode"].values

    majority_label = pd.Series(y_train).value_counts().idxmax()
    y_maj = np.full(len(y_test), majority_label)
    maj_acc = accuracy_score(y_test, y_maj)
    maj_f1 = f1_score(y_test, y_maj, average="macro", zero_division=0)

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"[{label}] SEED-BASED SPLIT (train 0-3, test seed 4, n_test={len(y_test)})")
    print(f"  Majority baseline: acc={maj_acc:.3f}  macro-F1={maj_f1:.3f}")
    print(f"  Random Forest:     acc={acc:.3f}  macro-F1={f1m:.3f}")
    print(classification_report(y_test, y_pred, labels=TARGET_LABELS, zero_division=0, digits=3))
    return {"maj_acc": maj_acc, "maj_f1": maj_f1, "acc": acc, "f1": f1m}, rf


def run_kfold(df, feature_cols, label, n_splits=5):
    X_all = df[feature_cols].values
    y_all = df["failure_mode"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs, fold_f1s, maj_fold_accs, maj_fold_f1s = [], [], [], []
    all_y_true, all_y_pred = [], []

    for tr_idx, te_idx in skf.split(X_all, y_all):
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        fold_majority = pd.Series(y_tr).value_counts().idxmax()
        y_maj_pred = np.full(len(y_te), fold_majority)
        maj_fold_accs.append(accuracy_score(y_te, y_maj_pred))
        maj_fold_f1s.append(f1_score(y_te, y_maj_pred, average="macro", zero_division=0))

        rf_fold = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                          class_weight="balanced", random_state=42, n_jobs=-1)
        rf_fold.fit(X_tr, y_tr)
        y_pred_fold = rf_fold.predict(X_te)
        fold_accs.append(accuracy_score(y_te, y_pred_fold))
        fold_f1s.append(f1_score(y_te, y_pred_fold, average="macro", zero_division=0))
        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred_fold)

    print(f"[{label}] STRATIFIED 5-FOLD (all 128 episodes)")
    print(f"  Majority baseline: mean acc={np.mean(maj_fold_accs):.3f}  mean macro-F1={np.mean(maj_fold_f1s):.3f}")
    print(f"  Random Forest:     mean acc={np.mean(fold_accs):.3f} (std {np.std(fold_accs):.3f})"
          f"  mean macro-F1={np.mean(fold_f1s):.3f} (std {np.std(fold_f1s):.3f})")
    print("  Pooled per-class report:")
    print(classification_report(all_y_true, all_y_pred, labels=TARGET_LABELS, zero_division=0, digits=3))
    return {
        "maj_acc": np.mean(maj_fold_accs), "maj_f1": np.mean(maj_fold_f1s),
        "acc": np.mean(fold_accs), "f1": np.mean(fold_f1s), "f1_std": np.std(fold_f1s),
    }


print("=" * 70)
print(f"BEFORE — existing {len(OLD_FEATURE_COLS)} features only (Step 1 reproduction)")
print("=" * 70)
before_seed, _ = run_seed_split(sub2, OLD_FEATURE_COLS, "BEFORE/seed-split")
print()
before_kfold = run_kfold(sub2, OLD_FEATURE_COLS, "BEFORE/k-fold")
print()

print("=" * 70)
print(f"AFTER — {len(OLD_FEATURE_COLS)} existing + {len(NEW_FEATURES)} tail-window features = {len(NEW_FEATURE_COLS)}")
print("=" * 70)
after_seed, rf_after = run_seed_split(sub2, NEW_FEATURE_COLS, "AFTER/seed-split")
print()
after_kfold = run_kfold(sub2, NEW_FEATURE_COLS, "AFTER/k-fold")
print()

print("=" * 70)
print("TOP-10 FEATURE IMPORTANCES — AFTER (seed-split RF, train seeds 0-3)")
print("=" * 70)
imp = pd.Series(rf_after.feature_importances_, index=NEW_FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(10).items():
    marker = "  <-- NEW" if feat in NEW_FEATURES else ""
    bar = "█" * int(val * 300)
    print(f"  {feat:<25}  {val:.4f}  {bar}{marker}")
print()

print("=" * 70)
print("BEFORE / AFTER SUMMARY")
print("=" * 70)
print(f"{'Split':<28}{'Majority F1':<14}{'Before F1':<14}{'After F1':<14}{'Δ (after-before)':<18}")
print(f"{'Seed-based (test seed 4)':<28}{before_seed['maj_f1']:<14.3f}{before_seed['f1']:<14.3f}{after_seed['f1']:<14.3f}{after_seed['f1']-before_seed['f1']:<18.3f}")
print(f"{'Stratified 5-fold':<28}{before_kfold['maj_f1']:<14.3f}{before_kfold['f1']:<14.3f}{after_kfold['f1']:<14.3f}{after_kfold['f1']-before_kfold['f1']:<18.3f}")
print()
print("[diag] Step 2 complete.")
