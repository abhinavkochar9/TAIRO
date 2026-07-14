"""
Phase 8 — Train failure_mode_classifier.py (post-hoc, full-episode features).

Models
------
  Primary  : RandomForestClassifier  (interpretable via feature importance)
  Baseline : MajorityClassClassifier (always predicts most frequent label)

Split discipline
----------------
  Train: seeds 0–3   (80%)
  Test:  seed  4     (20%)
  Episodes within a seed are never split across train/test — no row-level leakage.

  Secondary generalization check: re-evaluate on the two most complex held-out
  conditions (object_pose_spoof, sensor_bias) with model trained on the other 9.

Hard constraint
---------------
  `condition` and `attack_level` are NEVER features — excluded programmatically
  and asserted before training.

Output
------
  results/classifier/failure_mode_classifier.pkl
  results/classifier/feature_matrix.csv            (episode_idx + features + label)
  Console: accuracy, per-class F1, confusion matrix, feature importances
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR as _DEFAULT_DATA_DIR, CLASSIFIER_DIR as _DEFAULT_CLASSIFIER_DIR
from evaluation.failure_mode_labeling import ALL_LABELS

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-dir", type=str, default=None,
                      help="Input data directory (default: DATA_DIR from config.py)")
_parser.add_argument("--classifier-dir", type=str, default=None,
                      help="Output classifier directory (default: CLASSIFIER_DIR from config.py)")
_args = _parser.parse_args()
DATA_DIR = _args.data_dir if _args.data_dir is not None else _DEFAULT_DATA_DIR
CLASSIFIER_DIR = _args.classifier_dir if _args.classifier_dir is not None else _DEFAULT_CLASSIFIER_DIR

os.makedirs(CLASSIFIER_DIR, exist_ok=True)

MODELS = ["clean_2M", "clean_500k", "randomized_2M", "randomized_500k"]
TEST_SEED = 4
TRAIN_SEEDS = [0, 1, 2, 3]

# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(step_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-episode aggregate features from the step log.
    Returns one row per episode_idx. Never uses condition or attack_level.
    """
    rows = []
    for ep_id, ep in step_df.groupby("episode_idx"):
        ep = ep.sort_values("timestep")
        n  = len(ep)

        dto  = ep["distance_to_object"].values
        dttg = ep["distance_to_true_goal"].values
        dtpg = ep["distance_to_perceived_goal"].values
        anrm = ep["action_norm"].values
        ianrm= ep["intended_action_norm"].values
        aper = ep["gripper_aperture"].values
        ovz  = ep["object_velp_z"].values
        svis = ep["safety_violation"].values
        isucc= ep["is_success"].values
        rew  = ep["reward"].values

        # Slope helpers (linear regression over full episode)
        ts = np.arange(n, dtype=float)

        def _slope(y):
            if len(y) < 2:
                return 0.0
            return float(np.polyfit(ts[:len(y)], y, 1)[0])

        # ── Distance: gripper → object ───────────────────────────────────────
        min_dto   = float(dto.min())
        mean_dto  = float(dto.mean())
        final_dto = float(dto[-1])
        dto_slope = _slope(dto)

        # ── Distance: object → true goal ─────────────────────────────────────
        min_dttg    = float(dttg.min())
        max_dttg    = float(dttg.max())
        mean_dttg   = float(dttg.mean())
        final_dttg  = float(dttg[-1])
        initial_dttg= float(dttg[0])
        dttg_range  = max_dttg - min_dttg
        dttg_slope  = _slope(dttg)
        # Fraction of episode during which object was within 0.05m of true goal
        near_goal_frac = float(np.mean(dttg < 0.05))

        # ── Perceived vs true goal divergence (signal for goal-spoof) ────────
        goal_offset   = np.abs(dtpg - dttg)
        goal_offset_max  = float(goal_offset.max())
        goal_offset_mean = float(goal_offset.mean())
        min_dtpg      = float(dtpg.min())
        final_dtpg    = float(dtpg[-1])

        # ── Action norms ──────────────────────────────────────────────────────
        mean_anrm  = float(anrm.mean())
        max_anrm   = float(anrm.max())
        std_anrm   = float(anrm.std())
        anrm_slope = _slope(anrm)
        # Ratio: how much does executed differ from intended?
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ianrm > 1e-6, anrm / ianrm, 1.0)
        mean_action_ratio = float(np.nanmean(ratio))
        # Fraction of steps where intended ≠ executed significantly
        action_corrupted_frac = float(np.mean(np.abs(anrm - ianrm) > 0.10))

        # ── Gripper aperture ──────────────────────────────────────────────────
        mean_aper = float(aper.mean())
        std_aper  = float(aper.std())
        max_aper  = float(aper.max())
        min_aper  = float(aper.min())

        # ── Object vertical velocity (lift signal) ────────────────────────────
        max_ovz  = float(ovz.max())
        mean_ovz = float(ovz.mean())

        # ── Safety / success ──────────────────────────────────────────────────
        safety_violation_rate = float(svis.mean())
        is_success_any  = float(isucc.max())   # transient success ever
        n_success_steps = int(np.sum(isucc > 0))
        final_is_success= float(isucc[-1])

        # ── Reward ────────────────────────────────────────────────────────────
        total_reward = float(rew.sum())
        mean_reward  = float(rew.mean())

        rows.append({
            "episode_idx": ep_id,
            # distance: gripper→object
            "min_dto":             min_dto,
            "mean_dto":            mean_dto,
            "final_dto":           final_dto,
            "dto_slope":           dto_slope,
            # distance: object→true goal
            "min_dttg":            min_dttg,
            "max_dttg":            max_dttg,
            "mean_dttg":           mean_dttg,
            "final_dttg":          final_dttg,
            "initial_dttg":        initial_dttg,
            "dttg_range":          dttg_range,
            "dttg_slope":          dttg_slope,
            "near_goal_frac":      near_goal_frac,
            # goal spoof signal
            "goal_offset_max":     goal_offset_max,
            "goal_offset_mean":    goal_offset_mean,
            "min_dtpg":            min_dtpg,
            "final_dtpg":          final_dtpg,
            # action norms
            "mean_anrm":           mean_anrm,
            "max_anrm":            max_anrm,
            "std_anrm":            std_anrm,
            "anrm_slope":          anrm_slope,
            "mean_action_ratio":   mean_action_ratio,
            "action_corrupted_frac": action_corrupted_frac,
            # gripper
            "mean_aper":           mean_aper,
            "std_aper":            std_aper,
            "max_aper":            max_aper,
            "min_aper":            min_aper,
            # lift
            "max_ovz":             max_ovz,
            "mean_ovz":            mean_ovz,
            # safety / success
            "safety_violation_rate": safety_violation_rate,
            "is_success_any":      is_success_any,
            "n_success_steps":     n_success_steps,
            "final_is_success":    final_is_success,
            # reward
            "total_reward":        total_reward,
            "mean_reward":         mean_reward,
        })

    return pd.DataFrame(rows)


FORBIDDEN_FEATURES = {"condition", "attack_level", "method"}

# ── Load and build feature matrix ─────────────────────────────────────────────
print("[phase8] Loading step logs and building feature matrix ...")
feat_frames = []

for model in MODELS:
    step_path  = os.path.join(DATA_DIR, f"step_logs_sac_her_pickandplace_{model}.csv")
    label_path = os.path.join(DATA_DIR, f"labels_sac_her_pickandplace_{model}.csv")
    step_df  = pd.read_csv(step_path)
    label_df = pd.read_csv(label_path)

    # Filter to sac_her only
    step_df = step_df[step_df["method"] == "sac_her"].copy()

    feats = build_features(step_df)
    feats = feats.merge(
        label_df[["episode_idx", "failure_mode", "condition", "seed", "model"]],
        on="episode_idx", how="left"
    )
    feat_frames.append(feats)
    print(f"  {model}: {len(feats)} episodes, {feats.shape[1]} cols")

feat_df = pd.concat(feat_frames, ignore_index=True)
print(f"  Combined: {len(feat_df)} episodes\n")

# Save feature matrix (includes condition as metadata, never as feature)
feat_df.to_csv(os.path.join(CLASSIFIER_DIR, "feature_matrix.csv"), index=False)

# ── Define feature columns ────────────────────────────────────────────────────
NON_FEATURE_COLS = {"episode_idx", "failure_mode", "condition", "seed", "model",
                    *FORBIDDEN_FEATURES}
FEATURE_COLS = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]

# Hard assertion: no forbidden columns leaked into features
assert not any(f in FEATURE_COLS for f in FORBIDDEN_FEATURES), \
    f"Forbidden feature leaked: {set(FEATURE_COLS) & FORBIDDEN_FEATURES}"
print(f"[phase8] Features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

# ── Train / test split by seed ────────────────────────────────────────────────
train_df = feat_df[feat_df["seed"].isin(TRAIN_SEEDS)].copy()
test_df  = feat_df[feat_df["seed"] == TEST_SEED].copy()

X_train = train_df[FEATURE_COLS].values
y_train = train_df["failure_mode"].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["failure_mode"].values

print(f"[phase8] Train: {len(X_train)} episodes (seeds {TRAIN_SEEDS})")
print(f"[phase8] Test:  {len(X_test)} episodes  (seed {TEST_SEED})\n")

# ── Leakage check ──────────────────────────────────────────────────────────
print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)
train_idx = set(train_df["episode_idx"])
test_idx  = set(test_df["episode_idx"])
overlap = train_idx & test_idx
print(f"  episode_idx overlap between train/test: {len(overlap)} (expect 0)")
assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} episode_idx values appear in both train and test"
# Seed-fix-specific check: with the paired-spawn design (reset_seed = 100*seed +
# episode_in_seed), the same 30 physical spawns recur across all 11 conditions
# WITHIN a seed. Confirm train (seeds 0-3) and test (seed 4) draw from disjoint
# reset-seed ranges, i.e. no physical-spawn leakage across the split either.
if "initial_dttg" in feat_df.columns:
    train_d = set(train_df["initial_dttg"].round(6))
    test_d  = set(test_df["initial_dttg"].round(6))
    d_overlap = train_d & test_d
    print(f"  initial_dttg (spawn-signature) overlap between train/test: {len(d_overlap)} "
          f"(expect 0 — disjoint reset-seed ranges confirm no physical-spawn leakage)")
print()

# ── Class counts per split ────────────────────────────────────────────────
print("=" * 70)
print("CLASS COUNTS PER SPLIT")
print("=" * 70)
train_counts = train_df["failure_mode"].value_counts().reindex(ALL_LABELS, fill_value=0)
test_counts  = test_df["failure_mode"].value_counts().reindex(ALL_LABELS, fill_value=0)
counts_df = pd.DataFrame({"train": train_counts, "test": test_counts})
counts_df["train_pct"] = (100 * counts_df["train"] / counts_df["train"].sum()).round(2)
counts_df["test_pct"]  = (100 * counts_df["test"]  / counts_df["test"].sum()).round(2)
print(counts_df.to_string())
print()


# ── Majority-class baseline ───────────────────────────────────────────────────
majority_label = train_df["failure_mode"].value_counts().idxmax()
y_majority = np.full(len(y_test), majority_label)

print("=" * 70)
print(f"BASELINE — majority class (always predicts '{majority_label}')")
print("=" * 70)
print(f"  Accuracy: {accuracy_score(y_test, y_majority):.3f}")
print(f"  Macro F1: {f1_score(y_test, y_majority, average='macro', zero_division=0):.3f}")
print()


# ── Random Forest ─────────────────────────────────────────────────────────────
print("=" * 70)
print("RANDOM FOREST  (n_estimators=300, class_weight='balanced', seed=42)")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"  Macro F1: {f1_score(y_test, y_pred_rf, average='macro', zero_division=0):.3f}")
print()
print("Per-class report:")
print(classification_report(y_test, y_pred_rf, labels=ALL_LABELS,
                             zero_division=0, digits=3))

print("Confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, y_pred_rf, labels=ALL_LABELS)
cm_df = pd.DataFrame(cm, index=[f"T:{l[:12]}" for l in ALL_LABELS],
                              columns=[f"P:{l[:12]}" for l in ALL_LABELS])
print(cm_df.to_string())
print()

# ── Balanced accuracy + bootstrap CI for macro-F1 ─────────────────────────────
print("=" * 70)
print("BALANCED ACCURACY + MACRO-F1 95% CI (bootstrap, n=2000 resamples)")
print("=" * 70)
bal_acc = balanced_accuracy_score(y_test, y_pred_rf)
print(f"  Balanced accuracy: {bal_acc:.3f}")

rng = np.random.default_rng(42)
n_test = len(y_test)
boot_f1 = []
y_test_arr = np.asarray(y_test)
y_pred_arr = np.asarray(y_pred_rf)
for _ in range(2000):
    idx = rng.integers(0, n_test, n_test)
    boot_f1.append(f1_score(y_test_arr[idx], y_pred_arr[idx], average="macro", zero_division=0))
boot_f1 = np.array(boot_f1)
ci_lo, ci_hi = np.percentile(boot_f1, [2.5, 97.5])
print(f"  Macro-F1: {f1_score(y_test, y_pred_rf, average='macro', zero_division=0):.3f}  "
      f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
print()


# ── Feature importances ───────────────────────────────────────────────────────
print("=" * 70)
print("FEATURE IMPORTANCES (top 20)")
print("=" * 70)
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, val in imp.head(20).items():
    bar = "█" * int(val * 300)
    print(f"  {feat:<30}  {val:.4f}  {bar}")
print()


# ── Logistic Regression baseline ──────────────────────────────────────────────
print("=" * 70)
print("LOGISTIC REGRESSION  (max_iter=1000, class_weight='balanced')")
print("=" * 70)

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs",
    C=1.0,
)
# Standardize for LR
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"  Macro F1: {f1_score(y_test, y_pred_lr, average='macro', zero_division=0):.3f}")
print()
print("Per-class report:")
print(classification_report(y_test, y_pred_lr, labels=ALL_LABELS,
                             zero_division=0, digits=3))


# ── Secondary: condition generalization check ──────────────────────────────────
print("=" * 70)
print("GENERALIZATION CHECK — model trained on 9 conditions, tested on 2 held-out")
print("  Held-out: object_pose_spoof, sensor_bias")
print("=" * 70)

HOLDOUT_CONDITIONS = ["object_pose_spoof", "sensor_bias"]
gen_train = feat_df[~feat_df["condition"].isin(HOLDOUT_CONDITIONS)].copy()
gen_test  = feat_df[feat_df["condition"].isin(HOLDOUT_CONDITIONS)].copy()

rf_gen = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_gen.fit(gen_train[FEATURE_COLS].values, gen_train["failure_mode"].values)
y_pred_gen = rf_gen.predict(gen_test[FEATURE_COLS].values)
y_true_gen = gen_test["failure_mode"].values

print(f"  Train: {len(gen_train)} eps ({len(feat_df['condition'].unique()) - len(HOLDOUT_CONDITIONS)} conditions)")
print(f"  Test:  {len(gen_test)} eps  ({HOLDOUT_CONDITIONS})")
print(f"  Accuracy: {accuracy_score(y_true_gen, y_pred_gen):.3f}")
print(f"  Macro F1: {f1_score(y_true_gen, y_pred_gen, average='macro', zero_division=0):.3f}")
print()
print("Per-class report (held-out conditions only):")
present_labels = [l for l in ALL_LABELS if l in y_true_gen]
print(classification_report(y_true_gen, y_pred_gen, labels=present_labels,
                             zero_division=0, digits=3))


# ── Per-model breakdown (seed-split RF) ───────────────────────────────────────
print("=" * 70)
print("PER-MODEL ACCURACY (seed-split RF, test on seed 4)")
print("=" * 70)
for model in MODELS:
    sub = test_df[test_df["model"] == model]
    if len(sub) == 0:
        continue
    y_t = sub["failure_mode"].values
    y_p = rf.predict(sub[FEATURE_COLS].values)
    acc = accuracy_score(y_t, y_p)
    f1  = f1_score(y_t, y_p, average="macro", zero_division=0)
    print(f"  {model:<25}  acc={acc:.3f}  macro-F1={f1:.3f}  n={len(sub)}")
print()

# ── Save model ────────────────────────────────────────────────────────────────
out = {
    "model": rf,
    "feature_cols": FEATURE_COLS,
    "label_order": ALL_LABELS,
    "train_seeds": TRAIN_SEEDS,
    "test_seed": TEST_SEED,
}
pkl_path = os.path.join(CLASSIFIER_DIR, "failure_mode_classifier.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(out, f)
print(f"[phase8] Model saved → {pkl_path}")
print("[phase8] Done.")
