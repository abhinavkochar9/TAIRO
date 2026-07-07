"""
Phase 2 diagnostic: logging sanity check for new spatial fields.

Runs sac_her only, all 11 conditions, seed=0, N_DIAG_EPISODES episodes each,
using the clean_2M PickAndPlace model.  Outputs:
  results/data/diag_step_logs_phase2.csv
  results/data/diag_episode_results_phase2.csv
  results/figures/diag_phase2_<condition>.png  (one per condition)

Do NOT use this script for a full sweep — it is scoped for visual inspection only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ALL_CONDITIONS, ATTACK_LEVELS, DATA_DIR, FIGURES_DIR, MODELS_DIR
from envs.fetchpickandplace_env import make_env
from evaluation.episode_runner import run_episode

# ---------------------------------------------------------------------------
# Diagnostic constants — intentionally small, do not change for full sweep
# ---------------------------------------------------------------------------
DIAG_MODEL_PATH  = os.path.join(MODELS_DIR, "sac_her_pickandplace_clean_2M")
DIAG_SEED        = 0
N_DIAG_EPISODES  = 3
DIAG_MAX_STEPS   = 150
DIAG_METHOD      = "sac_her"
DIAG_OUT_DIR     = os.path.join(FIGURES_DIR, "diag_phase2")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DIAG_OUT_DIR, exist_ok=True)

    from stable_baselines3 import SAC
    tmp_env = make_env(seed=0)
    model = SAC.load(DIAG_MODEL_PATH, env=tmp_env)
    tmp_env.close()
    print(f"[diag] Loaded model: {DIAG_MODEL_PATH}")

    episode_rows = []
    all_step_dfs = []
    episode_idx  = 0

    env = make_env(seed=DIAG_SEED)

    for condition in ALL_CONDITIONS:
        attack_level = ATTACK_LEVELS[condition]
        print(f"[diag] condition={condition}  attack_level={attack_level}")

        for ep in range(N_DIAG_EPISODES):
            result, step_df = run_episode(
                env=env,
                method=DIAG_METHOD,
                seed=DIAG_SEED,
                condition=condition,
                attack_level=attack_level,
                model=model,
                max_steps=DIAG_MAX_STEPS,
            )

            row = vars(result).copy()
            row["episode_idx"] = episode_idx
            row["env"]         = "diag_clean_2M"
            episode_rows.append(row)

            step_df = step_df.copy()
            step_df["episode_idx"] = episode_idx
            step_df["env"]         = "diag_clean_2M"
            all_step_dfs.append(step_df)

            episode_idx += 1

        print(f"          success={result.success}  "
              f"first_success_step={result.first_success_step}  "
              f"final_distance={result.final_distance:.4f}")

    env.close()

    episode_df  = pd.DataFrame(episode_rows)
    step_df_all = pd.concat(all_step_dfs, ignore_index=True)

    ep_path   = os.path.join(DATA_DIR, "diag_episode_results_phase2.csv")
    step_path = os.path.join(DATA_DIR, "diag_step_logs_phase2.csv")
    episode_df.to_csv(ep_path,   index=False)
    step_df_all.to_csv(step_path, index=False)
    print(f"\n[diag] Saved episode results → {ep_path}")
    print(f"[diag] Saved step logs       → {step_path}")
    print(f"[diag] New step log columns: {[c for c in step_df_all.columns if c not in ('method','condition','seed','attack_level','timestep','reward','distance_to_goal','is_success','action_norm','intended_action_norm','safety_violation','recovery_triggered','episode_idx','env')]}")

    # -----------------------------------------------------------------------
    # Sanity-check prints
    # -----------------------------------------------------------------------
    print("\n[diag] === SANITY CHECKS ===")

    # 1. gripper_aperture — should vary across steps (not flat)
    aper = step_df_all["gripper_aperture"].dropna()
    print(f"gripper_aperture  min={aper.min():.4f}  max={aper.max():.4f}  "
          f"std={aper.std():.4f}  (non-flat: {aper.std() > 0.001})")

    # 2. object_pos — check it changes over the episode (not static)
    clean_ep0 = step_df_all[
        (step_df_all["condition"] == "clean") & (step_df_all["episode_idx"] == 0)
    ]
    obj_z = clean_ep0["object_pos_z"]
    print(f"object_pos_z (clean ep0)  min={obj_z.min():.4f}  max={obj_z.max():.4f}  "
          f"std={obj_z.std():.4f}  (moves: {obj_z.std() > 0.001})")

    # 3a. goal_spoof_immediate — perceived_goal should differ from true_goal from step 0
    gsi = step_df_all[step_df_all["condition"] == "goal_spoof_immediate"]
    if not gsi.empty:
        diff_x = (gsi["perceived_goal_x"] - gsi["true_goal_x"]).abs()
        print(f"goal_spoof_immediate  perceived-vs-true diff_x  "
              f"min={diff_x.min():.4f}  max={diff_x.max():.4f}  "
              f"(non-zero from step 0: {diff_x.iloc[0] > 0.001})")

    # 3b. goal_spoof_midep — perceived_goal should equal true_goal before step 60, differ after
    gsm = step_df_all[
        (step_df_all["condition"] == "goal_spoof_midep") &
        (step_df_all["episode_idx"] == step_df_all[step_df_all["condition"] == "goal_spoof_midep"]["episode_idx"].min())
    ]
    if not gsm.empty:
        before = gsm[gsm["timestep"] < 60]
        after  = gsm[gsm["timestep"] >= 60]
        diff_before = (before["perceived_goal_x"] - before["true_goal_x"]).abs().max() if not before.empty else float("nan")
        diff_after  = (after["perceived_goal_x"]  - after["true_goal_x"]).abs().max()  if not after.empty  else float("nan")
        print(f"goal_spoof_midep  max_diff before step 60: {diff_before:.4f}  "
              f"after step 60: {diff_after:.4f}  "
              f"(onset correct: before≈0 and after>0: {diff_before < 0.001 and diff_after > 0.001})")

    # -----------------------------------------------------------------------
    # Plots — one PNG per condition, episode 0 of each
    # -----------------------------------------------------------------------
    print("\n[diag] Generating plots...")
    ep0_per_condition = (
        step_df_all.groupby("condition")["episode_idx"].min().reset_index()
    )

    for _, row in ep0_per_condition.iterrows():
        cond = row["condition"]
        ep_id = row["episode_idx"]
        ep_data = step_df_all[step_df_all["episode_idx"] == ep_id].copy()

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Phase 2 diagnostic — condition: {cond}  (episode_idx={ep_id})",
                     fontsize=11)

        axes[0].plot(ep_data["timestep"], ep_data["distance_to_object"], color="steelblue")
        axes[0].set_ylabel("distance_to_object (m)")
        axes[0].set_title("Gripper → Object distance")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ep_data["timestep"], ep_data["distance_to_true_goal"],
                     color="forestgreen", label="to true goal")
        axes[1].plot(ep_data["timestep"], ep_data["distance_to_goal"],
                     color="limegreen", linestyle="--", alpha=0.5, label="distance_to_goal (alias check)")
        axes[1].set_ylabel("distance (m)")
        axes[1].set_title("Object → True Goal distance")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(ep_data["timestep"], ep_data["distance_to_perceived_goal"],
                     color="tomato", label="to perceived goal")
        axes[2].plot(ep_data["timestep"], ep_data["distance_to_true_goal"],
                     color="forestgreen", linestyle="--", alpha=0.5, label="to true goal")
        axes[2].set_ylabel("distance (m)")
        axes[2].set_title("Object → Perceived Goal vs True Goal")
        axes[2].set_xlabel("Timestep")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        # Mark success steps
        success_steps = ep_data[ep_data["is_success"] == 1.0]["timestep"]
        if not success_steps.empty:
            for ax in axes:
                ax.axvline(success_steps.iloc[0], color="gold", linewidth=1.5,
                           linestyle=":", label="first success" if ax == axes[0] else "")

        plt.tight_layout()
        out_path = os.path.join(DIAG_OUT_DIR, f"diag_phase2_{cond}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved → {out_path}")

    print("\n[diag] Done.")


if __name__ == "__main__":
    main()
