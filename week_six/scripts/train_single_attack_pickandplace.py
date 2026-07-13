"""
Train a single-attack binary-flag SAC+HER policy on FetchPickAndPlace-v4
(ATTACK_AWARE_TRACK.md §7, Dr. Ho email 2026-07-12).

Ground-truth scalar binary flag (0.0 clean / 1.0 attacked, against exactly
ONE fixed attack condition per run) is injected into obs["observation"] by
training.single_attack_wrapper.SingleAttackWrapper. Separate track from the
existing sac_her_pickandplace_clean / _randomized / _randomized_p50_2M models
AND from the 3-category attack-aware track (results/models/
sac_her_pickandplace_attackaware_3cat) — this script never touches those paths.

Resumability: mirrors the --resume pattern already used and validated in
scripts/train_attack_aware_pickandplace.py (itself mirroring
training/train_sac_her.py) — SAC.load() + model.load_replay_buffer() +
reset_num_timesteps=False, with the replay buffer persisted alongside the
model via model.save_replay_buffer().

Usage:
    /opt/miniconda3/envs/reu_robotics/bin/python3 scripts/train_single_attack_pickandplace.py \\
        --attack-condition action_reversal --total-timesteps 500000

    # Later, to extend training past the first checkpoint:
    /opt/miniconda3/envs/reu_robotics/bin/python3 scripts/train_single_attack_pickandplace.py \\
        --attack-condition action_reversal --total-timesteps <N> --resume
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GYM_AVAILABLE,
    MODEL_PATH_PICKANDPLACE_SINGLEATTACK_PREFIX,
    SB3_AVAILABLE,
    SINGLE_ATTACK_CONDITIONS,
    SINGLE_ATTACK_P_CLEAN,
    SINGLE_ATTACK_TOTAL_TIMESTEPS,
    TB_DIR,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train a single-attack binary-flag SAC+HER policy on FetchPickAndPlace-v4."
    )
    parser.add_argument(
        "--attack-condition", type=str, required=True, choices=SINGLE_ATTACK_CONDITIONS,
        help="Fixed attack condition this wrapper instance mixes with clean episodes.",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=SINGLE_ATTACK_TOTAL_TIMESTEPS,
        help=(
            "With --resume, this is ADDITIVE, not an absolute target: SB3's "
            "model.learn(reset_num_timesteps=False) does "
            "`total_timesteps += self.num_timesteps` internally (confirmed by "
            "reading stable_baselines3.common.base_class.BaseAlgorithm."
            "_setup_learn source, SB3 2.8.0). E.g. resuming a 500,000-step "
            "checkpoint with --total-timesteps 200000 trains to 700,000 "
            "cumulative, not to 200,000."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-path", type=str, default=None,
        help=(
            "Override model save path (no .zip extension). Default: "
            f"{MODEL_PATH_PICKANDPLACE_SINGLEATTACK_PREFIX}_<attack-condition>_seed<seed>"
        ),
    )
    parser.add_argument(
        "--p-clean", type=float, default=SINGLE_ATTACK_P_CLEAN,
        help=f"Fraction of clean episodes (default {SINGLE_ATTACK_P_CLEAN}).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Load an existing model from --save-path and continue training. "
            "The step counter is NOT reset. A replay buffer checkpoint "
            "(<save_path>_replay_buffer.pkl) is loaded automatically if present."
        ),
    )
    args = parser.parse_args()

    if not GYM_AVAILABLE:
        raise RuntimeError("Gymnasium Robotics not available.")
    if not SB3_AVAILABLE:
        raise RuntimeError("Stable-Baselines3 not available.")

    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    from stable_baselines3.common.vec_env import DummyVecEnv

    from envs.fetchpickandplace_env import make_env
    from training.single_attack_wrapper import SingleAttackWrapper

    _condition, _seed, _p_clean = args.attack_condition, args.seed, args.p_clean
    save_path = args.save_path or (
        f"{MODEL_PATH_PICKANDPLACE_SINGLEATTACK_PREFIX}_{_condition}_seed{_seed}"
    )

    # DummyVecEnv via callable factory, same reasoning as train_sac_her.py:
    # HER needs the TimeLimit truncated=True signal to flow through correctly.
    env = DummyVecEnv([lambda: SingleAttackWrapper(
        make_env(seed=_seed), condition=_condition, p_clean=_p_clean, seed=_seed,
    )])
    print(f"[train] Single-attack training enabled (condition={_condition}, p_clean={_p_clean}).")

    os.makedirs(TB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tb_log_name = f"sac_her_pickandplace_singleattack_{_condition}"

    if args.resume:
        model_file = save_path + ".zip" if os.path.exists(save_path + ".zip") else save_path
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"--resume was set but no model found at: {save_path}(.zip)\n"
                "Train from scratch first, or check --save-path."
            )
        print(f"[resume] Loading model from: {model_file}")
        model = SAC.load(save_path, env=env, tensorboard_log=TB_DIR)

        replay_buf_path = save_path + "_replay_buffer.pkl"
        if os.path.exists(replay_buf_path):
            print(f"[resume] Loading replay buffer from: {replay_buf_path}")
            model.load_replay_buffer(replay_buf_path)
        else:
            print(
                "[resume] No replay buffer checkpoint found — buffer starts empty. "
                f"The first {model.learning_starts} steps will be pure exploration "
                "before gradient updates resume."
            )
        reset_num_timesteps = False
    else:
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            verbose=1,
            seed=args.seed,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.95,
            tau=0.05,
            # Must be > max_episode_steps (150) so at least one full episode
            # completes before HER tries to sample (matches train_sac_her.py).
            learning_starts=300,
            tensorboard_log=TB_DIR,
        )
        reset_num_timesteps = True

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name=tb_log_name,
        reset_num_timesteps=reset_num_timesteps,
    )
    env.close()

    model.save(save_path)
    model.save_replay_buffer(save_path + "_replay_buffer.pkl")
    print(f"Training complete. Model saved to: {save_path}")
    print(f"Replay buffer saved to: {save_path}_replay_buffer.pkl")


if __name__ == "__main__":
    main()
