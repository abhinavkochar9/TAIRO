"""
Train the attack-aware SAC+HER policy on FetchPickAndPlace-v4
(ATTACK_AWARE_TRACK.md, Step 2).

Ground truth attack-category flag (clean/action/sensor/goal, one-hot,
ATTACK_AWARE_TRACK.md §4) is injected into obs["observation"] by
training.attack_aware_wrapper.AttackAwareWrapper. Separate track from the
existing sac_her_pickandplace_clean / _randomized / _randomized_p50_2M
models — this script never touches those paths.

Resumability (checkpoint, not a hard stop at 2,000,000 timesteps): mirrors
the --resume pattern already used and validated in training/train_sac_her.py
for the existing PickAndPlace models — SAC.load() + model.load_replay_buffer()
+ reset_num_timesteps=False, with the replay buffer persisted alongside the
model via model.save_replay_buffer(). See ATTACK_AWARE_TRACK.md for the
smoke-test confirmation that this mechanism works with this wrapper.

Usage:
    /opt/miniconda3/envs/reu_robotics/bin/python3 scripts/train_attack_aware_pickandplace.py \\
        --total-timesteps 2000000

    # Later, to extend training past the first checkpoint:
    /opt/miniconda3/envs/reu_robotics/bin/python3 scripts/train_attack_aware_pickandplace.py \\
        --total-timesteps <N> --resume
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ATTACK_AWARE_P_CLEAN,
    ATTACK_AWARE_TIMESTEPS_CHECKPOINT_1,
    GYM_AVAILABLE,
    MODEL_PATH_PICKANDPLACE_ATTACKAWARE,
    SB3_AVAILABLE,
    TB_DIR,
)

TB_LOG_NAME = "sac_her_pickandplace_attackaware_3cat"


def main():
    parser = argparse.ArgumentParser(
        description="Train the attack-aware SAC+HER policy on FetchPickAndPlace-v4."
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=ATTACK_AWARE_TIMESTEPS_CHECKPOINT_1,
        help=(
            "With --resume, this is ADDITIVE, not an absolute target: SB3's "
            "model.learn(reset_num_timesteps=False) does "
            "`total_timesteps += self.num_timesteps` internally (confirmed by "
            "reading stable_baselines3.common.base_class.BaseAlgorithm."
            "_setup_learn source, SB3 2.8.0). E.g. resuming a 2,000,000-step "
            "checkpoint with --total-timesteps 500000 trains to 2,500,000 "
            "cumulative, not to 500,000."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-path", type=str, default=MODEL_PATH_PICKANDPLACE_ATTACKAWARE,
        help="Override model save path (no .zip extension).",
    )
    parser.add_argument(
        "--p-clean", type=float, default=ATTACK_AWARE_P_CLEAN,
        help=f"Fraction of clean episodes (default {ATTACK_AWARE_P_CLEAN}, "
             "anchored per ATTACK_AWARE_TRACK.md §4d).",
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
    from training.attack_aware_wrapper import AttackAwareWrapper

    save_path = args.save_path
    _seed, _p_clean = args.seed, args.p_clean

    # DummyVecEnv via callable factory, same reasoning as train_sac_her.py:
    # HER needs the TimeLimit truncated=True signal to flow through correctly.
    env = DummyVecEnv([lambda: AttackAwareWrapper(
        make_env(seed=_seed), p_clean=_p_clean, seed=_seed,
    )])
    print(f"[train] Attack-aware training enabled (p_clean={_p_clean}).")

    os.makedirs(TB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
        tb_log_name=TB_LOG_NAME,
        reset_num_timesteps=reset_num_timesteps,
    )
    env.close()

    model.save(save_path)
    model.save_replay_buffer(save_path + "_replay_buffer.pkl")
    print(f"Training complete. Model saved to: {save_path}")
    print(f"Replay buffer saved to: {save_path}_replay_buffer.pkl")


if __name__ == "__main__":
    main()
