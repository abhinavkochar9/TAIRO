"""
SAC+HER training and evaluation for FetchReach-v4 and FetchPickAndPlace-v4.

train_sac_her         — trains a SAC model with HerReplayBuffer and saves it.
evaluate_trained_model — runs saved model against all attack conditions and
                         saves episode + step CSVs.

Both functions guard against missing Gymnasium Robotics or SB3 and raise
RuntimeError with a clear message so the rest of the pipeline can degrade
gracefully when dependencies are unavailable.

--attack-randomization flag (Phase 4):
    When set, wraps the training env with AttackRandomizationWrapper so the
    policy is trained under a mix of clean and adversarial episodes.  Only
    affects training; evaluation always uses fixed attack magnitudes from
    ATTACK_LEVELS.  Intended for FetchPickAndPlace-v4 models (pass
    --env pickandplace --save-path results/models/sac_her_pickandplace_randomized).
"""

import os
from dataclasses import asdict
from typing import List, Optional, Tuple

import pandas as pd

from config import (
    ENV_ID,
    GYM_AVAILABLE,
    MAX_EPISODE_STEPS,
    MODEL_PATH,
    MODEL_PATH_PICKANDPLACE,
    MODEL_PATH_PICKANDPLACE_RANDOMIZED,
    RANDOM_SEEDS,
    RESULTS_DIR,
    SB3_AVAILABLE,
    TB_DIR,
)
from envs.fetchreach_env import make_env
from evaluation.episode_runner import run_episode

# Standard attack conditions used across all evaluation templates
EVAL_CONDITIONS = [
    ("clean", 0.00),
    ("sensor_noise", 0.01),
    ("sensor_noise", 0.05),
    ("action_noise", 0.05),
    ("action_scale", 0.50),
    ("action_delay", 0.00),
    ("target_shift", 0.03),
]


def train_sac_her(
    total_timesteps: int = 500_000,
    seed: int = 0,
    learning_rate: float = 3e-4,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    gamma: float = 0.95,
    tau: float = 0.05,
    n_sampled_goal: int = 4,
    save_path: Optional[str] = None,
):
    """Train SAC+HER on FetchReach-v4 and save the model.

    Args:
        total_timesteps: Number of environment steps to train for.
                         Use 10_000 for a quick smoke test; 500_000+ for
                         a converged policy.
        seed:            Random seed for the model and environment.
        learning_rate:   SB3 SAC learning rate.
        buffer_size:     Replay buffer capacity.
        batch_size:      Mini-batch size for gradient updates.
        gamma:           Discount factor.
        tau:             Soft-update coefficient for target networks.
        n_sampled_goal:  Number of HER goal re-labellings per real transition.
        save_path:       Where to save the model. Defaults to
                         ``RESULT_DIR/sac_her_fetchreach_model``.

    Returns:
        Trained SB3 SAC model.
    """
    if not GYM_AVAILABLE:
        raise RuntimeError(
            "Gymnasium Robotics is not available. "
            "Install it with: pip install gymnasium gymnasium-robotics mujoco"
        )
    if not SB3_AVAILABLE:
        raise RuntimeError(
            "Stable-Baselines3 is not available. "
            "Install it with: pip install stable-baselines3"
        )

    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    env = make_env(seed=seed)

    tb_log_dir = os.path.join(RESULTS_DIR, "tb_logs")

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy="future",
        ),
        verbose=1,
        seed=seed,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        learning_starts=300, # 2 x MAX_EP_LENGTH, Guarantees x complete episodes for HER
        tensorboard_log=tb_log_dir,
    )

    model.learn(total_timesteps=total_timesteps)
    env.close()

    if save_path is None:
        save_path = MODEL_PATH

    model.save(save_path)
    print(f"Model saved to: {save_path}")
    return model


def evaluate_trained_model(
    model,
    seeds: List[int] = RANDOM_SEEDS,
    conditions=None,
    output_prefix: str = "sac_her",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a trained SAC+HER model under all attack conditions.

    Runs both the plain model and a recovery-aware variant for each
    condition and seed. Saves episode-level and step-level CSVs.

    Args:
        model:         Trained SB3 model returned by train_sac_her or
                       loaded via ``SAC.load(...)``.
        seeds:         List of random seeds to evaluate across.
        conditions:    List of (condition_str, attack_level) tuples.
                       Defaults to EVAL_CONDITIONS.
        output_prefix: Prefix for saved CSV filenames.

    Returns:
        Tuple of (episode_df, step_df).
    """
    if not GYM_AVAILABLE:
        raise RuntimeError("Gymnasium Robotics is not available.")

    if conditions is None:
        conditions = EVAL_CONDITIONS

    all_results = []
    all_steps = []

    for seed in seeds:
        env = make_env(seed=seed)

        for condition, attack_level in conditions:
            for use_recovery, method_name in [
                (False, "sac_her"),
                (True, "recovery_aware_sac_her"),
            ]:
                result, step_df = run_episode(
                    env=env,
                    method=method_name,
                    seed=seed,
                    condition=condition,
                    attack_level=attack_level,
                    model=model,
                    use_recovery=use_recovery,
                )
                all_results.append(asdict(result))
                all_steps.append(step_df)

        env.close()

    episode_df = pd.DataFrame(all_results)
    step_df = pd.concat(all_steps, ignore_index=True)

    episode_path = os.path.join(RESULTS_DIR, f"{output_prefix}_attack_episode_results.csv")
    step_path = os.path.join(RESULTS_DIR, f"{output_prefix}_attack_step_logs.csv")

    episode_df.to_csv(episode_path, index=False)
    step_df.to_csv(step_path, index=False)

    print(f"Saved: {episode_path}")
    print(f"Saved: {step_path}")

    return episode_df, step_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SAC+HER on FetchReach-v4 or FetchPickAndPlace-v4."
    )
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None,
                        help="Override model save path (no .zip extension).")
    parser.add_argument(
        "--env", choices=["fetchreach", "pickandplace"], default="fetchreach",
        help="Which environment to train on (default: fetchreach).",
    )
    parser.add_argument(
        "--attack-randomization", action="store_true",
        help=(
            "Wrap the training env with AttackRandomizationWrapper so the policy "
            "trains under a mix of clean and adversarial episodes. "
            "Intended for FetchPickAndPlace-v4 (--env pickandplace)."
        ),
    )
    parser.add_argument(
        "--p-clean", type=float, default=0.2,
        help="Fraction of clean episodes when --attack-randomization is set (default 0.2).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Load an existing model from --save-path (or the default path) and "
            "continue training from where it left off. The step counter is NOT "
            "reset so TensorBoard x-axis remains cumulative. "
            "A replay buffer checkpoint (<save_path>_replay_buffer.pkl) is loaded "
            "automatically if it exists alongside the model."
        ),
    )
    args = parser.parse_args()

    # --- Environment factory ---------------------------------------------------
    if args.env == "pickandplace":
        from envs.fetchpickandplace_env import make_env as _make_env
        default_save = (
            MODEL_PATH_PICKANDPLACE_RANDOMIZED
            if args.attack_randomization
            else MODEL_PATH_PICKANDPLACE
        )
        tb_log_name = (
            "sac_her_pickandplace_randomized"
            if args.attack_randomization
            else "sac_her_pickandplace_clean"
        )
    else:
        from envs.fetchreach_env import make_env as _make_env
        default_save = MODEL_PATH
        tb_log_name = "sac_her_fetchreach"

    save_path = args.save_path or default_save

    # --- Build env (optionally wrapped) ----------------------------------------
    if not GYM_AVAILABLE:
        raise RuntimeError("Gymnasium Robotics not available.")
    if not SB3_AVAILABLE:
        raise RuntimeError("Stable-Baselines3 not available.")

    from stable_baselines3.common.vec_env import DummyVecEnv

    # Env must be wrapped in DummyVecEnv via a callable factory so that the
    # TimeLimit truncated=True signal at episode end flows through correctly.
    # HER requires completed episodes before it can sample; passing a bare env
    # object loses the episode-boundary signal and triggers
    # "Unable to sample before end of first episode".
    if args.attack_randomization:
        from training.attack_randomization_wrapper import AttackRandomizationWrapper
        _seed, _p_clean = args.seed, args.p_clean
        env = DummyVecEnv([lambda: AttackRandomizationWrapper(
            _make_env(seed=_seed), p_clean=_p_clean, seed=_seed,
        )])
        print(f"[train] Attack-domain randomization enabled (p_clean={args.p_clean}).")
    else:
        _seed = args.seed
        env = DummyVecEnv([lambda: _make_env(seed=_seed)])

    # --- Train -----------------------------------------------------------------
    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    os.makedirs(TB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.resume:
        # SAC.load accepts both "path" and "path.zip"; MODEL_PATH has no extension.
        model_file = save_path + ".zip" if os.path.exists(save_path + ".zip") else save_path
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"--resume was set but no model found at: {save_path}(.zip)\n"
                "Train from scratch first, or check --save-path."
            )
        print(f"[resume] Loading model from: {model_file}")
        model = SAC.load(
            save_path,          # SB3 appends .zip automatically
            env=env,
            tensorboard_log=TB_DIR,
        )
        # Restore replay buffer if a checkpoint was saved alongside the model.
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
        reset_num_timesteps = False     # keep cumulative step count for TensorBoard
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
            # completes before HER tries to sample.  SB3 default is 100 which
            # triggers "Unable to sample before end of first episode".
            learning_starts=300,
            tensorboard_log=TB_DIR,
        )
        reset_num_timesteps = True      # fresh run starts counter at 0

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name=tb_log_name,
        reset_num_timesteps=reset_num_timesteps,
    )
    env.close()

    model.save(save_path)
    # Save replay buffer alongside the model so a future --resume can reuse it.
    model.save_replay_buffer(save_path + "_replay_buffer.pkl")
    print(f"Training complete. Model saved to: {save_path}")
    print(f"Replay buffer saved to: {save_path}_replay_buffer.pkl")
