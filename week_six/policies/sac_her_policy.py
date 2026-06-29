"""
Wrapper around a trained Stable-Baselines3 SAC+HER model.

Exposes the same two-argument callable signature as the rule-based and
random policies so episode_runner can treat all policies uniformly:

    action = policy(env, obs)

Usage
    from stable_baselines3 import SAC
    from policies.sac_her_policy import SACHerPolicy

    model = SAC.load("tairo_results/sac_her_fetchreach_model")
    policy = SACHerPolicy(model)
    action = policy(env, obs)
"""
from typing import Dict
import numpy as np


class SACHerPolicy:
    """Callable wrapper for a Stable-Baselines3 SAC (or SAC+HER) model.

    Args:
        model:       Trained SB3 model with a ``predict`` method.
        deterministic: Use deterministic actions at inference (default True).
    """

    def __init__(self, model, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = deterministic

    def __call__(self, env, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict an action from the current observation.

        Args:
            env: Gymnasium environment (unused; kept for uniform signature).
            obs: Goal-conditioned observation dict.

        Returns:
            Action array predicted by the SB3 model.
        """
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action

    def predict(self, obs: Dict[str, np.ndarray], deterministic: bool = True):
        """Mirror SB3's model.predict() so record_agent can call policy.predict()."""
        return self.model.predict(obs, deterministic=deterministic)

    def __repr__(self) -> str:
        return f"SACHerPolicy(model={self.model.__class__.__name__}, deterministic={self.deterministic})"
