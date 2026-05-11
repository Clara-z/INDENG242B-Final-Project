from __future__ import annotations

import unittest

import numpy as np


try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    from rl.train_portfolio_rl import (
        HAS_RL_DEPS,
        PanelData,
        PortfolioEnv,
        PortfolioTransformerPolicy,
        build_ppo_model,
    )
except ModuleNotFoundError:
    HAS_RL_DEPS = False


@unittest.skipUnless(HAS_RL_DEPS, "RL dependencies are not installed")
class TrainingPolicyWiringTest(unittest.TestCase):
    def test_transformer_policy_is_used_when_requested(self) -> None:
        feature_cols = ["ret_1d", "vol_20d", "has_news", "news_pca_00", "vix"]
        panel = PanelData(
            tickers=[f"T{i}" for i in range(4)],
            dates=np.array([np.datetime64(f"2024-01-{day:02d}") for day in range(1, 12)]),
            feature_cols=feature_cols,
            features=np.random.default_rng(3).normal(size=(11, 4, len(feature_cols))).astype(np.float32),
            next_returns=np.random.default_rng(4).normal(scale=0.01, size=(11, 4)).astype(np.float32),
        )
        env = VecMonitor(
            DummyVecEnv(
                [
                    lambda: PortfolioEnv(
                        panel=panel,
                        episode_length=5,
                        turnover_cost=0.0005,
                        max_weight=0.40,
                        reward_mode="simple",
                        icvar_alpha=0.05,
                        icvar_lambda=0.5,
                        random_start=False,
                        seed=11,
                    )
                ]
            )
        )

        model = build_ppo_model(
            policy_name="transformer",
            train_env=env,
            feature_cols=feature_cols,
            learning_rate=3e-4,
            seed=11,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
        )

        self.assertIsInstance(model.policy, PortfolioTransformerPolicy)


if __name__ == "__main__":
    unittest.main()
