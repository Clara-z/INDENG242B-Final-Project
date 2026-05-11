from __future__ import annotations

import unittest

import numpy as np


try:
    import gymnasium as gym
    import torch as th

    from rl.train_portfolio_rl import HAS_RL_DEPS, PortfolioTransformerPolicy
except ModuleNotFoundError:
    HAS_RL_DEPS = False


@unittest.skipUnless(HAS_RL_DEPS, "RL dependencies are not installed")
class PortfolioTransformerPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        th.manual_seed(7)
        self.n_assets = 4
        self.feature_cols = [
            "ret_1d",
            "vol_20d",
            "has_news",
            "sentiment_net",
            "news_pca_00",
            "vix",
        ]
        obs_dim = self.n_assets * (len(self.feature_cols) + 1) + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

    def make_policy(self) -> PortfolioTransformerPolicy:
        policy = PortfolioTransformerPolicy(
            self.observation_space,
            self.action_space,
            lr_schedule=lambda _: 3e-4,
            feature_cols=self.feature_cols,
            hidden_dim=32,
            asset_encoder_dim=32,
            news_proj_dim=8,
            transformer_ff_dim=64,
            attention_heads=4,
            transformer_layers=1,
            dropout=0.0,
        )
        policy.eval()
        return policy

    def test_policy_outputs_valid_simplex_actions_and_log_probs(self) -> None:
        policy = self.make_policy()
        obs = th.randn(5, self.observation_space.shape[0])

        actions, values, log_prob = policy(obs, deterministic=False)

        self.assertEqual(tuple(actions.shape), (5, self.n_assets))
        self.assertEqual(tuple(values.shape), (5, 1))
        self.assertEqual(tuple(log_prob.shape), (5,))
        self.assertTrue(th.all(actions >= 0.0).item())
        self.assertTrue(th.allclose(actions.sum(dim=-1), th.ones(5), atol=1e-5))
        self.assertTrue(th.isfinite(log_prob).all().item())

    def test_policy_accepts_single_unbatched_observation(self) -> None:
        policy = self.make_policy()
        obs = th.randn(self.observation_space.shape[0])

        action, value, log_prob = policy(obs, deterministic=True)

        self.assertEqual(tuple(action.shape), (1, self.n_assets))
        self.assertEqual(tuple(value.shape), (1, 1))
        self.assertEqual(tuple(log_prob.shape), (1,))
        self.assertTrue(th.allclose(action.sum(dim=-1), th.ones(1), atol=1e-5))

    def test_default_initial_dirichlet_is_not_sparse_for_thirty_assets(self) -> None:
        n_assets = 30
        feature_cols = ["ret_1d", "has_news", "news_pca_00", "vix"]
        obs_dim = n_assets * (len(feature_cols) + 1) + 3
        policy = PortfolioTransformerPolicy(
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            gym.spaces.Box(low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32),
            lr_schedule=lambda _: 3e-4,
            feature_cols=feature_cols,
            hidden_dim=32,
            asset_encoder_dim=32,
            news_proj_dim=8,
            transformer_ff_dim=64,
            attention_heads=4,
            transformer_layers=1,
            dropout=0.0,
        )
        policy.eval()

        dist = policy.get_distribution(th.zeros(2, obs_dim))

        alpha = dist.alpha.detach()
        self.assertGreaterEqual(float(alpha.min()), 1.0)
        self.assertGreaterEqual(float(alpha.sum(dim=-1).mean()), 2.0 * n_assets)

    def test_policy_is_asset_permutation_equivariant_and_value_invariant(self) -> None:
        policy = self.make_policy()
        obs = th.randn(3, self.observation_space.shape[0])
        per_asset = obs[:, : self.n_assets * (len(self.feature_cols) + 1)].view(
            3,
            self.n_assets,
            len(self.feature_cols) + 1,
        )
        per_asset[:, :, self.feature_cols.index("vix")] = per_asset[:, :1, self.feature_cols.index("vix")]
        obs[:, : self.n_assets * (len(self.feature_cols) + 1)] = per_asset.reshape(3, -1)

        perm = th.tensor([2, 0, 3, 1])
        permuted = obs.clone()
        permuted_assets = per_asset[:, perm, :]
        permuted[:, : self.n_assets * (len(self.feature_cols) + 1)] = permuted_assets.reshape(3, -1)

        actions, values, _ = policy(obs, deterministic=True)
        perm_actions, perm_values, _ = policy(permuted, deterministic=True)

        self.assertTrue(th.allclose(actions[:, perm], perm_actions, atol=1e-5))
        self.assertTrue(th.allclose(values, perm_values, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
