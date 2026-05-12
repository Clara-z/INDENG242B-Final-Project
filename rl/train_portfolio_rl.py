#!/usr/bin/env python3
"""
Train a portfolio RL agent from a prebuilt feature parquet.

Expected input parquet:
  data/rl_features_with_news_pca32.parquet

The script trains PPO, evaluates on val/test windows, and writes:
  - model checkpoints
  - CSV/JSON metrics
  - training/evaluation plots (loss, reward, hit-rate, equity curve)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.distributions import Distribution
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    import torch as th
    from torch import nn
    import torch.nn.functional as th_f

    HAS_RL_DEPS = True
except ModuleNotFoundError:
    HAS_RL_DEPS = False
    gym = None
    spaces = None
    PPO = None
    BaseCallback = object
    DummyVecEnv = None
    VecMonitor = None
    Distribution = object
    ActorCriticPolicy = object
    th = None
    nn = None
    th_f = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO portfolio agent from parquet features.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/rl_features_with_news_pca32.parquet"),
        help="Path to merged feature parquet (price + fundamentals + PCA news).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/rl_runs"),
        help="Root folder where run artifacts are written.",
    )
    parser.add_argument("--train-start", type=str, default="2018-01-01")
    parser.add_argument("--train-end", type=str, default="2022-12-31")
    parser.add_argument("--val-start", type=str, default="2023-01-01")
    parser.add_argument("--val-end", type=str, default="2023-12-31")
    parser.add_argument("--test-start", type=str, default="2024-01-01")
    parser.add_argument("--test-end", type=str, default="2025-12-31")
    parser.add_argument(
        "--drawdown-penalty",
        type=float,
        default=0.3,
        help="Weight on |max_drawdown| in composite model-selection score: sharpe - penalty * |drawdown|.",
    )
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--episode-length", type=int, default=252)
    parser.add_argument("--turnover-cost", type=float, default=0.0005)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["simple", "icvar"],
        default="simple",
        help="simple: log-return - turnover; icvar: adds incremental CVaR penalty.",
    )
    parser.add_argument("--icvar-alpha", type=float, default=0.05)
    parser.add_argument("--icvar-lambda", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--policy",
        type=str,
        choices=["transformer", "mlp"],
        default="transformer",
        help="transformer: custom cross-asset attention + Dirichlet policy; mlp: legacy SB3 MlpPolicy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate data loading/splits and print summary; no RL training.",
    )
    return parser.parse_args()


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    if daily_returns.size < 2:
        return 0.0
    vol = float(np.std(daily_returns, ddof=1))
    if vol < 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(daily_returns) / vol)


def max_drawdown(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / np.maximum(running_max, 1e-12)) - 1.0
    return float(np.min(drawdowns))


def project_with_cap(weights: np.ndarray, max_weight: float) -> np.ndarray:
    w = np.maximum(weights.astype(np.float64), 0.0)
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w /= w.sum()

    # Iteratively clip and redistribute excess to uncapped names.
    for _ in range(10):
        over = w > max_weight + 1e-12
        if not np.any(over):
            break
        excess = float(np.sum(w[over] - max_weight))
        w[over] = max_weight
        under = ~over
        if not np.any(under):
            w[:] = 1.0 / len(w)
            break
        under_sum = float(np.sum(w[under]))
        if under_sum <= 1e-12:
            w[under] = (1.0 - np.sum(w[over])) / np.sum(under)
        else:
            w[under] += excess * (w[under] / under_sum)
        w = np.maximum(w, 0.0)
        w /= max(np.sum(w), 1e-12)
    return w.astype(np.float32)


def empirical_cvar(returns: np.ndarray, alpha: float) -> float:
    if returns.size < 2:
        return 0.0
    losses = -returns
    q = float(np.quantile(losses, 1.0 - alpha))
    tail = losses[losses >= q]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail))


@dataclass
class PanelData:
    tickers: list[str]
    dates: np.ndarray
    feature_cols: list[str]
    features: np.ndarray  # (T, N, F)
    next_returns: np.ndarray  # (T, N)


def load_panel_data(features_path: Path) -> PanelData:
    if not features_path.exists():
        raise FileNotFoundError(f"Features parquet not found: {features_path}")

    df = pd.read_parquet(features_path)
    required_cols = {"ticker", "date", "close_adj"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["next_ret"] = df.groupby("ticker", sort=False)["close_adj"].pct_change().shift(-1)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"close_adj", "next_ret"}]
    if not feature_cols:
        raise ValueError("No numeric feature columns were found for state input.")

    tickers = sorted(df["ticker"].dropna().unique().tolist())
    n_assets = len(tickers)

    feature_frames: list[np.ndarray] = []
    ret_frames: list[np.ndarray] = []
    date_list: list[np.datetime64] = []

    for date, g in df.groupby("date", sort=True):
        rows = g.set_index("ticker").reindex(tickers)
        if rows[feature_cols + ["next_ret"]].isna().any().any():
            continue
        feature_frames.append(rows[feature_cols].to_numpy(dtype=np.float32))
        ret_frames.append(rows["next_ret"].to_numpy(dtype=np.float32))
        date_list.append(np.datetime64(date.to_datetime64()))

    if len(date_list) < 260:
        raise ValueError(f"Insufficient aligned trading days after cleaning: {len(date_list)}")

    features = np.stack(feature_frames, axis=0)  # (T, N, F)
    next_returns = np.stack(ret_frames, axis=0)  # (T, N)

    if features.shape[1] != n_assets:
        raise ValueError("Asset dimension mismatch after panel construction.")

    return PanelData(
        tickers=tickers,
        dates=np.array(date_list),
        feature_cols=feature_cols,
        features=features,
        next_returns=next_returns,
    )


def select_window(panel: PanelData, start: str, end: str) -> PanelData:
    start_dt = np.datetime64(pd.Timestamp(start).normalize().to_datetime64())
    end_dt = np.datetime64(pd.Timestamp(end).normalize().to_datetime64())
    mask = (panel.dates >= start_dt) & (panel.dates <= end_dt)
    idx = np.where(mask)[0]
    if idx.size < 40:
        raise ValueError(f"Window {start}..{end} has too few trading days: {idx.size}")
    lo, hi = int(idx[0]), int(idx[-1])
    return PanelData(
        tickers=panel.tickers,
        dates=panel.dates[lo : hi + 1],
        feature_cols=panel.feature_cols,
        features=panel.features[lo : hi + 1],
        next_returns=panel.next_returns[lo : hi + 1],
    )


def standardize_by_train(train: PanelData, *others: PanelData) -> tuple[PanelData, ...]:
    mu = train.features.mean(axis=(0, 1), keepdims=True)
    sigma = train.features.std(axis=(0, 1), keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)

    def transform(p: PanelData) -> PanelData:
        x = (p.features - mu) / sigma
        x = np.clip(x, -5.0, 5.0)
        return PanelData(
            tickers=p.tickers,
            dates=p.dates,
            feature_cols=p.feature_cols,
            features=x.astype(np.float32),
            next_returns=p.next_returns.astype(np.float32),
        )

    return (transform(train),) + tuple(transform(p) for p in others)


if HAS_RL_DEPS:

    class DirichletPortfolioDistribution(Distribution):
        """Dirichlet distribution over long-only portfolio simplex weights."""

        def __init__(self, action_dim: int, eps: float = 1e-8) -> None:
            super().__init__()
            self.action_dim = int(action_dim)
            self.eps = float(eps)
            self.alpha: th.Tensor | None = None

        def proba_distribution_net(self, *args: Any, **kwargs: Any) -> nn.Module:
            return nn.Identity()

        def proba_distribution(self, alpha: th.Tensor) -> "DirichletPortfolioDistribution":
            self.alpha = alpha.clamp_min(1e-3)
            self.distribution = th.distributions.Dirichlet(self.alpha)
            return self

        def _normalize_actions(self, actions: th.Tensor) -> th.Tensor:
            actions = actions.reshape((-1, self.action_dim))
            actions = actions.clamp_min(self.eps)
            return actions / actions.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        def log_prob(self, actions: th.Tensor) -> th.Tensor:
            return self.distribution.log_prob(self._normalize_actions(actions))

        def entropy(self) -> th.Tensor:
            return self.distribution.entropy()

        def sample(self) -> th.Tensor:
            actions = self.distribution.rsample()
            return self._normalize_actions(actions)

        def mode(self) -> th.Tensor:
            alpha = self.alpha
            if alpha is None:
                raise RuntimeError("Distribution parameters have not been set.")
            return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        def actions_from_params(self, alpha: th.Tensor, deterministic: bool = False) -> th.Tensor:
            self.proba_distribution(alpha)
            return self.get_actions(deterministic=deterministic)

        def log_prob_from_params(self, alpha: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
            actions = self.actions_from_params(alpha)
            return actions, self.log_prob(actions)


    class PortfolioTransformerPolicy(ActorCriticPolicy):  # type: ignore[misc]
        """
        Transformer actor-critic for portfolio weights.

        The environment still exposes the current flat Box observation for SB3
        compatibility. This policy reshapes that vector back to
        (batch, assets, features + prev_weight), applies a shared per-asset
        encoder, cross-asset transformer attention, and a Dirichlet actor head.
        """

        MARKET_FEATURES = {
            "vix",
            "vix_change_20d",
            "tsy_10y_2y_spread",
            "spy_ret_20d",
            "spy_vol_20d",
        }

        def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Any,
            feature_cols: list[str] | None = None,
            global_dim: int = 3,
            hidden_dim: int = 128,
            asset_encoder_dim: int = 128,
            news_proj_dim: int = 64,
            transformer_ff_dim: int = 256,
            attention_heads: int = 4,
            transformer_layers: int = 1,
            dropout: float = 0.0,
            concentration_init: float | None = None,
            concentration_max: float = 20.0,
            **kwargs: Any,
        ) -> None:
            self.feature_cols = list(feature_cols) if feature_cols is not None else None
            self.global_dim = int(global_dim)
            self.hidden_dim = int(hidden_dim)
            self.asset_encoder_dim = int(asset_encoder_dim)
            self.news_proj_dim = int(news_proj_dim)
            self.transformer_ff_dim = int(transformer_ff_dim)
            self.attention_heads = int(attention_heads)
            self.transformer_layers = int(transformer_layers)
            self.dropout = float(dropout)
            self.concentration_init = None if concentration_init is None else float(concentration_init)
            self.concentration_max = float(concentration_max)
            kwargs.setdefault("ortho_init", False)
            kwargs.setdefault("net_arch", [])
            super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        def _build(self, lr_schedule: Any) -> None:
            if len(self.action_space.shape) != 1:
                raise ValueError("PortfolioTransformerPolicy requires a 1D Box action space.")
            if len(self.observation_space.shape) != 1:
                raise ValueError("PortfolioTransformerPolicy requires a flat 1D observation space.")

            self.n_assets = int(self.action_space.shape[0])
            obs_dim = int(self.observation_space.shape[0])
            per_asset_total, remainder = divmod(obs_dim - self.global_dim, self.n_assets)
            if remainder != 0 or per_asset_total < 2:
                raise ValueError(
                    f"Observation dim {obs_dim} is incompatible with {self.n_assets} assets "
                    f"and global_dim={self.global_dim}."
                )
            self.n_raw_features = per_asset_total - 1
            if self.feature_cols is None:
                self.feature_cols = [f"feature_{i}" for i in range(self.n_raw_features)]
            if len(self.feature_cols) != self.n_raw_features:
                raise ValueError(
                    f"feature_cols has {len(self.feature_cols)} columns, "
                    f"but observation implies {self.n_raw_features} raw features."
                )

            news_idx = [
                i
                for i, col in enumerate(self.feature_cols)
                if col.startswith(("news_pca_", "emb_", "sentiment_"))
                or col in {"n_articles", "mean_tone"}
            ]
            has_news_idx = [i for i, col in enumerate(self.feature_cols) if col == "has_news"]
            market_idx = [i for i, col in enumerate(self.feature_cols) if col in self.MARKET_FEATURES]
            excluded = set(news_idx) | set(has_news_idx) | set(market_idx)
            scalar_idx = [i for i in range(self.n_raw_features) if i not in excluded]

            self.register_buffer("scalar_idx", th.as_tensor(scalar_idx, dtype=th.long), persistent=False)
            self.register_buffer("news_idx", th.as_tensor(news_idx, dtype=th.long), persistent=False)
            self.register_buffer("has_news_idx", th.as_tensor(has_news_idx, dtype=th.long), persistent=False)
            self.register_buffer("market_idx", th.as_tensor(market_idx, dtype=th.long), persistent=False)

            effective_news_dim = self.news_proj_dim if news_idx else 0
            if news_idx:
                self.news_projection = nn.Sequential(
                    nn.Linear(len(news_idx), self.news_proj_dim),
                    nn.LayerNorm(self.news_proj_dim),
                )
            else:
                self.news_projection = None

            asset_input_dim = len(scalar_idx) + effective_news_dim + len(has_news_idx) + 1
            self.asset_encoder = nn.Sequential(
                nn.Linear(asset_input_dim, self.asset_encoder_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.asset_encoder_dim, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.attention_heads,
                dim_feedforward=self.transformer_ff_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cross_asset_attention = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)

            self.global_query = nn.Parameter(th.randn(1, 1, self.hidden_dim) * 0.02)
            self.global_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.attention_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            trunk_input_dim = self.hidden_dim + self.global_dim + len(market_idx)
            self.shared_trunk = nn.Sequential(
                nn.Linear(trunk_input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(self.hidden_dim),
            )
            self.actor_head = nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            final_actor = self.actor_head[-1]
            if isinstance(final_actor, nn.Linear):
                final_actor.weight.data.zero_()
                final_actor.bias.data.zero_()
            self.concentration_head = nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            self.value_net = nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            final_concentration = self.concentration_head[-1]
            if isinstance(final_concentration, nn.Linear):
                initial_concentration = self.concentration_init
                if initial_concentration is None:
                    initial_concentration = max(10.0, 2.0 * float(self.n_assets))
                self.concentration_max = max(self.concentration_max, initial_concentration)
                target = max(initial_concentration - 1.0, 1e-3)
                final_concentration.bias.data.fill_(float(np.log(np.expm1(target))))

            self.action_dist = DirichletPortfolioDistribution(self.n_assets)
            self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
                self.parameters(),
                lr=lr_schedule(1),
                **self.optimizer_kwargs,
            )

        def _split_flat_obs(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            obs = obs.float()
            asset_flat_dim = self.n_assets * (self.n_raw_features + 1)
            asset_block = obs[:, :asset_flat_dim].reshape(
                obs.shape[0],
                self.n_assets,
                self.n_raw_features + 1,
            )
            global_portfolio = obs[:, asset_flat_dim : asset_flat_dim + self.global_dim]
            return asset_block[:, :, : self.n_raw_features], asset_block[:, :, -1:], global_portfolio

        def _select_features(self, raw_features: th.Tensor, indices: th.Tensor) -> th.Tensor:
            if indices.numel() == 0:
                return raw_features.new_empty((*raw_features.shape[:2], 0))
            return raw_features.index_select(dim=-1, index=indices)

        def _encode(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
            raw_features, prev_w, global_portfolio = self._split_flat_obs(obs)
            pieces = [self._select_features(raw_features, self.scalar_idx)]
            if self.news_projection is not None:
                pieces.append(self.news_projection(self._select_features(raw_features, self.news_idx)))
            pieces.append(self._select_features(raw_features, self.has_news_idx))
            pieces.append(prev_w)

            asset_input = th.cat(pieces, dim=-1)
            encoded_assets = self.asset_encoder(asset_input)
            attended_assets = self.cross_asset_attention(encoded_assets)

            if self.market_idx.numel() > 0:
                market = raw_features[:, 0, :].index_select(dim=-1, index=self.market_idx)
                global_features = th.cat([global_portfolio, market], dim=-1)
            else:
                global_features = global_portfolio
            global_broadcast = global_features.unsqueeze(1).expand(-1, self.n_assets, -1)
            h = self.shared_trunk(th.cat([attended_assets, global_broadcast], dim=-1))

            query = self.global_query.expand(raw_features.shape[0], -1, -1)
            pooled, _ = self.global_attention(query, h, h, need_weights=False)
            return h, pooled.squeeze(1)

        def _distribution_and_value(self, obs: th.Tensor) -> tuple[DirichletPortfolioDistribution, th.Tensor]:
            h, pooled = self._encode(obs)
            logits = self.actor_head(h).squeeze(-1)
            mean = th_f.softmax(logits, dim=-1)
            log_concentration = self.concentration_head(pooled)
            concentration = (th_f.softplus(log_concentration) + 1.0).clamp(max=self.concentration_max)
            alpha = mean * concentration + 1e-3
            values = self.value_net(pooled)
            return self.action_dist.proba_distribution(alpha), values

        def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
            distribution, values = self._distribution_and_value(obs)
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return actions.reshape((-1, *self.action_space.shape)), values, log_prob

        def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
            distribution, values = self._distribution_and_value(obs)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return values, log_prob, entropy

        def get_distribution(self, obs: th.Tensor) -> DirichletPortfolioDistribution:
            distribution, _ = self._distribution_and_value(obs)
            return distribution

        def predict_values(self, obs: th.Tensor) -> th.Tensor:
            _, values = self._distribution_and_value(obs)
            return values

        def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
            return self.get_distribution(observation).get_actions(deterministic=deterministic)

        def _get_constructor_parameters(self) -> dict[str, Any]:
            data = super()._get_constructor_parameters()
            data.update(
                dict(
                    feature_cols=self.feature_cols,
                    global_dim=self.global_dim,
                    hidden_dim=self.hidden_dim,
                    asset_encoder_dim=self.asset_encoder_dim,
                    news_proj_dim=self.news_proj_dim,
                    transformer_ff_dim=self.transformer_ff_dim,
                    attention_heads=self.attention_heads,
                    transformer_layers=self.transformer_layers,
                    dropout=self.dropout,
                    concentration_init=self.concentration_init,
                    concentration_max=self.concentration_max,
                )
            )
            return data

    class PortfolioEnv(gym.Env):  # type: ignore[misc]
        metadata = {"render_modes": []}

        def __init__(
            self,
            panel: PanelData,
            episode_length: int,
            turnover_cost: float,
            max_weight: float,
            reward_mode: str,
            icvar_alpha: float,
            icvar_lambda: float,
            random_start: bool,
            seed: int,
        ) -> None:
            super().__init__()
            self.panel = panel
            self.episode_length = int(episode_length)
            self.turnover_cost = float(turnover_cost)
            self.max_weight = float(max_weight)
            self.reward_mode = reward_mode
            self.icvar_alpha = float(icvar_alpha)
            self.icvar_lambda = float(icvar_lambda)
            self.random_start = bool(random_start)
            self.rng = np.random.default_rng(seed)

            self.num_dates = panel.features.shape[0]
            self.n_assets = panel.features.shape[1]
            self.n_features = panel.features.shape[2]
            self.max_steps = self.num_dates - 1
            if self.max_steps < 2:
                raise ValueError("Panel window is too short for RL stepping.")

            obs_dim = self.n_assets * (self.n_features + 1) + 3
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

            self._start = 0
            self._end = 0
            self._idx = 0
            self._prev_w = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
            self._value = 1.0
            self._returns: list[float] = []
            self._equity: list[float] = [1.0]
            self._prev_cvar = 0.0

        def _episode_steps(self) -> int:
            if self.episode_length <= 0 or self.episode_length > self.max_steps:
                return self.max_steps
            return self.episode_length

        def _obs(self) -> np.ndarray:
            asset_slice = self.panel.features[self._idx]  # (N, F)
            combined = np.concatenate([asset_slice, self._prev_w[:, None]], axis=1).reshape(-1)
            lookback = np.array(self._returns[-20:], dtype=np.float32)
            rolling_vol = float(np.std(lookback, ddof=1) * np.sqrt(252.0)) if lookback.size >= 2 else 0.0
            steps_done = float(self._idx - self._start)
            global_state = np.array(
                [
                    rolling_vol,
                    steps_done,
                    float(self._value - 1.0),
                ],
                dtype=np.float32,
            )
            return np.concatenate([combined, global_state], axis=0).astype(np.float32)

        def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            steps = self._episode_steps()
            latest_start = self.max_steps - steps
            if self.random_start and latest_start > 0:
                self._start = int(self.rng.integers(0, latest_start + 1))
            else:
                self._start = 0
            self._end = self._start + steps - 1
            self._idx = self._start
            self._prev_w = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
            self._value = 1.0
            self._returns = []
            self._equity = [1.0]
            self._prev_cvar = 0.0
            return self._obs(), {}

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            w = project_with_cap(np.asarray(action, dtype=np.float32).reshape(-1), self.max_weight)
            turnover = float(np.sum(np.abs(w - self._prev_w)))
            gross = float(np.dot(w, self.panel.next_returns[self._idx]))
            net = gross - (self.turnover_cost * turnover)
            log_return = float(np.log1p(max(net, -0.999999)))

            self._value *= max(1.0 + net, 1e-8)
            self._returns.append(net)
            self._equity.append(self._value)
            self._prev_w = w

            icvar_penalty = 0.0
            if self.reward_mode == "icvar" and len(self._returns) > 20:
                cvar_t = empirical_cvar(np.array(self._returns, dtype=np.float64), self.icvar_alpha)
                icvar = cvar_t - self._prev_cvar
                self._prev_cvar = cvar_t
                icvar_penalty = self.icvar_lambda * icvar

            reward = log_return - icvar_penalty

            self._idx += 1
            terminated = self._idx > self._end
            truncated = False
            info: dict[str, Any] = {
                "date": str(pd.Timestamp(self.panel.dates[min(self._idx, self.num_dates - 1)]).date()),
                "step_return": net,
                "turnover": turnover,
                "reward_log_return": log_return,
                "reward_icvar_penalty": icvar_penalty,
            }
            if terminated:
                rets = np.array(self._returns, dtype=np.float64)
                eq = np.array(self._equity, dtype=np.float64)
                info.update(
                    {
                        "episode_return": float(self._value - 1.0),
                        "episode_sharpe": annualized_sharpe(rets),
                        "episode_hit_rate": float(np.mean(rets > 0.0)) if rets.size else 0.0,
                        "episode_max_drawdown": max_drawdown(eq),
                        "daily_returns": rets.tolist(),
                        "equity_curve": eq.tolist(),
                        "dates": [
                            str(pd.Timestamp(d).date())
                            for d in self.panel.dates[self._start : (self._end + 2)]
                        ],
                    }
                )
            return self._obs(), reward, terminated, truncated, info

else:

    class PortfolioEnv:  # pragma: no cover
        pass


class TrainLogCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        metrics: dict[str, float] = {"num_timesteps": float(self.num_timesteps)}
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, (float, int, np.floating)):
                metrics[key] = float(value)
        self.rows.append(metrics)


def evaluate_policy(model: Any, env: Any) -> dict[str, Any]:
    obs, _ = env.reset()
    done = False
    terminal_info: dict[str, Any] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        if done:
            terminal_info = info
    return terminal_info


def evaluate_equal_weight(env: Any) -> dict[str, Any]:
    obs, _ = env.reset()
    done = False
    n_assets = env.n_assets
    terminal_info: dict[str, Any] = {}
    while not done:
        action = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        if done:
            terminal_info = info
    return terminal_info


def build_test_daily_dataframe(test_info: dict[str, Any]) -> pd.DataFrame:
    """Create a length-safe daily test table from terminal episode info."""
    dates = list(test_info.get("dates", []))
    daily_returns = list(test_info.get("daily_returns", []))
    equity_curve = list(test_info.get("equity_curve", []))

    # Expected shape from env: len(dates) == len(equity_curve) == len(daily_returns) + 1.
    if len(dates) == len(equity_curve) and len(daily_returns) + 1 == len(dates):
        export_dates = dates[1:]
        export_equity = equity_curve[1:]
        return pd.DataFrame(
            {
                "date": export_dates,
                "rl_daily_return": daily_returns,
                "rl_equity": export_equity,
            }
        )

    # Defensive fallback if future env changes alter list lengths.
    min_len = min(len(dates), len(daily_returns), max(0, len(equity_curve) - 1))
    if min_len == 0:
        return pd.DataFrame(columns=["date", "rl_daily_return", "rl_equity"])
    return pd.DataFrame(
        {
            "date": dates[:min_len],
            "rl_daily_return": daily_returns[:min_len],
            "rl_equity": equity_curve[1 : min_len + 1],
        }
    )


def _linear_decay(initial_lr: float, min_lr: float = 1e-5):
    """Returns an SB3-compatible LR schedule: linearly decays from initial_lr to min_lr."""
    def schedule(progress_remaining: float) -> float:
        return min_lr + (initial_lr - min_lr) * progress_remaining
    return schedule


def build_ppo_model(
    policy_name: str,
    train_env: Any,
    feature_cols: list[str],
    learning_rate: float,
    seed: int,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
) -> Any:
    if not HAS_RL_DEPS:
        raise RuntimeError("RL dependencies are required to build a PPO model.")

    if policy_name == "transformer":
        policy: Any = PortfolioTransformerPolicy
        policy_kwargs: dict[str, Any] | None = {"feature_cols": feature_cols}
    elif policy_name == "mlp":
        policy = "MlpPolicy"
        policy_kwargs = None
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")

    kwargs: dict[str, Any] = {
        "learning_rate": _linear_decay(learning_rate),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.3,
        "target_kl": 0.1,
        "verbose": 1,
        "seed": seed,
        "policy_kwargs": {**(policy_kwargs or {}), "optimizer_kwargs": {"weight_decay": 1e-4}},
    }
    policy_kwargs = None  # already merged above
    if policy_kwargs is not None:
        kwargs["policy_kwargs"] = policy_kwargs
    return PPO(policy, train_env, **kwargs)


def write_run_config(
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    train: "PanelData",
    val: "PanelData",
    test: "PanelData",
) -> None:
    """Write a human-readable record of every hyperparameter and design choice for this run."""

    def fmt_date(dates: "np.ndarray") -> str:
        return f"{pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()}"

    # Score formula description
    score_formula = f"val_sharpe  −  {args.drawdown_penalty} × |val_max_drawdown|"

    # Selection metric name
    if args.drawdown_penalty == 0.0:
        selection_metric = "pure Sharpe ratio"
    else:
        selection_metric = "composite (Sharpe − drawdown penalty)"

    lines = [
        "=" * 60,
        f"RUN CONFIGURATION  —  {run_id}",
        "=" * 60,
        "",
        "[DATA]",
        f"  features_path    : {args.features_path}",
        f"  n_assets         : {len(train.tickers)}",
        f"  tickers          : {', '.join(train.tickers)}",
        f"  n_features       : {len(train.feature_cols)}",
        f"  train_days       : {len(train.dates):>4}   ({fmt_date(train.dates)})",
        f"  val_days         : {len(val.dates):>4}   ({fmt_date(val.dates)})",
        f"  test_days        : {len(test.dates):>4}   ({fmt_date(test.dates)})",
        "",
        "[ENVIRONMENT]",
        f"  episode_length   : {args.episode_length}  (0 = full window)",
        f"  turnover_cost    : {args.turnover_cost}",
        f"  max_weight       : {args.max_weight}",
        f"  reward_mode      : {args.reward_mode}",
        f"  icvar_alpha      : {args.icvar_alpha}  (only active when reward_mode=icvar)",
        f"  icvar_lambda     : {args.icvar_lambda}  (only active when reward_mode=icvar)",
        f"  seed             : {args.seed}",
        "",
        "[PPO HYPERPARAMETERS]",
        f"  total_timesteps  : {args.total_timesteps}",
        f"  eval_freq        : {args.eval_freq}",
        f"  learning_rate    : {args.learning_rate} → 1e-5  (linear decay with floor)",
        "  n_steps          : 2048",
        "  batch_size       : 256",
        "  n_epochs         : 10",
        "  gamma            : 0.99",
        "  gae_lambda       : 0.95",
        "  clip_range       : 0.2",
        "  ent_coef         : 0.001",
        "  vf_coef          : 0.5",
        "  max_grad_norm    : 0.3",
        "  target_kl        : 0.1",
        "  weight_decay     : 1e-4   (Adam optimizer L2 penalty)",
        "",
        f"[POLICY  —  {args.policy}]",
    ]

    if args.policy == "transformer":
        lines += [
            "  hidden_dim           : 128",
            "  asset_encoder_dim    : 128",
            "  news_proj_dim        : 64",
            "  transformer_ff_dim   : 256",
            "  attention_heads      : 4",
            "  transformer_layers   : 1",
            "  dropout              : 0.0   (disabled — causes train/eval KL mismatch in PPO)",
            "  concentration_max    : max(20.0, initial concentration)  (caps Dirichlet sharpness without making init sparse)",
            "  concentration_init   : auto  (max(10.0, 2 × n_assets) → alpha_i ≈ 2 at init)",
            "  distribution         : Dirichlet  (long-only simplex weights)",
            "  actor_init           : zeros  (starts at uniform allocation)",
        ]
    else:
        lines.append("  architecture         : SB3 MlpPolicy defaults")

    lines += [
        "",
        "[MODEL SELECTION]",
        f"  selection_metric : {selection_metric}",
        f"  score_formula    : {score_formula}",
        f"  drawdown_penalty : {args.drawdown_penalty}",
        "  interpretation   : penalises deep drawdowns when choosing best checkpoint;",
        "                     set --drawdown-penalty 0 to revert to pure Sharpe",
        "",
        "[NOTES]",
        "  (add run-specific observations below this line)",
        "",
    ]

    config_path = run_dir / "run_config.txt"
    config_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[config] written to {config_path}")


def write_plots(run_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, test_info: dict, base_info: dict) -> None:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Loss curves (if available from logger).
    loss_cols = [c for c in train_df.columns if "loss" in c.lower()]
    if loss_cols:
        plt.figure(figsize=(10, 5))
        for col in loss_cols:
            plt.plot(train_df["num_timesteps"], train_df[col], label=col)
        plt.title("Training Loss Curves")
        plt.xlabel("Timesteps")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "loss_curves.png", dpi=150)
        plt.close()

    if "rollout/ep_rew_mean" in train_df.columns:
        plt.figure(figsize=(9, 4))
        plt.plot(train_df["num_timesteps"], train_df["rollout/ep_rew_mean"])
        plt.title("Average Episode Reward During Training")
        plt.xlabel("Timesteps")
        plt.ylabel("Avg Reward")
        plt.tight_layout()
        plt.savefig(plot_dir / "avg_reward.png", dpi=150)
        plt.close()

    if not val_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(val_df["num_timesteps"], val_df["val_sharpe"], color="tab:blue", label="Val Sharpe")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Sharpe", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(
            val_df["num_timesteps"],
            val_df["val_hit_rate"],
            color="tab:orange",
            label="Val Hit Rate (Accuracy Proxy)",
        )
        ax2.set_ylabel("Hit Rate", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        plt.title("Validation Sharpe and Avg Accuracy Proxy")
        fig.tight_layout()
        fig.savefig(plot_dir / "val_sharpe_hit_rate.png", dpi=150)
        plt.close(fig)

    rl_dates = test_info.get("dates", [])
    rl_eq = test_info.get("equity_curve", [])
    base_eq = base_info.get("equity_curve", [])
    if rl_dates and rl_eq and base_eq:
        fig, ax = plt.subplots(figsize=(10, 5))
        rl_dates_dt = pd.to_datetime(rl_dates, errors="coerce")
        ax.plot(rl_dates_dt, rl_eq, label="RL Agent")
        ax.plot(rl_dates_dt[: len(base_eq)], base_eq, label="Equal Weight Baseline")

        # Auto-thin date ticks so labels stay readable for long test windows.
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        ax.set_title("Test Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "test_equity_curve.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    panel = load_panel_data(args.features_path)
    train = select_window(panel, args.train_start, args.train_end)
    val = select_window(panel, args.val_start, args.val_end)
    test = select_window(panel, args.test_start, args.test_end)
    train, val, test = standardize_by_train(train, val, test)

    print("[data] loaded:", args.features_path)
    print("[data] assets:", len(panel.tickers))
    print("[data] feature_count:", len(panel.feature_cols))
    print("[data] train days:", len(train.dates), "val days:", len(val.dates), "test days:", len(test.dates))

    if args.dry_run:
        print("[dry-run] Data contract and split windows look valid. Exiting without training.")
        return

    if not HAS_RL_DEPS:
        raise SystemExit(
            "Missing RL dependencies. Install with:\n"
            "  python -m pip install torch stable-baselines3 gymnasium"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    write_run_config(run_dir, run_id, args, train, val, test)

    train_env = VecMonitor(
        DummyVecEnv(
            [
                lambda: PortfolioEnv(
                    panel=train,
                    episode_length=args.episode_length,
                    turnover_cost=args.turnover_cost,
                    max_weight=args.max_weight,
                    reward_mode=args.reward_mode,
                    icvar_alpha=args.icvar_alpha,
                    icvar_lambda=args.icvar_lambda,
                    random_start=True,
                    seed=args.seed,
                )
            ]
        )
    )
    val_env = PortfolioEnv(
        panel=val,
        episode_length=0,
        turnover_cost=args.turnover_cost,
        max_weight=args.max_weight,
        reward_mode=args.reward_mode,
        icvar_alpha=args.icvar_alpha,
        icvar_lambda=args.icvar_lambda,
        random_start=False,
        seed=args.seed,
    )
    test_env = PortfolioEnv(
        panel=test,
        episode_length=0,
        turnover_cost=args.turnover_cost,
        max_weight=args.max_weight,
        reward_mode=args.reward_mode,
        icvar_alpha=args.icvar_alpha,
        icvar_lambda=args.icvar_lambda,
        random_start=False,
        seed=args.seed,
    )

    model = build_ppo_model(
        policy_name=args.policy,
        train_env=train_env,
        feature_cols=train.feature_cols,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    callback = TrainLogCallback()
    val_rows: list[dict[str, float]] = []
    best_score = float("-inf")
    best_path = model_dir / "best_model.zip"

    train_start_wall = datetime.now()
    trained = 0
    while trained < args.total_timesteps:
        chunk = min(args.eval_freq, args.total_timesteps - trained)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=callback, progress_bar=False)
        trained += chunk
        val_info = evaluate_policy(model, val_env)
        row = {
            "num_timesteps": float(trained),
            "val_return": float(val_info.get("episode_return", 0.0)),
            "val_sharpe": float(val_info.get("episode_sharpe", 0.0)),
            "val_hit_rate": float(val_info.get("episode_hit_rate", 0.0)),
            "val_max_drawdown": float(val_info.get("episode_max_drawdown", 0.0)),
        }
        row["val_score"] = row["val_sharpe"] - args.drawdown_penalty * abs(row["val_max_drawdown"])
        val_rows.append(row)
        print(
            f"[eval] t={trained} "
            f"return={row['val_return']:.4f} sharpe={row['val_sharpe']:.3f} "
            f"drawdown={row['val_max_drawdown']:.3f} score={row['val_score']:.3f}"
        )
        if row["val_score"] > best_score:
            best_score = row["val_score"]
            model.save(best_path)
            print(f"[checkpoint] new best val score={best_score:.4f} (sharpe={row['val_sharpe']:.4f}, dd={row['val_max_drawdown']:.4f})")

    train_end_wall = datetime.now()
    elapsed = train_end_wall - train_start_wall
    total_seconds = int(elapsed.total_seconds())
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
    print(f"[time] training finished in {elapsed_str}  ({total_seconds}s)")

    final_path = model_dir / "final_model.zip"
    model.save(final_path)

    best_model = PPO.load(best_path if best_path.exists() else final_path)
    test_info = evaluate_policy(best_model, test_env)
    base_info = evaluate_equal_weight(test_env)

    train_df = pd.DataFrame(callback.rows)
    val_df = pd.DataFrame(val_rows)
    train_df.to_csv(metrics_dir / "train_metrics.csv", index=False)
    val_df.to_csv(metrics_dir / "val_metrics.csv", index=False)

    test_daily = build_test_daily_dataframe(test_info)
    test_daily.to_csv(metrics_dir / "test_daily_returns.csv", index=False)

    summary = {
        "run_id": run_id,
        "features_path": str(args.features_path),
        "split": {
            "train": [args.train_start, args.train_end],
            "val": [args.val_start, args.val_end],
            "test": [args.test_start, args.test_end],
        },
        "reward_mode": args.reward_mode,
        "policy": args.policy,
        "icvar_alpha": float(args.icvar_alpha),
        "icvar_lambda": float(args.icvar_lambda),
        "train_assets": len(train.tickers),
        "feature_count": len(train.feature_cols),
        "best_val_score": float(best_score),
        "drawdown_penalty": float(args.drawdown_penalty),
        "training_time": elapsed_str,
        "training_seconds": total_seconds,
        "test_rl": {
            "return": float(test_info.get("episode_return", 0.0)),
            "sharpe": float(test_info.get("episode_sharpe", 0.0)),
            "hit_rate": float(test_info.get("episode_hit_rate", 0.0)),
            "max_drawdown": float(test_info.get("episode_max_drawdown", 0.0)),
        },
        "test_equal_weight": {
            "return": float(base_info.get("episode_return", 0.0)),
            "sharpe": float(base_info.get("episode_sharpe", 0.0)),
            "hit_rate": float(base_info.get("episode_hit_rate", 0.0)),
            "max_drawdown": float(base_info.get("episode_max_drawdown", 0.0)),
        },
        "artifacts": {
            "best_model": str(best_path if best_path.exists() else final_path),
            "final_model": str(final_path),
            "train_metrics": str(metrics_dir / "train_metrics.csv"),
            "val_metrics": str(metrics_dir / "val_metrics.csv"),
            "test_daily_returns": str(metrics_dir / "test_daily_returns.csv"),
        },
    }
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_plots(run_dir, train_df, val_df, test_info, base_info)

    config_path = run_dir / "run_config.txt"
    with config_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[TIMING]\n")
        f.write(f"  started          : {train_start_wall.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  finished         : {train_end_wall.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  total_duration   : {elapsed_str}  ({total_seconds}s)\n")

    print(f"[done] Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
