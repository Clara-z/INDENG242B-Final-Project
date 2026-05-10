#!/usr/bin/env python3
"""
Train a portfolio RL agent from a prebuilt feature parquet.

Expected input parquet:
  data/processed/rl_features_with_news_pca32.parquet

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    HAS_RL_DEPS = True
except ModuleNotFoundError:
    HAS_RL_DEPS = False
    gym = None
    spaces = None
    PPO = None
    BaseCallback = object
    DummyVecEnv = None
    VecMonitor = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO portfolio agent from parquet features.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/processed/rl_features_with_news_pca32.parquet"),
        help="Path to merged feature parquet (price + fundamentals + PCA news).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/rl_runs"),
        help="Root folder where run artifacts are written.",
    )
    parser.add_argument("--train-start", type=str, default="2020-01-01")
    parser.add_argument("--train-end", type=str, default="2023-06-30")
    parser.add_argument("--val-start", type=str, default="2023-07-01")
    parser.add_argument("--val-end", type=str, default="2023-12-31")
    parser.add_argument("--test-start", type=str, default="2024-01-01")
    parser.add_argument("--test-end", type=str, default="2026-01-31")
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
        plt.figure(figsize=(10, 5))
        plt.plot(rl_dates, rl_eq, label="RL Agent")
        plt.plot(rl_dates[: len(base_eq)], base_eq, label="Equal Weight Baseline")
        plt.xticks(rotation=45, ha="right")
        plt.title("Test Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "test_equity_curve.png", dpi=150)
        plt.close()


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

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
    )

    callback = TrainLogCallback()
    val_rows: list[dict[str, float]] = []
    best_sharpe = float("-inf")
    best_path = model_dir / "best_model.zip"

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
        val_rows.append(row)
        print(
            f"[eval] t={trained} "
            f"return={row['val_return']:.4f} sharpe={row['val_sharpe']:.3f} "
            f"hit_rate={row['val_hit_rate']:.3f}"
        )
        if row["val_sharpe"] > best_sharpe:
            best_sharpe = row["val_sharpe"]
            model.save(best_path)
            print(f"[checkpoint] new best val sharpe={best_sharpe:.4f}")

    final_path = model_dir / "final_model.zip"
    model.save(final_path)

    best_model = PPO.load(best_path if best_path.exists() else final_path)
    test_info = evaluate_policy(best_model, test_env)
    base_info = evaluate_equal_weight(test_env)

    train_df = pd.DataFrame(callback.rows)
    val_df = pd.DataFrame(val_rows)
    train_df.to_csv(metrics_dir / "train_metrics.csv", index=False)
    val_df.to_csv(metrics_dir / "val_metrics.csv", index=False)

    test_daily = pd.DataFrame(
        {
            "date": test_info.get("dates", []),
            "rl_daily_return": test_info.get("daily_returns", []),
            "rl_equity": test_info.get("equity_curve", [])[1:],
        }
    )
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
        "icvar_alpha": float(args.icvar_alpha),
        "icvar_lambda": float(args.icvar_lambda),
        "train_assets": len(train.tickers),
        "feature_count": len(train.feature_cols),
        "best_val_sharpe": float(best_sharpe),
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
    print(f"[done] Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
