# RL Training Plug-In Guide

This folder contains a runnable RL pipeline for portfolio training with your existing feature engineering outputs.

## 1) Required data files and folders

Put these files exactly here:

- `data/processed/rl_features_with_news_pca32.parquet` (required)
- `data/universe.json` (recommended for consistency checks)

The training script assumes the parquet already includes:

- Price/fundamental features (numeric)
- News sentiment features (`has_news`, `sentiment_*`, etc.)
- PCA-compressed news vectors (`news_pca_00` ... `news_pca_31`)
- Core identifiers: `ticker`, `date`, `close_adj`

If your parquet has a different name (for example PCA64), pass it with `--features-path`.

If you currently have separate files (`asset_features.parquet` and news parquet with `news_pca_00..31`), build the merged RL parquet with:

```bash
python rl/build_merged_rl_features.py \
  --asset-path data/processed/asset_features.parquet \
  --news-path data/processed/news_embeddings.parquet \
  --output-path data/processed/rl_features_with_news_pca32.parquet
```

## 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch stable-baselines3 gymnasium matplotlib pandas pyarrow
```

## 3) Validate data contract first (no training)

```bash
python rl/train_portfolio_rl.py --dry-run
```

This verifies:
- Required columns exist
- Ticker-date panel aligns
- Train/val/test windows have enough trading days

## 4) Train and evaluate

```bash
python rl/train_portfolio_rl.py \
  --features-path data/processed/rl_features_with_news_pca32.parquet \
  --total-timesteps 300000 \
  --eval-freq 25000 \
  --episode-length 252

# ICVaR reward variant (risk-aware):
python rl/train_portfolio_rl.py \
  --features-path data/processed/rl_features_with_news_pca32.parquet \
  --reward-mode icvar \
  --icvar-alpha 0.05 \
  --icvar-lambda 0.5
```

## 5) Output artifacts (auto-generated)

Each run is written to:

- `results/rl_runs/<timestamp>/models/best_model.zip`
- `results/rl_runs/<timestamp>/models/final_model.zip`
- `results/rl_runs/<timestamp>/metrics/train_metrics.csv`
- `results/rl_runs/<timestamp>/metrics/val_metrics.csv`
- `results/rl_runs/<timestamp>/metrics/test_daily_returns.csv`
- `results/rl_runs/<timestamp>/metrics/summary.json`
- `results/rl_runs/<timestamp>/plots/loss_curves.png`
- `results/rl_runs/<timestamp>/plots/avg_reward.png`
- `results/rl_runs/<timestamp>/plots/val_sharpe_hit_rate.png`
- `results/rl_runs/<timestamp>/plots/test_equity_curve.png`

`val_sharpe_hit_rate.png` includes average hit-rate, used as an accuracy proxy for trading decisions.
