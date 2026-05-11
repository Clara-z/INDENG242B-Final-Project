# Architecture Specification — Portfolio RL Agent

**Document purpose:** Authoritative reference for the actual implemented system. All details verified against current codebase.

**Project scope:** Daily-rebalancing portfolio allocation over 30 fixed US equities, trained with PPO using Stable-Baselines3, evaluated on post-cutoff test window.

**Last updated:** 2026-05-11

---

## 1. Universe and Time Range

### 1.1 Ticker Universe

**30 US equities** defined in `tickers.txt` and `data/universe.json`. The list includes a mix of large-cap tech (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NFLX), mid-cap, and small-cap stocks across multiple sectors.

**Current tickers:**
```
AAPL, AAL, ABEO, ACAD, ACGL, ACHC, ACHV, ACLS, ACMR, ACRS,
ADBE, ADI, ADMA, ADP, ADSK, AEHR, AEIS, AEP, CCO, CCRN,
CCS, TSLA, NFLX, MSFT, META, GOOGL, AMZN, AGNC, AKBA, AMC
```

**Universe file:** `data/universe.json`
- `tickers`: array of 30 ticker symbols
- `companies`: array of objects with `ticker`, `company_name`, `entity_match_patterns`

### 1.2 Date Range

| Split | Start | End | Trading Days (aligned panel) |
|-------|-------|-----|------------------------------|
| **Full data** | 2018-01-02 | 2026-01-30 | ~2,031 |
| **Train** | 2018-01-01 | 2023-06-30 | ~1,131 |
| **Validation** | 2023-07-01 | 2023-12-31 | ~126 |
| **Test** | 2024-01-01 | 2026-01-31 | ~521 |

**Default in code:** `rl/train_portfolio_rl.py` uses `--train-start 2018-01-01` (first aligned dates may be slightly after 2018-01-02 where all assets have complete numeric rows, e.g. long lookbacks like `ret_252d`).

---

## 2. Data Pipeline

### 2.1 Source Data

#### 2.1.1 Price and Fundamental Data

**Source CSV:** `data/fundamentals & technical features.csv`
- 60,932 rows (30 tickers × ~2,031 trading days)
- Contains: `date`, `ticker`, `close_adj`, `volume`, fundamental ratios, market indicators, sector dummies

**Build script:** `rl/build_asset_and_prices_from_csv.py`

```bash
python rl/build_asset_and_prices_from_csv.py \
  --source-csv "data/fundamentals & technical features.csv" \
  --asset-out data/asset_features.parquet \
  --prices-out data/prices.parquet \
  --report-out data/asset_feature_build_report.json
```

#### 2.1.2 News Data Acquisition

**GDELT Scraping:** `month_ticker_scrape.py`
- Queries GDELT DOC API for company news per ticker per month
- Uses `config_companies.py` for query terms and entity matching patterns
- Downloads article full text via concurrent HTTP fetching
- Outputs: metadata CSV + full-text parquet per year

```bash
python month_ticker_scrape.py \
  --year 2024 \
  --tickers_file tickers.txt \
  --window_days 30 \
  --max_articles_per_month 30 \
  --output_dir data
```

**Alternative scraper:** `scrape_guardian_news.py`
- Simpler GDELT scraper for yearly JSON exports

#### 2.1.3 FinBERT Embedding Pipeline

**Notebook:** `colab_news_finbert_pipeline.ipynb` (run on Google Colab with GPU)

Processing steps:
1. Load news articles from scraped data
2. Run FinBERT sentiment classification → `sentiment_pos`, `sentiment_neu`, `sentiment_neg`
3. Extract FinBERT [CLS] embeddings (768-dim)
4. Apply PCA to compress 768-dim → 32-dim (`news_pca_00` ... `news_pca_31`)
5. Aggregate to daily (ticker, date) level
6. Output: `data/news/news_finbert_YYYY_daily_pca32.parquet` per year

---

### 2.2 Derived Parquet Files

#### 2.2.1 `data/asset_features.parquet`

**Schema (38 columns):**

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Ticker symbol |
| `date` | datetime | Trading date |
| `close_adj` | float32 | Split/dividend-adjusted close price |
| `volume` | float32 | Daily traded volume |
| `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`, `ret_252d` | float32 | Percentage returns over lookback periods |
| `vol_20d`, `vol_60d` | float32 | Realized volatility (rolling std of daily returns) |
| `volume_zscore_20d` | float32 | Volume z-score over 20-day rolling window |
| `price_to_ma20`, `price_to_ma50` | float32 | Price relative to moving averages |
| `high_low_range_5d` | float32 | 5-day high-low range / close (0.0 if no high/low data) |
| `rsi_14` | float32 | 14-period RSI |
| `macd_signal` | float32 | MACD signal line |
| `pe`, `pb`, `roe`, `de_ratio`, `profit_margin`, `asset_turnover`, `current_ratio` | float32 | Fundamental ratios |
| `days_since_filing` | float32 | Days since last SEC filing |
| `vix`, `vix_change_20d` | float32 | VIX level and 20-day change |
| `spy_ret_20d`, `spy_vol_20d` | float32 | SPY 20-day return and volatility |
| `tsy_10y_2y_spread` | float32 | Treasury yield spread |
| `sector_*` | float32 | 8 sector one-hot columns |

**Excluded:** `gross_margin` (too many missing values)

#### 2.2.2 `data/prices.parquet`

**Schema (4 columns):**

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Ticker symbol |
| `date` | datetime | Trading date |
| `close_adj` | float32 | Adjusted close price |
| `is_trading_day` | int8 | Always 1 for trading days |

#### 2.2.3 `data/news/news_finbert_YYYY_daily_pca32.parquet`

Per-year files with daily aggregated news features.

**Schema (40 columns):**

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Ticker symbol |
| `date` | datetime | Trading date |
| `sentiment_pos`, `sentiment_neu`, `sentiment_neg` | float32 | FinBERT sentiment probabilities (mean across articles) |
| `n_articles` | int32 | Number of articles that day |
| `has_news` | int8 | 1 if any articles, else 0 |
| `mean_tone` | float32 | Mean tone score |
| `news_pca_00` ... `news_pca_31` | float32 | PCA-compressed FinBERT embeddings (32 dims) |

#### 2.2.4 `data/rl_features_with_news_pca32.parquet`

**Merged RL-ready feature table (76 columns):**

Built by merging `asset_features.parquet` with news PCA features.

**Merge command** (news input must be a single parquet with `ticker`, `date`, and `news_pca_*` columns—you may need to concatenate yearly `data/news/news_finbert_YYYY_daily_pca32.parquet` files first):

```bash
python rl/build_merged_rl_features.py \
  --asset-path data/asset_features.parquet \
  --news-path data/news/news_finbert_2024_daily_pca32.parquet \
  --output-path data/rl_features_with_news_pca32.parquet
```

**Key columns:**
- All 38 columns from `asset_features.parquet`
- `has_news`, `n_articles`, `sentiment_pos`, `sentiment_neu`, `sentiment_neg`, `sentiment_net`
- `news_pca_00` ... `news_pca_31` (32 PCA dimensions)

**Total feature count for RL:** 73 numeric features (excluding `ticker`, `date`, `close_adj`)

---

## 3. RL Environment Specification

### 3.1 Implementation

**File:** `rl/train_portfolio_rl.py`
**Class:** `PortfolioEnv` (Gymnasium-compatible)

### 3.2 Episode Structure

| Parameter | Value |
|-----------|-------|
| Episode length | 252 trading days (1 year) |
| Episode start | Randomly sampled (training) or fixed at window start (eval) |
| Step frequency | Daily, close-to-close |
| Termination | End of episode length |

### 3.3 Observation Space

```python
obs_dim = n_assets * (n_features + 1) + 3
# With current data: 30 * (73 + 1) + 3 = 2,223

observation_space = spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(obs_dim,), dtype=np.float32
)
```

**Observation composition:**

1. **Per asset, 73 numbers from the parquet** (all numeric columns except `close_adj`; `close_adj` is excluded because the environment uses it only to form `next_ret` for the transition dynamics). These 73 are, in column order:

   - **Price / volume / technical (15):** `volume`, `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`, `ret_252d`, `vol_20d`, `vol_60d`, `volume_zscore_20d`, `price_to_ma20`, `price_to_ma50`, `high_low_range_5d`, `rsi_14`, `macd_signal`
   - **Fundamentals + filing (8):** `pe`, `pb`, `roe`, `de_ratio`, `profit_margin`, `asset_turnover`, `current_ratio`, `days_since_filing`
   - **Market-wide (broadcast per row, 5):** `vix`, `spy_ret_20d`, `spy_vol_20d`, `vix_change_20d`, `tsy_10y_2y_spread`
   - **Sector dummies (8):** `sector_Communication Services`, `sector_Consumer Cyclical`, `sector_Financial Services`, `sector_Healthcare`, `sector_Industrials`, `sector_Real Estate`, `sector_Technology`, `sector_Utilities`
   - **News / FinBERT side (37):** `has_news`, `n_articles`, `sentiment_pos`, `sentiment_neu`, `sentiment_neg`, `sentiment_net`, `news_pca_00` … `news_pca_31` (32 PCA dimensions summarizing FinBERT embeddings)

2. **Plus 1 per asset:** **previous-day portfolio weight** for that asset (from `load_panel_data` this is **not** stored in the parquet; the env appends `prev_w` in `PortfolioEnv._obs()`).

3. **So “74 per asset” = 73 (table features, including PCA + sentiment) + 1 (`prev_w`).**

4. **Trailing “+3” is not FinBERT.** It is a **global portfolio tail** appended once per observation (same for all assets in the flattened vector):
   - **Rolling portfolio volatility** (√252 × stdev of last up to 20 daily *portfolio* returns),
   - **Steps since episode start** (within the current episode),
   - **Cumulative episode return** (portfolio value minus 1).

**Full vector:** `30 × 74 + 3 = 2,223` floats. FinBERT information is **inside** the 73 columns via `news_pca_*` and sentiment fields, not in the +3.

### 3.4 Action Space

```python
action_space = spaces.Box(low=0.0, high=1.0, shape=(30,), dtype=np.float32)
```

**Action processing:**
1. Normalize to sum to 1.0
2. Apply max-weight cap (default 0.25 per asset)
3. Iteratively redistribute excess until all weights ≤ max_weight

### 3.5 Reward Function

**Two variants supported via `--reward-mode` flag:**

**Variant A: Simple (default)**
```
reward = log(1 + portfolio_return) - κ * turnover
```
- `κ = 0.0005` (5 basis points per unit turnover)

**Variant B: ICVaR-augmented**
```
reward = log(1 + portfolio_return) - κ * turnover - λ * ICVaR
```
- `λ = 0.5` (configurable via `--icvar-lambda`)
- `α = 0.05` (configurable via `--icvar-alpha`)
- ICVaR = CVaR_t - CVaR_{t-1} (incremental conditional value-at-risk)
- ICVaR contribution set to 0 for first 20 steps (insufficient history)

### 3.6 Data Standardization

- Computed per-feature mean/std on **training split only**
- Applied as: `(x - mean) / std`, then clipped to [-5, 5]
- Applied identically to train/val/test at environment load time

---

## 4. PPO Training Configuration

### 4.1 Algorithm

**Library:** Stable-Baselines3 PPO  
**Default policy:** `PortfolioTransformerPolicy` (custom cross-asset Transformer + Dirichlet simplex distribution)
**Fallback policy:** `--policy mlp` keeps the previous SB3 `MlpPolicy` path for legacy comparisons.

### 4.1.1 Implemented custom policy

The repository now implements the earlier custom Actor-Critic idea in `rl/train_portfolio_rl.py` while preserving the current flat `Box` environment API. The policy reconstructs the flat vector into `(batch, 30, feature_count + previous_weight)`, splits scalar/news/market/previous-weight fields from `feature_cols`, applies a shared per-asset encoder, runs cross-asset Transformer attention, and emits a Dirichlet distribution over long-only portfolio weights.

The implementation adapts the old architecture to the actual data contract:
- Observation remains a flat `Box` of size 2,223 for compatibility with the existing `PortfolioEnv`, `VecMonitor`, and run outputs.
- News input uses the current PCA/sentiment columns (`news_pca_*`, `sentiment_*`, `n_articles`, `mean_tone`) rather than raw 768-dimensional FinBERT embeddings.
- Market-wide columns are extracted from the first asset row and concatenated with the 3 portfolio globals.
- The Dirichlet actor uses `mean = softmax(asset_logits)` and `alpha = mean * concentration + 1e-3`; deterministic evaluation uses the Dirichlet mean. Initial concentration scales with the asset count (`max(10, 2 * n_assets)`) so the 30-name policy starts dense rather than boundary-heavy.

| Aspect | Older / aspirational spec | Current repo |
|--------|---------------------------|--------------|
| Observation | Dict: `(30, 805)` per asset + globals + market | Single `Box` of size 2,223, reshaped inside the policy |
| News in state | 768-dim FinBERT + `has_news` | 32 PCA + sentiment + `has_news` (inside the 73) |
| Policy | Custom Transformer + Dirichlet | `PortfolioTransformerPolicy` by default; `--policy mlp` for legacy runs |
| Equivariance | Required actor equivariance and critic invariance | Covered by `tests/test_portfolio_transformer_policy.py` |

### 4.2 Hyperparameters

```python
PPO(
    PortfolioTransformerPolicy,
    env,
    policy_kwargs={"feature_cols": train.feature_cols},
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.02,
    seed=42,
)
```

### 4.3 Training Budget

| Setting | Recommended | Quick Test |
|---------|-------------|------------|
| Total timesteps | 5,000,000 | 300,000 |
| Eval frequency | 100,000 | 25,000 |
| Episode length | 252 | 252 |

**Estimated training time for 5M steps:**
- Mac M4 Pro: ~3h 15m
- Colab T4: ~2h 30m
- Colab A100: ~1h

### 4.4 Checkpointing

- Save best model when validation Sharpe improves
- Save final model at training completion
- Outputs to: `results/rl_runs/<timestamp>/models/`

---

## 5. Training and Evaluation

### 5.1 Run Commands

**Validate data contract (dry run):**
```bash
python rl/train_portfolio_rl.py --dry-run
```

**Quick experiment (300K steps):**
```bash
python rl/train_portfolio_rl.py \
  --features-path data/rl_features_with_news_pca32.parquet \
  --total-timesteps 300000 \
  --eval-freq 25000 \
  --episode-length 252
```

**Full training (5M steps):**
```bash
python rl/train_portfolio_rl.py \
  --features-path data/rl_features_with_news_pca32.parquet \
  --total-timesteps 5000000 \
  --eval-freq 100000 \
  --episode-length 252
```

**ICVaR reward variant:**
```bash
python rl/train_portfolio_rl.py \
  --features-path data/rl_features_with_news_pca32.parquet \
  --reward-mode icvar \
  --icvar-alpha 0.05 \
  --icvar-lambda 0.5
```

### 5.2 Output Artifacts

Each run creates `results/rl_runs/<timestamp>/`:

```
<timestamp>/
├── models/
│   ├── best_model.zip      # Best validation Sharpe checkpoint
│   └── final_model.zip     # Final training checkpoint
├── metrics/
│   ├── train_metrics.csv   # Per-rollout training stats
│   ├── val_metrics.csv     # Validation metrics per eval
│   ├── test_daily_returns.csv  # Daily test performance
│   └── summary.json        # Run configuration and final metrics
└── plots/
    ├── loss_curves.png
    ├── avg_reward.png
    ├── val_sharpe_hit_rate.png
    └── test_equity_curve.png
```

### 5.3 Metrics

| Metric | Description |
|--------|-------------|
| `episode_return` | Total portfolio return over episode |
| `episode_sharpe` | Annualized Sharpe ratio: √252 × mean(daily_ret) / std(daily_ret) |
| `episode_hit_rate` | Fraction of days with positive return |
| `episode_max_drawdown` | Maximum peak-to-trough decline |

### 5.4 Baselines

The test evaluation automatically compares against:
- **Equal-weight baseline:** Static 1/30 allocation, rebalanced daily

---

## 6. Repository Structure (Actual)

```
INDENG242B-Final-Project/
├── architecture_spec_final.md  # This document
├── tickers.txt                 # 30 ticker symbols
├── config_companies.py         # Company query config for GDELT
├── month_ticker_scrape.py      # News scraping pipeline
├── scrape_guardian_news.py     # Alternative news scraper
├── colab_news_finbert_pipeline.ipynb  # FinBERT + PCA (run on Colab)
│
├── rl/
│   ├── train_portfolio_rl.py           # Main training script
│   ├── build_asset_and_prices_from_csv.py  # CSV → parquet converter
│   └── build_merged_rl_features.py     # Merge asset + news features
│
├── data/
│   ├── universe.json                   # Ticker universe definition
│   ├── tickers.txt                     # Ticker list
│   ├── fundamentals & technical features.csv  # Source price/fundamental data
│   ├── asset_features.parquet          # Built asset features
│   ├── prices.parquet                  # Price-only table
│   ├── rl_features_with_news_pca32.parquet  # Merged RL features
│   ├── rl_features_with_news_pca64.parquet  # Alternative 64-dim PCA
│   ├── asset_feature_build_report.json # Build metadata
│   └── news/
│       └── news_finbert_YYYY_daily_pca32.parquet  # Per-year news features
│
├── results/
│   └── rl_runs/
│       └── <timestamp>/
│           ├── models/
│           ├── metrics/
│           └── plots/
│
└── venv/  # Python virtual environment
```

---

## 7. Dependencies

### 7.1 Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch stable-baselines3 gymnasium matplotlib pandas pyarrow numpy scikit-learn requests tqdm
```

### 7.2 For FinBERT Pipeline (Colab)

```python
!pip install transformers torch pandas pyarrow scikit-learn
```

---

## 8. Data Pipeline Workflow

### Step 1: Prepare Universe
```bash
# Ensure tickers.txt and data/universe.json are in sync
```

### Step 2: Build Asset Features
```bash
python rl/build_asset_and_prices_from_csv.py \
  --source-csv "data/fundamentals & technical features.csv"
```

### Step 3: Scrape News (per year)
```bash
python month_ticker_scrape.py --year 2024 --tickers_file tickers.txt --output_dir data
```

### Step 4: Run FinBERT Pipeline
```
Upload to Colab, run colab_news_finbert_pipeline.ipynb
Download output: news_finbert_YYYY_daily_pca32.parquet
```

### Step 5: Merge Features
```bash
python rl/build_merged_rl_features.py \
  --asset-path data/asset_features.parquet \
  --news-path <path-to-news-parquet> \
  --output-path data/rl_features_with_news_pca32.parquet
```

### Step 6: Validate and Train
```bash
python rl/train_portfolio_rl.py --dry-run  # Validate
python rl/train_portfolio_rl.py --total-timesteps 5000000  # Train
```

---

## 9. Known Limitations

1. **Survivorship bias:** Universe is fixed; no handling of delistings or index changes
2. **No high/low prices:** `high_low_range_5d` is set to 0.0 (source CSV lacks high/low)
3. **Model comparison still needed:** Custom Transformer + Dirichlet is now implemented, but final results should compare it against the legacy `--policy mlp` path on the same splits and seeds.
4. **PCA news features:** 32-dim PCA compression may lose information vs full 768-dim FinBERT
5. **Single-seed results:** Production should run 3+ seeds and report mean ± std

---

## 10. Example Results

From run `20260510_234839` (300K steps; trained before the default train window was extended to start in 2018—re-run to compare on the same split):

| Metric | RL Agent | Equal-Weight |
|--------|----------|--------------|
| Test Return | +90.9% | +50.6% |
| Test Sharpe | 1.31 | 1.00 |
| Test Hit Rate | 52.7% | 53.8% |
| Max Drawdown | -22.8% | -26.3% |

Best validation Sharpe achieved: 1.37

---

*End of specification.*
