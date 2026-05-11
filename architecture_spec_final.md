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
| **Full data** | 2018-01-02 | 2025-12-31 | ~2,010 |
| **Train** | 2018-01-01 | 2022-12-31 | ~1,007 |
| **Validation** | 2023-01-01 | 2023-12-31 | ~250 |
| **Test** | 2024-01-01 | 2025-12-31 | ~502 |

**Default in code:** `rl/train_portfolio_rl.py` uses `--train-start 2018-01-01` (first aligned trading day is 2019-01-03 after panel cleaning drops rows where any asset has NaN features due to long lookbacks like `ret_252d`). Validation uses the full 2023 calendar year (~250 days), replacing the earlier 6-month window which was too short for reliable Sharpe estimation.

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
    policy_kwargs={
        "feature_cols": train.feature_cols,
        "optimizer_kwargs": {"weight_decay": 1e-4},  # L2 regularization
    },
    learning_rate=_linear_decay(3e-4, min_lr=1e-5),  # 3e-4 → 1e-5 linear decay
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,       # lowered from 0.01 — entropy was dominating policy gradient
    vf_coef=0.5,
    max_grad_norm=0.3,    # tightened from 0.5 to prevent late-stage gradient spikes
    target_kl=0.15,       # raised from 0.02 — Dirichlet log-probs are ~50× Gaussian scale
    seed=42,
)
```

**Key deviations from SB3 defaults and rationale:**

| Parameter | SB3 default | Current value | Why |
|-----------|-------------|---------------|-----|
| `ent_coef` | 0.0 | 0.001 | With 0.01 entropy dominated 1000× over policy gradient — policy locked at equal-weight |
| `target_kl` | None | 0.15 | Dirichlet log-prob magnitude (~73–9000) is far above Gaussian range — 0.02 fires after 4–5 minibatch steps |
| `max_grad_norm` | 0.5 | 0.3 | Late-stage Dirichlet concentration collapse caused approx_kl spikes >1 without tighter clipping |
| `learning_rate` | constant | linear decay | Constant 3e-4 caused entropy explosion at ~220k steps; decay to 1e-5 prevents late instability |
| `dropout` | N/A | **0.0 (disabled)** | PPO collects rollouts in eval mode but updates in train mode — dropout makes `old_log_prob ≠ new_log_prob` at step 0, triggering spurious KL violations |

### 4.3 Training Budget

| Setting | Recommended | Quick Test |
|---------|-------------|------------|
| Total timesteps | 5,000,000 | 300,000 |
| Eval frequency | 25,000 | 25,000 |
| Episode length | 252 | 252 |

**Measured training time (Mac M2/M4, CPU only):**
- 5M steps: **~77 minutes** (measured, run `20260511_031334`)
- 300K steps: ~8–10 minutes

**Note:** The eval frequency of 100,000 steps in early estimates was too coarse. With `--eval-freq 25000` the val Sharpe trajectory is much more granular and checkpoint selection is more accurate at the cost of ~5% extra wall time per 5M run.

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
├── run_config.txt          # All hyperparameters, splits, model selection formula, timing
├── models/
│   ├── best_model.zip      # Best composite-score checkpoint
│   └── final_model.zip     # Final training checkpoint
├── metrics/
│   ├── train_metrics.csv   # Per-rollout training stats (LR, entropy, KL, clip_fraction…)
│   ├── val_metrics.csv     # val_return, val_sharpe, val_hit_rate, val_max_drawdown, val_score
│   ├── test_daily_returns.csv  # Daily RL returns and equity curve
│   └── summary.json        # Run config + final metrics + training_time
└── plots/
    ├── loss_curves.png
    ├── avg_reward.png
    ├── val_sharpe_hit_rate.png
    └── test_equity_curve.png
```

`run_config.txt` is written immediately when the run directory is created (before training starts) and a `[TIMING]` block is appended on completion. It is the primary artifact for tracing hyperparameter choices across runs.

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

### 9.1 Data and Universe

1. **Survivorship bias:** Universe is fixed at 30 tickers selected retrospectively. Stocks that were delisted, merged, or went bankrupt during 2018–2025 are absent. The model is trained and evaluated on survivors only, which upward-biases expected returns.
2. **No intraday price data:** `high_low_range_5d` is set to 0.0 because the source CSV lacks daily high/low prices. True volatility proxies (ATR, Garman-Klass) are unavailable.
4. **Fixed PCA basis:** The 32 PCA components are fitted on all news in the corpus before training. If the semantic topics of financial news shift over the test window (e.g., AI in 2023–2025 vs. COVID in 2020), earlier PCA dimensions may not capture the new dominant themes.

### 9.2 Training Stability

5. **Dirichlet log-prob scale mismatch with PPO:** The Dirichlet distribution's log-probability over a 30-asset simplex is typically in the range +40 to +100 — roughly 50–100× larger in magnitude than the Gaussian log-probs PPO was designed for. This makes the standard `target_kl=0.02` threshold fire after only 4–5 minibatch updates. The working threshold used here (`target_kl=0.1`) is an empirical fix, not a principled value.
6. **Dropout incompatible with PPO's KL check:** Standard dropout introduces a train/eval mode discrepancy. During rollout collection the policy runs in eval mode (dropout off); during the update step it runs in train mode (dropout on). This means `old_log_prob ≠ new_log_prob` even before any gradient step is applied — the KL check fires at step 0 as training progresses and the Dirichlet distribution sharpens. Dropout is disabled in this implementation as a result; weight decay is used instead.
7. **Entropy explosion at high learning rate:** With `ent_coef=0.001`, the policy gradient has enough influence to move the Dirichlet concentration parameters toward zero. Beyond ~200k steps at constant `lr=3e-4`, the concentration collapses, entropy diverges (observed reaching 1900 in training logs), and the policy becomes incoherent. Linear LR decay with a floor partially mitigates this but the fundamental tension between exploration (low concentration) and stability (bounded entropy) is not resolved.
8. **Policy slow to differentiate from equal-weight:** The Dirichlet distribution initialized at `alpha_i ≈ 2` for all assets is nearly uniform. With `ent_coef=0.01`, entropy regularization dominated the policy gradient (~1000×), keeping the policy locked near uniform allocation through 300k steps. Even with `ent_coef=0.001`, the hit rate variation across checkpoints (0.52–0.58) is modest, suggesting the policy has only weakly learned to differentiate asset bets.

### 9.3 Evaluation and Model Selection

9. **Validation window regime bias:** The current 2023 validation window was an unusually strong bull year for tech/growth stocks (+24% S&P, higher for concentrated portfolios). Any near-equal-weight allocation of these 30 tickers achieves val Sharpe ~1.7–1.8 regardless of skill. Model selection on this window selects checkpoints that happen to be closest to equal-weight, not those with the best differentiation signal.
10. **Absolute Sharpe as selection criterion:** The composite score `val_sharpe − 0.3 × |val_max_drawdown|` penalizes drawdown but does not control for market beta. A better selection criterion would be `val_sharpe − baseline_sharpe` (excess Sharpe over equal-weight), which directly measures whether the policy adds value beyond naive diversification.
12. **Single seed:** All results are from seed 42. Variance across random initialisations is unknown.

### 9.4 Architecture

13. **Flat observation vs. structured input:** The environment exposes a flat 2,223-dimensional `Box` for SB3 compatibility. The policy reshapes it internally, but this means the environment's observation space cannot be used directly for debugging or alternative policy implementations without understanding the internal reshape logic.
14. **Global portfolio state only captures short-term dynamics:** The three global state variables (20-day portfolio vol, steps since episode start, cumulative episode return) have no memory beyond 20 days. The policy cannot condition on longer-term regime state (e.g., bear vs. bull market phase) except indirectly through the standardized feature values.

---

## 10. Example Results

All runs: train 2019-01-03 → 2022-12-30 (1,007 days), val 2023 (250 days), test 2024-01-02 → 2025-12-31 (502 days). Config: `ent_coef=0.001`, `target_kl=0.15`, `lr=3e-4 → 1e-5` linear decay, `max_grad_norm=0.3`, composite model selection (`val_sharpe − 0.3 × |val_max_drawdown|`).

### Run A — 300k steps (`20260511_023358`, ~8 min)

| Metric | RL Agent | Equal-Weight |
|--------|----------|--------------|
| Test Return | +45.8% | +46.3% |
| Test Sharpe | 0.953 | 0.963 |
| Test Hit Rate | 53.9% | 54.1% |
| Max Drawdown | −26.4% | −26.3% |
| Best Val Score | 1.766 (at t=25k) | — |

Policy nearly identical to equal-weight. Val Sharpe stable across all checkpoints (~1.76–1.83), val hit rate locked at 0.554. Policy did not differentiate assets meaningfully at this training budget.

---

### Run B — 5M steps (`20260511_031334`, 77 min)

| Metric | RL Agent | Equal-Weight |
|--------|----------|--------------|
| Test Return | **−12.9%** | +46.3% |
| Test Sharpe | **−0.165** | 0.963 |
| Test Hit Rate | 52.1% | 54.1% |
| Max Drawdown | −28.9% | −26.3% |
| Best Val Score | 2.028 (at t=1.175M) | — |
| Training time | 77 min | — |

**Key observations:**

1. **Val Sharpe peaked at ~2.12 around t=1.175M then declined steadily** — the policy learned something useful in the first million steps (val score climbed from 1.3 to 2.12), then degraded. The selected checkpoint had val return +80% and val Sharpe 2.12 but this did not generalise to the 2024–2025 test period.

2. **Entropy explosion** — `train/entropy_loss` rose from ~73 at t=0 to 4,000–9,000+ by t=5M. The Dirichlet concentration parameters drifted to extreme values that maximised 2023 performance at the cost of generalisation. Linear LR decay with floor 1e-5 slowed but did not prevent this.

3. **Val hit rate varied (0.54–0.58)** — unlike 300k runs where hit rate was static, the 5M policy made real asset-level bets. Those bets worked on 2023 val but reversed on 2024–2025 test.

4. **Overfitting to the 2023 bull regime** — 2023 was a strong bull year for tech/growth. The selected checkpoint concentrated on the positions optimal for that regime, which underperformed severely in the 2025 sell-off (April 2025 tariff crash, late-2025 correction). The equity curve ends at ~0.87 vs equal-weight at ~1.46.

5. **The 5M run is worse than 300k on test** — more training made the model confidently wrong rather than uncertain and close-to-equal-weight. This is a clear case of regime-specific overfitting driven by the single 2023 val window.

**Implication:** The composite `val_sharpe − 0.3×|max_drawdown|` selection criterion with a single 2023 bull-year window is insufficient to prevent regime overfitting at 5M steps. Walk-forward validation or excess-Sharpe selection (RL vs baseline on val) is a prerequisite before extended training is productive.

---

## 11. Future Work

### 11.1 Training Scale and Stability

1. **Fix model selection before scaling further:** A 5M-step run (`20260511_031334`) produced a policy that aggressively underperformed equal-weight on test (−12.9% vs +46.3%), despite a val score of 2.028. More training makes the model *confidently wrong* when the val window is a single bull year. The priority is excess-Sharpe or walk-forward selection **before** running more than 1M steps.
2. **Analytical KL for Dirichlet:** Replace PPO's approximate KL formula (designed for Gaussian policies) with the exact closed-form KL divergence between two Dirichlet distributions. This would make `target_kl` a principled quantity and remove the need for empirical threshold tuning.
3. **Cyclical or cosine LR annealing:** Linear decay with a floor prevents late-stage explosion but suppresses learning in the second half of training. Cosine annealing with warm restarts could periodically re-raise the LR to escape local optima while still decaying toward fine-tuning behavior near convergence.
4. **Population-based training (PBT):** Automatically tune `ent_coef`, `learning_rate`, and `target_kl` across parallel runs, replacing the current manual trial-and-error schedule of hyperparameter changes.

### 11.2 Reward and Objective Design

5. **Excess Sharpe reward:** Replace `log(1 + portfolio_return)` with `log(1 + portfolio_return) − log(1 + equal_weight_return)`, i.e., reward the agent for beating the equal-weight baseline each step, not just for absolute returns. This directly aligns the training objective with the evaluation criterion.
6. **Sortino ratio as validation metric:** Once the val window is extended beyond one year, switch the primary model selection metric from Sharpe to Sortino ratio (penalises only downside volatility), which is more appropriate for a risk-managed portfolio objective.
7. **Calmar ratio in composite score:** Replace or supplement the current `sharpe − 0.3 × |max_drawdown|` composite with the Calmar ratio `annualised_return / |max_drawdown|`, which is the standard institutional metric for drawdown-adjusted performance.
8. **ICVaR reward tuning:** The `--reward-mode icvar` variant is implemented but untested against the simple mode on the current architecture. A systematic comparison using the 2019–2022 bear-market period in training may reveal whether CVaR penalisation improves out-of-sample drawdown without sacrificing return.

### 11.3 Model Selection and Evaluation

9. **Excess Sharpe model selection:** Replace the absolute composite score with `val_sharpe − val_equal_weight_sharpe` as the primary checkpoint selection criterion. This directly measures skill over the baseline and is immune to bull-market inflation of the validation window.
10. **Walk-forward validation:** Instead of a fixed 2023 validation window, use rolling 6-month windows across 2021–2023, selecting the checkpoint that maximises average excess Sharpe across windows. This reduces regime-specific overfitting in model selection.
11. **Multi-seed evaluation:** Report mean ± standard deviation across at least 3 random seeds. Current results are from seed 42 only; variance is unknown and the best checkpoint may reflect favourable initialisation rather than learned policy quality.
12. **Benchmark expansion:** Add momentum (12-1 month) and minimum-variance portfolios as additional baselines alongside equal-weight. These are standard factor-based alternatives that a data-driven RL policy should outperform to demonstrate value.

### 11.4 Architecture Improvements

13. **Structured observation space:** Replace the flat 2,223-dim Box with a dict observation space keyed by `asset_features (30, 73)`, `prev_weights (30,)`, and `portfolio_state (3,)`. This would allow the environment and policy to evolve independently and make debugging observation contents straightforward.
14. **Richer temporal context:** Add a recurrent layer (e.g., GRU per asset) before the cross-asset Transformer to encode asset-level momentum over a 20-day lookback, replacing the current stateless per-step encoding. The environment already stores `_returns` as a list; this history could be exposed in the observation.
15. **Separate news architecture:** The current policy projects news PCA columns through the shared scalar encoder. A dedicated news cross-attention branch (attending over the 32 PCA dims per asset, then mixing across assets) could better capture cross-company news co-movement (e.g., sector-wide sentiment shocks).
16. **Larger universe:** Extending from 30 to 100–500 assets would test the cross-asset Transformer's scalability and provide more diversification opportunity for the policy to exploit. The permutation equivariance property of the architecture already supports variable-length asset lists.

### 11.5 Production Readiness

17. **Periodic retraining:** The current model is trained once on a fixed historical window. A production deployment would require scheduled retraining as new price, fundamental, and news data arrives, along with a data freshness check before each rollout.
18. **Transaction cost modelling:** The current 5bp per unit turnover cost is a simplification. Realistic costs depend on asset liquidity, trade size, and market impact. Incorporating a realistic execution cost model (e.g., square-root impact law) would improve live-trading alignment.
19. **Risk constraints:** The 25% max-weight cap is the only hard constraint. Production portfolios typically require sector exposure limits, factor neutrality, and tracking error bounds, none of which are currently enforced.

---

*End of specification.*
