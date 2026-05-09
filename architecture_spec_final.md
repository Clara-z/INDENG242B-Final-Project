# Architecture Specification — Portfolio RL Agent

**Document purpose:** Authoritative reference for code generation and inter-team integration. Read end to end before implementing.

**Project scope:** Daily-rebalancing portfolio allocation over 30 fixed NASDAQ-100 tickers, trained with PPO, evaluated post-cutoff with leakage-aware diagnostics.

---

## 1. Universe and Time Range

- **Universe:** 30 NASDAQ-100 tickers continuously listed from 2020-01-01 to 2026-01-31. Frozen list saved at `data/universe.json` with fields: `ticker`, `company_name`, `cik`, `gics_sector`, `entity_match_patterns`.
- **Splits (chronological, no shuffle):**
  - Train: 2020-01-01 → 2023-06-30
  - Validation: 2023-07-01 → 2023-12-31
  - Test: 2024-01-01 → 2026-01-31 (post-cutoff, single-evaluation, walk-forward windows applied)
- **Trading calendar:** NYSE, ~252 trading days/year, holidays excluded.

---

## 2. Data Contract Between Teams

The data engineering team produces three Parquet files; the modeling team consumes them. **No other data flows between teams.** Schemas are binding.

### 2.1 `data/processed/asset_features.parquet`

One row per (ticker, trading_date). All features computed using only data available *strictly before* market open on `trading_date`.

| Column | Dtype | Notes |
|---|---|---|
| `ticker` | string | From universe.json |
| `date` | date | Trading day, NYSE calendar |
| `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`, `ret_252d` | float32 | Log returns over lookback. NaN if insufficient history. |
| `vol_20d`, `vol_60d` | float32 | Realized stdev of daily log returns |
| `volume_zscore_20d` | float32 | (volume - mean_20d) / std_20d |
| `price_to_ma20`, `price_to_ma50` | float32 | Close / MA |
| `high_low_range_5d` | float32 | (high_5d - low_5d) / close |
| `rsi_14` | float32 | Standard 14-period RSI |
| `macd_signal` | float32 | MACD signal line |
| `pe`, `pb`, `roe`, `de_ratio`, `profit_margin`, `asset_turnover`, `current_ratio`, `gross_margin` | float32 | Most recently *filed* fundamentals (EDGAR `filed` date < `trading_date`) |
| `days_since_filing` | int32 | Days between most recent filing and `trading_date` |
| `sector_<S>` | int8 | 11 columns, one-hot for GICS sectors |

**Total:** 15 price/vol + 8 fundamentals + 1 days_since_filing + 11 sector = **35 scalar columns** per row.

### 2.2 `data/processed/news_embeddings.parquet`

One row per (ticker, trading_date). Includes a missing-news flag.

| Column | Dtype | Notes |
|---|---|---|
| `ticker` | string | |
| `date` | date | Trading day. News from prior 24h pre-market open. |
| `emb_0` … `emb_767` | float32 | Mean-pooled FinBERT [CLS] embeddings across articles for this (ticker, date). All zeros if `has_news=0`. |
| `n_articles` | int32 | Article count used for mean pool |
| `has_news` | int8 | 1 if at least one article matched, else 0 |
| `mean_tone` | float32 | Mean V2Tone score (sanity-check feature, optional) |

### 2.3 `data/processed/market_features.parquet`

One row per trading_date. Single time series, broadcast to all tickers in the env.

| Column | Dtype | Notes |
|---|---|---|
| `date` | date | Trading day |
| `vix` | float32 | VIX close from prior day |
| `vix_change_20d` | float32 | (vix_t - vix_{t-20}) / vix_{t-20} |
| `tsy_10y_2y_spread` | float32 | 10y minus 2y Treasury yield, % |
| `spy_ret_20d` | float32 | 20-day log return of SPY |
| `spy_vol_20d` | float32 | 20-day realized vol of SPY |

### 2.4 `data/processed/prices.parquet`

For environment use only — agent does not see raw prices, but the env needs them to compute realized returns.

| Column | Dtype | Notes |
|---|---|---|
| `ticker` | string | |
| `date` | date | |
| `close_adj` | float32 | Split- and dividend-adjusted close |
| `is_trading_day` | int8 | 1 for NYSE trading days |

### 2.5 Hard data integrity tests (`tests/test_no_leakage.py`)

These are blocking — must pass on every PR before merge:

```python
def test_news_predates_trading_open():
    # All articles aggregated into news embedding for date D
    # must have publication timestamp < market open on D (UTC).

def test_fundamentals_filed_before_state_date():
    # For every (ticker, date) in asset_features, the underlying
    # filing's `filed` date < date.

def test_no_future_in_returns():
    # Verify `ret_Nd` at date D depends only on prices up to D-1.

def test_universe_alignment():
    # All three feature parquets have identical (ticker, date) keys
    # for asset_features and news_embeddings; market_features matches dates.

def test_split_dates_dont_overlap():
    # Train.max(date) < Val.min(date) < Test.min(date)
```

### 2.6 Standardization

- Compute per-feature `mean` and `std` on **train split only** for all scalar features in `asset_features.parquet` and `market_features.parquet`.
- Save constants to `data/processed/feature_stats.json`.
- Apply `(x - mean) / std` then `clip(-5, 5)` at env-load time, identically across train/val/test/inference.
- News embeddings are **NOT** standardized — they're already in FinBERT's coordinate system.

---

## 3. Environment Specification

`PortfolioEnv` — Gymnasium-compatible. Implemented as a custom class.

### 3.1 Episode structure

- **Episode length:** 252 trading days
- **Episode start:** randomly sampled from training-window dates with at least 252 days remaining
- **Termination:** end of 252 steps OR drawdown > 25% from episode high
- **Step frequency:** daily, close-to-close

### 3.2 Action space

```python
self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(30,), dtype=np.float32)
```

Note: although the policy outputs Dirichlet samples (which are valid simplex points by construction), the action space is declared as Box for SB3 compatibility. The env validates and projects (see 3.4).

### 3.3 Observation space

```python
self.observation_space = gym.spaces.Dict({
    "assets": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30, 35 + 768 + 1 + 1), dtype=np.float32),
    # 35 scalar + 768 news + 1 has_news flag + 1 own_prev_weight = 805 per asset
    "global_portfolio": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
    # portfolio_vol_20d, days_since_rebalance, cumulative_episode_return
    "market": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
})
```

**Observation construction at step t:**

For each ticker $i$:
1. Look up `asset_features` row for (ticker_i, date_t) → 35 scalars
2. Look up `news_embeddings` row → 768 emb dims + has_news flag
3. Append agent's previous weight $w_{t-1, i}$ (1 scalar)
4. Concatenate → 805-dim row

Stack across 30 tickers → `(30, 805)` matrix for `assets` field.

For global features: compute portfolio vol from the last 20 days of realized returns; days_since_rebalance and episode return tracked internally; market features looked up from `market_features.parquet`.

### 3.4 Action processing inside env

The env receives a Box action $a \in [0,1]^{30}$ from SB3. Three steps:

1. **Validate**: assert all entries non-negative, shape (30,)
2. **Normalize**: $w = a / \sum a$ (defensive — Dirichlet samples are already on simplex)
3. **Apply max-weight cap**: if any $w_i > 0.25$, clip to 0.25, redistribute excess proportionally to remaining names. Iterate until all $w_i \leq 0.25$ (typically 1-2 iterations).

### 3.5 Reward

The environment supports two reward variants, selected via env config flag `reward_mode ∈ {"simple", "icvar"}`. Both must be implemented; both will be trained and compared.

**Variant A — Simple log-return with turnover penalty (default):**

$$r_t = \log\left(\frac{V_{t+1}}{V_t}\right) - \kappa \cdot \|w_t - w_{t-1}\|_1$$

where:
- $V_{t+1}/V_t = 1 + \sum_i w_{t,i} \cdot r_{t \to t+1, i}$
- $\kappa = 0.0005$ (5 basis points per unit turnover)

This is a linear approximation to the exact transaction remainder factor derived in Jiang et al. (2017, Theorem 1). For daily rebalancing the approximation error is negligible.

**Variant B — ICVaR-augmented reward (Gu et al., 2025):**

$$r_t = \log\left(\frac{V_{t+1}}{V_t}\right) - \kappa \cdot \|w_t - w_{t-1}\|_1 - \lambda \cdot \text{ICVaR}_{\alpha,t}(X)$$

Definitions:
- $X = \{x_1, x_2, \ldots, x_t\}$ is the realized portfolio return series within the current episode
- $\text{VaR}_{\alpha,t}(X) = -\inf\{x_k \in \mathbb{R} : F_X(x_k) > \alpha\}$, the empirical VaR at confidence $\alpha$
- $\text{CVaR}_{\alpha,t}(X) = \text{VaR}_{\alpha,t}(X) + \frac{1}{\alpha t} \sum_{k=1}^t \max(-x_k - \text{VaR}_{\alpha,k}(X), 0)$
- $\text{ICVaR}_{\alpha,t}(X) = \text{CVaR}_{\alpha,t}(X) - \text{CVaR}_{\alpha,t-1}(X)$

Hyperparameters:
- $\alpha = 0.05$ (5% tail confidence)
- $\lambda = 0.5$ (risk aversion; ablate in {0.1, 0.5, 1.0})
- For the first 20 steps of an episode, set ICVaR contribution to 0 (insufficient history)

**Implementation requirement:** the env tracks the within-episode return history `self._return_history: list[float]` and recomputes VaR/CVaR each step. CVaR from the previous step is cached in `self._prev_cvar` to compute ICVaR.

**Reset:** clear return history and previous CVaR on every `env.reset()`.

### 3.6 Action timing — explicit

- Day t-1 close: features computed using only data through close on t-1
- **Day t open: agent submits $w_t$**
- Day t close: portfolio realizes return from $w_t$ applied to actual close-to-close moves t-1 → t
- Cost paid on $\|w_t - w_{t-1}\|_1$, deducted from reward

This is a one-day execution lag. Document in env docstring. Any code path that lets the agent see day-t prices when choosing $w_t$ is a bug.

---

## 4. Network Architecture

Subclasses `stable_baselines3.common.policies.ActorCriticPolicy`. Total trainable parameters: ~400K. FinBERT (110M) is **frozen and not part of this network** — it's used in offline preprocessing only.

### 4.1 Forward pass

```
INPUT (per batch item):
  obs["assets"]:           (B, 30, 805)   # 35 scalar + 768 news + 1 has_news + 1 prev_weight
  obs["global_portfolio"]: (B, 3)
  obs["market"]:           (B, 5)

(0) SPLIT per-asset input:
    scalars   = obs["assets"][..., :35]            (B, 30, 35)
    news_emb  = obs["assets"][..., 35:803]         (B, 30, 768)
    has_news  = obs["assets"][..., 803:804]        (B, 30, 1)
    prev_w    = obs["assets"][..., 804:805]        (B, 30, 1)

(1) NEWS PROJECTION:
    news_proj = Linear(768 → 64) → LayerNorm
    Output: (B, 30, 64)

(2) PER-ASSET FEATURE ASSEMBLY:
    asset_input = concat([scalars, news_proj, has_news, prev_w], dim=-1)
    Output: (B, 30, 35 + 64 + 1 + 1) = (B, 30, 101)

(3) PER-ASSET MLP ENCODER (weight-shared across 30 assets):
    Linear(101 → 128) → GELU → Dropout(0.1) →
    Linear(128 → 128) → GELU → LayerNorm
    Output: (B, 30, 128)

(4) CROSS-ASSET ATTENTION (1× transformer block, no positional encoding):
    nn.TransformerEncoderLayer(
        d_model=128, nhead=4, dim_feedforward=256,
        dropout=0.1, activation="gelu",
        batch_first=True, norm_first=True
    )
    Output: (B, 30, 128)

(5) GLOBAL CONCATENATION:
    global = concat([obs["global_portfolio"], obs["market"]], dim=-1)   # (B, 8)
    global_broadcast = global.unsqueeze(1).expand(-1, 30, -1)           # (B, 30, 8)
    h = concat([attn_out, global_broadcast], dim=-1)                    # (B, 30, 136)

(6) SHARED TRUNK:
    Linear(136 → 128) → GELU → Dropout(0.1) → LayerNorm
    Output h: (B, 30, 128)

(7) ACTOR HEAD (decomposed Dirichlet via mean × concentration):
    (7a) Per-asset logits:
         Linear(128 → 64) → GELU → Linear(64 → 1)
         Output logits: (B, 30)
         mean = softmax(logits, dim=-1)                                 # (B, 30), sums to 1

    (7b) Concentration scalar (uses attention-pooled global summary):
         q = self.global_query.expand(B, 1, 128)   # learnable parameter (1, 1, 128)
         pooled, _ = MultiheadAttention(d=128, heads=4)(q, h, h)        # (B, 1, 128)
         pooled = pooled.squeeze(1)                                     # (B, 128)
         log_c = Linear(128 → 64) → GELU → Linear(64 → 1)               # (B, 1)
         c = (softplus(log_c) + 1.0).clamp(max=100.0)                   # (B, 1)

    (7c) Compose:
         alpha = mean * c + 1e-3                                        # (B, 30)
         dist = torch.distributions.Dirichlet(alpha)

(8) CRITIC HEAD (uses same attention-pooled summary):
    Linear(128 → 64) → GELU → Linear(64 → 1)
    Input: pooled from (7b) — REUSE, do not recompute
    Output V(s): (B,)

RETURNS:
    dist (Dirichlet over simplex), value V(s)
```

> **Note on architectural lineage:** the per-asset MLP encoder with weight-sharing across assets follows the EIIE topology of Jiang et al. (2017). The cross-asset attention block is our extension; their original EIIE coupled assets only through the final softmax. The Dirichlet output replaces their softmax-of-scores parameterization for proper simplex-distribution properties under PPO.

### 4.2 Action sampling and log-probability

```python
# Sample
action = dist.rsample()                # (B, 30), guaranteed on simplex
log_prob = dist.log_prob(action)       # (B,)
entropy = dist.entropy()               # (B,)
```

`rsample` uses reparameterization for differentiable sampling; PPO doesn't strictly need it but doesn't harm.

### 4.3 Numerical safety

- Clamp `c` upper at 100 to prevent peaked Dirichlet → exploding gradients
- Clamp `alpha` minimum at 1e-3 (already enforced by `mean * c + 1e-3`)
- Sampled weights: add tiny epsilon and renormalize before downstream env use, in case any entry rounds to exactly 0:
  ```python
  action = action + 1e-8
  action = action / action.sum(dim=-1, keepdim=True)
  ```

### 4.4 Initialization

- Linear layers: PyTorch default (Kaiming uniform)
- LayerNorm: default (gain=1, bias=0)
- `global_query` parameter: `nn.Parameter(torch.randn(1, 1, 128) * 0.02)`
- Concentration head's final Linear: bias initialized to `log(softplus_inv(10.0 - 1.0)) ≈ 2.2` so initial $c \approx 10$ (moderately diverse)

### 4.5 Permutation equivariance / invariance — required guarantees

- **Actor must be permutation-equivariant:** if input assets are reordered, output weights reorder identically
- **Critic must be permutation-invariant:** V(s) is unchanged under asset reordering
- Sanity test in `tests/test_perm_equivariance.py`:
  ```python
  def test_actor_equivariance():
      perm = torch.randperm(30)
      out_orig = policy(obs)
      out_perm = policy(permute_assets(obs, perm))
      assert torch.allclose(out_orig.mean[:, perm], out_perm.mean, atol=1e-5)

  def test_critic_invariance():
      assert torch.allclose(out_orig.value, out_perm.value, atol=1e-5)
  ```

These tests must pass before any training run.

---

## 5. PPO Configuration

Stable-Baselines3 `PPO` class with custom policy.

### 5.1 Hyperparameters

```python
ppo_kwargs = dict(
    learning_rate=3e-4,
    n_steps=2048,           # rollout buffer size per update
    batch_size=256,
    n_epochs=10,            # gradient passes per rollout
    gamma=0.99,             # discount
    gae_lambda=0.95,        # GAE
    clip_range=0.2,
    ent_coef=0.01,          # entropy bonus — tune up if policy collapses
    vf_coef=0.5,            # critic loss weight
    max_grad_norm=0.5,      # gradient clipping
    target_kl=0.02,         # early-stop epoch if KL exceeds
)
```

### 5.2 Training budget

- Total steps: 5M
- Wall-clock estimate: ~2–3 hours on Colab T4
- Seeds: train 3 separately, report mean ± std on all metrics
- Model checkpoints: every 500K steps to W&B + disk

### 5.3 Logging (Weights & Biases)

Required metrics per update:
- `train/episode_return`
- `train/episode_sharpe` (rolling 60-day from rollout)
- `policy/entropy`
- `policy/concentration_mean` (mean of `c` across batch)
- `policy/max_weight` (max single-name weight, averaged across batch)
- `policy/effective_n_holdings` ($1 / \sum w_i^2$, the inverse Herfindahl)
- `loss/policy_loss`, `loss/value_loss`, `loss/total`
- `kl_divergence`
- `val/sharpe_per_eval` (every 100K steps, run val rollout)

### 5.4 Early stopping

- Validate every 100K steps on val window (single 6-month rollout)
- Track best val Sharpe; save best checkpoint to disk
- No automatic stopping — train all 5M steps, select best-val checkpoint at end

---

## 6. Evaluation Protocol

### 6.1 Test-set evaluation

- Use **best-val checkpoint** (across 3 seeds, pick the seed with best val Sharpe; or evaluate all 3 and report distribution)
- Run policy deterministically on test window: use `dist.mean` instead of `dist.rsample()` for evaluation
- Apply same env (transaction costs, max-weight cap) as training

### 6.2 Walk-forward windows on test set

- Split 2024-01 → 2026-01 into 4 non-overlapping 6-month windows: 2024H1, 2024H2, 2025H1, 2025H2
- Run evaluation on each window independently
- Report Sharpe, CAGR, MDD per window + aggregated mean ± std

### 6.3 Metrics

| Metric | Formula | Notes |
|---|---|---|
| Sharpe (annualized) | $\sqrt{252} \cdot \bar{r}/\sigma_r$ | Net of transaction costs, $r_f = 0$ in excess-return mode |
| Sortino (annualized) | $\sqrt{252} \cdot (\bar{r} - r_{\text{MAR}})/\sigma_{\text{down}}$ | $r_{\text{MAR}} = 0.03/252$ daily MAR; $\sigma_{\text{down}}$ over returns below MAR |
| Omega ratio | $\frac{\int_{r_{\text{MAR}}}^{\infty}(1-F(x))dx}{\int_{-\infty}^{r_{\text{MAR}}}F(x)dx}$ | Computed empirically from daily returns, $r_{\text{MAR}} = 0.03/252$ |
| CAGR | $(V_T / V_0)^{252/T} - 1$ | |
| Max drawdown | $\max_t (V_t - \min_{s \geq t} V_s) / V_t$ | |
| Calmar | CAGR / |MDD| | |
| Avg turnover | $\bar{\|w_t - w_{t-1}\|_1}$ | Daily mean |
| Effective N holdings | $\bar{1/\sum w_i^2}$ | Daily mean, inverse Herfindahl |

Sortino and Omega added per Gu et al. (2025) reporting standard. Both use a 3% annualized minimum acceptable return ($r_{\text{MAR}}$).

### 6.4 Required experiments (all on test set)

1. **Main result:** our model vs all 6 baselines, all metrics, walk-forward
2. **Pre-cutoff vs post-cutoff Sharpe:** same model, run on 2022H1+H2 vs 2025H1+H2; document the gap
3. **Counterfactual ticker anonymization:** at test time, replace ticker mentions and company names in news with synthetic IDs (`STOCK_0042`) before encoding. Re-run policy. Report:
   - KL divergence between original and counterfactual action distributions, averaged over test days
   - Sharpe gap (original − counterfactual)
4. **No-news ablation:** retrain with `news_embeddings` zeroed out; report Sharpe gap
5. **Reward ablation:** train two policies with identical architecture, one with simple log-return reward, one with ICVaR-augmented reward. Report all metrics for both. Discuss whether ICVaR improves Sharpe/Sortino at the cost of CAGR (per Gu et al. 2025's finding).
6. **Multi-seed:** all of the above × 3 seeds

### 6.5 Baselines to implement

| # | Baseline | Implementation |
|---|---|---|
| 1 | Equal-weight buy & hold | Static $w = 1/30$ |
| 2 | NASDAQ-100 (QQQ) | Single-asset buy & hold |
| 3 | Mean-variance | Rolling 60d covariance, max-Sharpe weights, max 25% per name |
| 4 | Risk parity | Inverse-vol weights, rebalance daily |
| 5 | Momentum | Top-10 by 60d return, equal-weight, monthly rebalance |
| 6 | Deep supervised | Transformer predicting next-day returns, Markowitz allocation |
| 7 (optional) | Best Stock (oracle) | Single best-performing asset over test window. Cannot be implemented in real time. Used only as upper-bound reference. |

---

## 7. Critical Risks and Mitigations

These have caused project failures in similar work. Address explicitly.

### 7.1 Survivorship bias

- Universe is "30 tickers continuously listed 2020-2026"
- Document as known limitation
- Stretch: dynamic universe via Wikipedia revision history of NDX constituents

### 7.2 Look-ahead in fundamentals

- Use SEC EDGAR `filed` field, not period-end
- `tests/test_no_leakage.py` enforces it
- Failed test = blocked PR

### 7.3 Look-ahead in news

- News timestamps in UTC; articles after 21:00 UTC (NYSE close) attribute to next trading day
- Document timezone handling in news pipeline docstrings

### 7.4 PPO seed variance

- 3 seeds minimum; 5 if compute allows
- All reported numbers as mean ± std
- Single-seed numbers in any figure = blocked review

### 7.5 Reward hacking

- If policy collapses to one stock or extreme concentration: turn on risk penalty
- Monitor `policy/effective_n_holdings` — should stay above ~5
- Monitor `policy/max_weight` — should rarely hit 0.25 cap

### 7.6 Encoder pretraining contamination

- FinBERT was trained on text potentially including 2024–2026 events
- The counterfactual experiment (6.4 #3) is the diagnostic — if anonymization barely changes actions, the embeddings are leaking memorized identities
- Pre-cutoff vs post-cutoff Sharpe gap (6.4 #2) is the second diagnostic

### 7.7 Reward function sensitivity

PPO is known to be sensitive to reward scaling and shaping. The ICVaR term can dominate or vanish depending on $\lambda$. Mitigations:
- Log per-component reward magnitudes during training (`reward/log_return`, `reward/turnover_cost`, `reward/icvar_penalty`)
- If ICVaR magnitude exceeds log-return magnitude by >10× consistently, reduce $\lambda$
- Compare convergence behavior of simple vs ICVaR reward across seeds — if ICVaR variant is wildly less stable, document and consider $\lambda < 0.1$

---

## 8. Repository Structure (binding)

```
news-rl-2026/
├── README.md
├── pyproject.toml          # pinned: torch==2.4, sb3==2.3.2, gymnasium==0.29.1, transformers==4.44, etc.
├── .env.example
├── data/
│   ├── universe.json
│   ├── raw/                # gitignored
│   ├── processed/          # gitignored
│   │   ├── asset_features.parquet
│   │   ├── news_embeddings.parquet
│   │   ├── market_features.parquet
│   │   ├── prices.parquet
│   │   └── feature_stats.json
│   └── README.md           # how to reproduce
├── src/
│   ├── data/
│   │   ├── universe.py     # build universe.json
│   │   ├── prices.py       # yfinance pull → prices.parquet
│   │   ├── fundamentals.py # EDGAR XBRL → fundamentals subset
│   │   ├── news.py         # GDELT BigQuery + cleaning + ±N sentence context
│   │   ├── embeddings.py   # frozen FinBERT inference, mean-pool, write parquet
│   │   ├── market.py       # VIX, treasuries, SPY → market_features.parquet
│   │   └── features.py     # joins everything, produces asset_features.parquet, computes feature_stats.json
│   ├── env/
│   │   ├── portfolio_env.py   # gym.Env subclass, observation+action spaces, step, reset
│   │   └── __init__.py        # gym.register
│   ├── models/
│   │   ├── policy.py          # PortfolioPolicy(ActorCriticPolicy) — the spec in §4
│   │   └── baselines.py       # equal-weight, MV, risk parity, momentum, deep supervised
│   ├── training/
│   │   ├── train.py           # PPO training entry point
│   │   └── config.py          # ppo_kwargs, paths, seeds
│   ├── eval/
│   │   ├── backtest.py        # apply policy to a window, return metrics
│   │   ├── walk_forward.py    # split test, aggregate
│   │   ├── counterfactual.py  # ticker anonymization experiment
│   │   ├── pre_post_cutoff.py
│   │   └── metrics.py
│   └── utils/
│       └── seeding.py
├── notebooks/
│   ├── 01_data_eda.ipynb
│   ├── 02_baseline_results.ipynb
│   └── 03_final_figures.ipynb
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_no_leakage.py
│   ├── test_perm_equivariance.py
│   ├── test_action_validity.py
│   └── test_env_reward.py
└── report/
    ├── main.tex
    └── figures/
```

---

## 9. Inter-Team Sync Points

The data team's deliverables are the four parquet files in §2. The modeling team's deliverables consume them. **Coupling points:**

1. **Schema freeze:** §2.1–2.4 schemas are locked after first review. Any change requires both team leads to sign off and bumps a version number.
2. **Universe freeze:** `universe.json` produced first, frozen, both teams reference it.
3. **Standardization constants:** `feature_stats.json` produced by data team, consumed by env. Single source of truth.
4. **News embeddings:** data team responsible for FinBERT inference. Modeling team trusts the parquet and never re-runs FinBERT.
5. **Test split sanctity:** modeling team agrees not to look at test-window metrics until the final eval phase. Data team ensures test-window parquets exist but tests/code reference only train+val until then.

---

## 10. Related Work and Positioning

The project sits at the intersection of three lines of work, all of which the report's related-work section must address explicitly.

**Deep RL for portfolio management.** Jiang et al. (2017, "Deep Portfolio Management") established the foundational framework for applying deep RL to multi-asset portfolio allocation: weight-shared per-asset evaluators (the EIIE topology), portfolio-vector memory for transaction-cost-aware decisions, and an explicit reward function based on log returns. Our architecture is a transformer-augmented EIIE — we retain weight-shared per-asset encoding and previous-weight conditioning, and add explicit cross-asset attention to replace softmax-only coupling. We use a linear approximation to the transaction remainder factor; the exact iterative form is given in Jiang et al. (2017, Theorem 1).

**Risk-aware DRL portfolio strategies.** Gu et al. (2025, "MTS") demonstrated that augmenting the reward function with Incremental Conditional Value at Risk (ICVaR) consistently improves risk-adjusted performance on US equity markets across diverse market regimes. We adopt their ICVaR formulation as one of two reward variants and replicate their multi-metric reporting (Sharpe, Sortino, Omega) and ablation methodology. Our work differs from MTS in three ways: (a) we add news-conditioning via frozen FinBERT embeddings rather than relying on price + technical indicators alone, (b) we use cross-asset transformer attention for explicit pairwise interaction rather than time-aware attention with hand-crafted weekly/monthly masks, and (c) we conduct leakage-aware evaluation against post-cutoff data with counterfactual diagnostics.

**Look-ahead bias in LLM-based financial agents.** Recent work has documented that many LLM-based trading agents fail to beat random baselines once evaluation crosses the model's knowledge cutoff (Profit Mirage, 2510.07920; Evaluating LLMs in Finance, 2602.14233). Our project's central methodological contribution is treating this risk as first-class: strict pre-cutoff training, single post-cutoff test window, and a counterfactual ticker-anonymization experiment that probes whether the policy depends on news content or on memorized ticker identities.

**Positioning summary:** we build on the EIIE topology (Jiang 2017) and ICVaR risk shaping (Gu 2025), extend with news conditioning and explicit cross-asset attention, and evaluate under a leakage-aware protocol. We do not claim novelty in the RL algorithm (PPO is standard) or the broader problem framing (portfolio allocation as MDP is well-established).

---

## 11. Out of Scope (Explicit)

To prevent scope creep mid-build:

- Long-short portfolios (Dirichlet is long-only; would need different action parameterization)
- Intraday rebalancing
- Recurrent policy (LSTM/GRU)
- FinBERT fine-tuning
- Dynamic universe (constituent changes over time)
- Multi-frequency rebalancing (daily only)
- Options or derivatives
- Real-time deployment (only Streamlit demo with static post-cutoff data)

If these come up, defer to "future work" section of report.

---

This spec is complete enough to feed to Codex section by section. The likely failure modes have all been called out 喵.
