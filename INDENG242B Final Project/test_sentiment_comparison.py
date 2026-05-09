"""
Sentiment Comparison Test: FinBERT vs Guardian Tone (GDELT-style)
------------------------------------------------------------------
Workflow:
  1. Load 2018 Guardian news text from guardian_news_2018.json.
  2. Load daily FinBERT sentiment from news_embeddings.parquet and
     daily Guardian tone from guardian_finbert_features_2018_2025.parquet.
  3. Align Guardian article text to trading dates and merge to (ticker, date).
  3. Sample 1/3 of (ticker, date) pairs that have news.
  4. From that sample, draw another 1/3 for GPT evaluation.
  5. GPT reads the article text and decides which system's score is more
     appropriate for financial sentiment analysis.
  6. Print a summary report.

Requirements:
    pip install pandas pyarrow openai python-dotenv

Set your OpenAI API key in the environment or a .env file:
    OPENAI_API_KEY=sk-...
"""

import json
import os
import random
import re

import pandas as pd
from pandas.tseries.offsets import BDay
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
NEWS_JSON   = os.path.join(BASE, "Guardian news/company news/guardian_news_2018.json")
FINBERT_PKT = os.path.join(BASE, "Guardian news/Finbert Features/news_embeddings.parquet")
TONE_PKT    = os.path.join(BASE, "Guardian news/Finbert Features/guardian_finbert_features_2018_2025.parquet")
RANDOM_SEED = 42
DIFF_TOO_LARGE_THRESHOLD = 0.35
TONE_NEUTRAL_BAND = 0.05


# ── 1. Load and aggregate 2018 Guardian text to (ticker, trading_date) ───────

def load_guardian_articles(path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        ticker, date, headlines, snippets, n_articles_guardian

    Notes:
    - For this dataset, 2018 JSON currently has null trading_date/guardian_tone.
    - We align article text to trading dates by:
        1) using trading_date when present;
        2) otherwise using publication_date;
        3) shifting weekend dates to the next business day.
    """
    with open(path) as f:
        articles = json.load(f)

    rows = []
    for art in articles:
        trading_date = art.get("trading_date")
        publication_date = art.get("publication_date")

        if trading_date:
            event_date = pd.to_datetime(trading_date)
        elif publication_date:
            event_date = pd.to_datetime(publication_date)
        else:
            continue

        # Market features are on trading days; push weekend publication dates
        # to the next business day so they can map onto daily score rows.
        if event_date.dayofweek >= 5:
            event_date = event_date + BDay(1)

        rows.append({
            "ticker":      art["ticker"],
            "date":        event_date.normalize(),
            "headline":    art.get("headline", ""),
            "body_snippet": (art.get("bodyText") or "")[:400],
        })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=["ticker", "date", "headlines", "snippets", "n_articles_guardian"])

    text_agg = (
        raw.groupby(["ticker", "date"])
        .apply(
            lambda g: pd.Series({
                "headlines": " | ".join(g["headline"].tolist()),
                "snippets":  " ... ".join(g["body_snippet"].tolist()),
                "n_articles_guardian": len(g),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    return text_agg


# ── 2. Load daily FinBERT + Guardian tone scores ──────────────────────────────

def load_daily_scores(finbert_path: str, tone_path: str) -> pd.DataFrame:
    """
    Returns columns:
      ticker, date, finbert_pos, finbert_neu, finbert_neg, finbert_label,
      mean_tone, guardian_tone_label, has_news, n_articles_finbert
    """
    finbert = pd.read_parquet(finbert_path, columns=[
        "ticker", "date", "sentiment_pos", "sentiment_neu", "sentiment_neg",
        "has_news", "n_articles",
    ])
    finbert = finbert[finbert["has_news"] == 1].copy()
    finbert["date"] = pd.to_datetime(finbert["date"])
    finbert = finbert.rename(columns={
        "sentiment_pos": "finbert_pos",
        "sentiment_neu": "finbert_neu",
        "sentiment_neg": "finbert_neg",
        "n_articles":    "n_articles_finbert",
    })

    def finbert_label(row):
        scores = {"positive": row.finbert_pos, "neutral": row.finbert_neu, "negative": row.finbert_neg}
        return max(scores, key=scores.get)

    finbert["finbert_label"] = finbert.apply(finbert_label, axis=1)

    tone = pd.read_parquet(tone_path, columns=["ticker", "date", "mean_tone"])
    tone["date"] = pd.to_datetime(tone["date"])

    merged = finbert.merge(tone, on=["ticker", "date"], how="inner")

    def tone_label(score):
        if score > TONE_NEUTRAL_BAND:
            return "positive"
        if score < -TONE_NEUTRAL_BAND:
            return "negative"
        return "neutral"

    merged["guardian_tone_label"] = merged["mean_tone"].apply(tone_label)
    return merged


# ── 3. Merge and sample ───────────────────────────────────────────────────────

def build_comparison_dataset(seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (comparison_sample, gpt_sample):
        comparison_sample  – 1/3 of (ticker, date) pairs present in BOTH datasets
        gpt_sample         – 1/3 of comparison_sample chosen for GPT evaluation
    """
    guardian_text = load_guardian_articles(NEWS_JSON)
    daily_scores = load_daily_scores(FINBERT_PKT, TONE_PKT)

    merged = daily_scores.merge(guardian_text, on=["ticker", "date"], how="inner")
    print(f"[data]  Daily score rows (2018, has_news=1): {len(daily_scores)}")
    print(f"[data]  Guardian text rows mapped to trading date: {len(guardian_text)}")
    print(f"[data]  Matched (ticker, date) rows for comparison: {len(merged)}")
    if not merged.empty:
        net = merged["finbert_pos"] - merged["finbert_neg"]
        if (net - merged["mean_tone"]).abs().max() < 1e-12:
            print("[warn]  mean_tone is identical to (finbert_pos - finbert_neg) "
                  "for all matched rows.")
            print("[warn]  This indicates both score columns come from the same "
                  "signal, so this comparison is not independent.")
    if merged.empty:
        print("[data]  No matched rows were found. Check date alignment or source files.")
        return merged, merged

    rng = random.Random(seed)
    comparison_k = max(1, len(merged) // 3)
    comparison_idx = rng.sample(range(len(merged)), k=comparison_k)
    comparison = merged.iloc[comparison_idx].reset_index(drop=True)

    gpt_k = max(1, len(comparison) // 3)
    gpt_idx = rng.sample(range(len(comparison)), k=gpt_k)
    gpt_sample = comparison.iloc[gpt_idx].reset_index(drop=True)

    print(f"[sample] Comparison sample (1/3 of matched): {len(comparison)} / {len(merged)}")
    print(f"[sample] GPT evaluation subset (1/3 of comparison): {len(gpt_sample)} / {len(comparison)}")
    return comparison, gpt_sample


# ── 4. Agreement / disagreement analysis (no GPT) ────────────────────────────

def agreement_stats(df: pd.DataFrame, diff_threshold: float = DIFF_TOO_LARGE_THRESHOLD) -> dict:
    """
    Computes label-level agreement between FinBERT and Guardian tone.
    Also reports mean absolute difference of FinBERT dominant score
    vs normalised guardian tone (mapped to [0,1]).
    """
    agree = (df["finbert_label"] == df["guardian_tone_label"]).mean()

    # Normalise guardian tone to [0,1] for rough numeric comparison
    # guardian_tone_mean typically in [-10, +10]; clip and rescale
    norm_gdelt = (df["mean_tone"].clip(-10, 10) + 10) / 20

    # FinBERT "net positivity" = pos - neg
    finbert_net = df["finbert_pos"] - df["finbert_neg"]
    # Rescale to [0,1]
    finbert_norm = (finbert_net + 1) / 2

    abs_diff = (finbert_norm - norm_gdelt).abs()
    mae = abs_diff.mean()
    large_mask = abs_diff > diff_threshold

    label_cross = pd.crosstab(df["finbert_label"], df["guardian_tone_label"])
    large_examples = (
        df.loc[large_mask, [
            "ticker", "date", "finbert_pos", "finbert_neu", "finbert_neg",
            "finbert_label", "mean_tone", "guardian_tone_label",
        ]]
        .copy()
        .head(10)
    )

    return {
        "label_agreement_pct": round(agree * 100, 1),
        "mean_abs_error_normalised": round(float(mae), 4),
        "diff_threshold": diff_threshold,
        "large_diff_count": int(large_mask.sum()),
        "large_diff_pct": round(float(large_mask.mean() * 100), 1),
        "large_diff_examples": large_examples,
        "label_crosstab": label_cross,
    }


# ── 5. GPT evaluation ─────────────────────────────────────────────────────────

GPT_MODEL = "gpt-4o-mini"   # cheap and fast; swap to gpt-4o for higher quality

SYSTEM_PROMPT = """You are a quantitative financial analyst evaluating two automated \
news-sentiment scoring systems for stock market analysis.

You will be given:
  - A news article (headline + snippet) about a specific company.
  - Score A: FinBERT (a neural model fine-tuned on financial text).
    Reports probability triplet [positive, neutral, negative] that sums to 1.
  - Score B: Guardian/GDELT tone (a lexicon-based method).
    Reports a continuous score (positive = bullish, negative = bearish).

Your task:
  1. Read the article carefully.
  2. State your own judgment of the article's financial sentiment in ONE word:
     positive / neutral / negative.
  3. Decide which scoring system is more appropriate:
     - "finbert"  if Score A better matches your judgment
     - "gdelt"    if Score B better matches your judgment
     - "tie"      if both are equally good or bad
  4. Give a ONE-sentence reason.

Respond ONLY in this exact JSON format (no markdown fences):
{"my_sentiment": "...", "better_system": "...", "reason": "..."}
"""

def build_gpt_prompt(row: pd.Series) -> str:
    return (
        f"Ticker: {row['ticker']}\n"
        f"Date:   {row['date'].date()}\n\n"
        f"Headline(s): {row['headlines']}\n\n"
        f"Article snippet: {row['snippets'][:600]}\n\n"
        f"Score A — FinBERT:\n"
        f"  positive={row['finbert_pos']:.3f}  "
        f"neutral={row['finbert_neu']:.3f}  "
        f"negative={row['finbert_neg']:.3f}  "
        f"→ label: {row['finbert_label']}\n\n"
        f"Score B — Guardian/GDELT tone:\n"
        f"  tone={row['mean_tone']:.3f}  "
        f"→ label: {row['guardian_tone_label']}"
    )


def call_gpt(client: OpenAI, row: pd.Series) -> dict:
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_gpt_prompt(row)},
            ],
            temperature=0,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if model adds them despite instructions
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("```").strip()
        return json.loads(raw)
    except Exception as e:
        return {"my_sentiment": "error", "better_system": "error", "reason": str(e)}


def run_gpt_evaluation(gpt_sample: pd.DataFrame) -> pd.DataFrame:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[gpt]  OPENAI_API_KEY not set — skipping GPT evaluation.")
        return pd.DataFrame()

    client = OpenAI(api_key=api_key)
    results = []

    for i, (_, row) in enumerate(gpt_sample.iterrows(), 1):
        print(f"[gpt]  Evaluating {i}/{len(gpt_sample)}: {row['ticker']} {row['date'].date()} ...", end=" ")
        verdict = call_gpt(client, row)
        print(verdict.get("better_system", "?"))
        results.append({
            "ticker":           row["ticker"],
            "date":             row["date"].date(),
            "finbert_label":    row["finbert_label"],
            "gdelt_label":      row["guardian_tone_label"],
            "gpt_sentiment":    verdict.get("my_sentiment", "error"),
            "better_system":    verdict.get("better_system", "error"),
            "reason":           verdict.get("reason", ""),
        })

    return pd.DataFrame(results)


# ── 6. Final report ───────────────────────────────────────────────────────────

def print_report(stats: dict, gpt_results: pd.DataFrame):
    print("\n" + "=" * 60)
    print("SENTIMENT COMPARISON REPORT  (2018, 1/3 sample)")
    print("=" * 60)

    print(f"\n[1] Label agreement (FinBERT vs Guardian/GDELT): "
          f"{stats['label_agreement_pct']}%")
    print(f"    Normalised MAE (continuous scores): "
          f"{stats['mean_abs_error_normalised']}")
    print(f"    Large-difference threshold: {stats['diff_threshold']:.2f}")
    print(f"    Rows with |FinBERT-GDELT| > threshold: {stats['large_diff_count']} "
          f"({stats['large_diff_pct']}%)")
    print("\n[2] Label cross-tabulation (rows=FinBERT, cols=Guardian):")
    print(stats["label_crosstab"].to_string())
    if not stats["large_diff_examples"].empty:
        print("\n[2b] First rows with large score difference:")
        print(stats["large_diff_examples"].to_string(index=False))

    if gpt_results.empty:
        print("\n[3] GPT evaluation: skipped (no API key)")
        return

    print(f"\n[3] GPT evaluation ({len(gpt_results)} articles):")
    votes = gpt_results["better_system"].value_counts()
    total = len(gpt_results)
    for system, count in votes.items():
        print(f"    {system:10s}: {count:3d} / {total}  ({100*count/total:.1f}%)")

    print("\n[4] Per-label GPT accuracy:")
    for system in ["finbert", "gdelt"]:
        mask = gpt_results["better_system"] == system
        if system == "finbert":
            correct = (
                gpt_results.loc[mask, "finbert_label"] ==
                gpt_results.loc[mask, "gpt_sentiment"]
            ).mean()
        else:
            correct = (
                gpt_results.loc[mask, "gdelt_label"] ==
                gpt_results.loc[mask, "gpt_sentiment"]
            ).mean()
        print(f"    When GPT prefers {system}: its label matches GPT judgment "
              f"{correct*100:.1f}% of the time")

    print("\n[5] Sample GPT verdicts:")
    for _, row in gpt_results.head(5).iterrows():
        print(f"    [{row['ticker']} {row['date']}] "
              f"FinBERT={row['finbert_label']:8s} | "
              f"GDELT={row['gdelt_label']:8s} | "
              f"GPT={row['gpt_sentiment']:8s} | "
              f"winner={row['better_system']:8s} | "
              f"{row['reason']}")

    print("\n[6] Full GPT results saved to: sentiment_comparison_gpt_results.csv")
    gpt_results.to_csv(
        os.path.join(BASE, "sentiment_comparison_gpt_results.csv"), index=False
    )
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data and building sample ...")
    comparison, gpt_sample = build_comparison_dataset()

    print("\nComputing agreement statistics on comparison sample ...")
    stats = agreement_stats(comparison)

    print("\nRunning GPT evaluation on GPT subset ...")
    gpt_results = run_gpt_evaluation(gpt_sample)

    print_report(stats, gpt_results)
