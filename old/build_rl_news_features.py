"""
Build RL-ready features by combining:
  - core asset/fundamental features
  - FinBERT sentiment probabilities
  - PCA-compressed FinBERT embeddings (32 or 64 dims)

Usage examples:
  python build_rl_news_features.py --n-components 32
  python build_rl_news_features.py --n-components 64
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RL feature table with compressed news embeddings.")
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        choices=[32, 64],
        help="PCA dimension for FinBERT embedding compression.",
    )
    return parser.parse_args()


def entropy3(prob_array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy for 3-class probabilities."""
    p = np.clip(prob_array, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def main() -> None:
    args = parse_args()
    n_components = args.n_components

    base = Path(__file__).resolve().parent
    news_path = base / "Guardian news/Finbert Features/news_embeddings.parquet"
    asset_path = base / "processed/asset_features.parquet"
    out_path = base / f"processed/rl_features_with_news_pca{n_components}.parquet"

    print(f"[load] News features:  {news_path}")
    print(f"[load] Asset features: {asset_path}")

    news = pd.read_parquet(news_path)
    asset = pd.read_parquet(asset_path)

    emb_cols = sorted(
        [c for c in news.columns if re.fullmatch(r"emb_\d+", c)],
        key=lambda s: int(s.split("_")[1]),
    )
    if not emb_cols:
        raise ValueError("No embedding columns found (expected emb_0..emb_*).")

    needed_cols = [
        "ticker",
        "date",
        "sentiment_pos",
        "sentiment_neu",
        "sentiment_neg",
        "has_news",
        "n_articles",
    ] + emb_cols

    news = news[needed_cols].copy()
    news["date"] = pd.to_datetime(news["date"])
    asset["date"] = pd.to_datetime(asset["date"])

    # Keep interpretable sentiment features for RL state.
    probs = news[["sentiment_pos", "sentiment_neu", "sentiment_neg"]].to_numpy(dtype=np.float64)
    news["sentiment_net"] = news["sentiment_pos"] - news["sentiment_neg"]
    news["sentiment_confidence"] = probs.max(axis=1)
    news["sentiment_entropy"] = entropy3(probs)

    # Fit PCA only on news days so zero/no-news rows do not dominate variance.
    fit_mask = news["has_news"] == 1
    fit_count = int(fit_mask.sum())
    if fit_count < n_components:
        raise ValueError(
            f"Not enough news rows to fit PCA-{n_components}. "
            f"Found {fit_count} rows with has_news=1."
        )

    # Use float64 + full SVD for numerical stability in this dataset.
    x_fit = news.loc[fit_mask, emb_cols].to_numpy(dtype=np.float64)
    x_all = news[emb_cols].to_numpy(dtype=np.float64)

    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    pca.fit(x_fit)
    z_all = pca.transform(x_all)

    pca_cols = [f"news_pca_{i:02d}" for i in range(n_components)]
    z_df = pd.DataFrame(z_all, columns=pca_cols, index=news.index)
    news = pd.concat([news.drop(columns=emb_cols), z_df], axis=1)

    explained = float(pca.explained_variance_ratio_.sum())
    print(f"[pca] Fitted PCA-{n_components} on {fit_count} news rows")
    print(f"[pca] Total explained variance: {explained:.4f}")

    # Merge into asset/fundamental features table.
    merged = asset.merge(news, on=["ticker", "date"], how="left")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Outside the news-date coverage window, fill missing news features with 0.
    fill_cols = [
        "sentiment_pos",
        "sentiment_neu",
        "sentiment_neg",
        "has_news",
        "n_articles",
        "sentiment_net",
        "sentiment_confidence",
        "sentiment_entropy",
    ] + pca_cols
    for col in fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    merged.to_parquet(out_path, index=False)

    print(f"[save] Output rows: {len(merged)}")
    print(f"[save] Output cols: {len(merged.columns)}")
    print(f"[save] Wrote: {out_path}")


if __name__ == "__main__":
    main()

