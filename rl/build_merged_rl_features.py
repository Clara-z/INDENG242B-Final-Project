#!/usr/bin/env python3
"""
Merge asset features with PCA-compressed news features for RL training.

This expects the news parquet to already contain FinBERT-derived PCA columns,
typically news_pca_00 ... news_pca_31 from the Colab FinBERT pipeline.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge asset features and PCA news features.")
    parser.add_argument(
        "--asset-path",
        type=Path,
        default=Path("data/processed/asset_features.parquet"),
        help="Asset/fundamental features parquet.",
    )
    parser.add_argument(
        "--news-path",
        type=Path,
        default=Path("data/processed/news_embeddings.parquet"),
        help="News parquet with PCA columns (news_pca_XX) and optional sentiment columns.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/rl_features_with_news_pca32.parquet"),
        help="Output merged parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.asset_path.exists():
        raise FileNotFoundError(f"Missing asset parquet: {args.asset_path}")
    if not args.news_path.exists():
        raise FileNotFoundError(f"Missing news parquet: {args.news_path}")

    asset = pd.read_parquet(args.asset_path)
    news = pd.read_parquet(args.news_path)

    for name, frame in [("asset", asset), ("news", news)]:
        for col in ("ticker", "date"):
            if col not in frame.columns:
                raise ValueError(f"{name} parquet missing required column: {col}")

    pca_cols = sorted(
        [c for c in news.columns if re.fullmatch(r"news_pca_\d{2}", c)],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not pca_cols:
        raise ValueError(
            "No PCA columns found in news parquet. Expected names like news_pca_00 ... news_pca_31."
        )

    keep_optional = [
        c
        for c in ["has_news", "n_articles", "sentiment_pos", "sentiment_neu", "sentiment_neg", "sentiment_net"]
        if c in news.columns
    ]
    keep_news_cols = ["ticker", "date"] + keep_optional + pca_cols

    asset = asset.copy()
    news = news[keep_news_cols].copy()
    asset["date"] = pd.to_datetime(asset["date"]).dt.normalize()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()

    merged = asset.merge(news, on=["ticker", "date"], how="left")
    fill_cols = keep_optional + pca_cols
    for col in fill_cols:
        merged[col] = merged[col].fillna(0.0)

    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_path, index=False)

    print(f"[save] rows={len(merged)} cols={len(merged.columns)}")
    print(f"[save] wrote {args.output_path}")


if __name__ == "__main__":
    main()
