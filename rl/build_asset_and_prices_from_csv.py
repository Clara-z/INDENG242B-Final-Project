#!/usr/bin/env python3
"""
Rebuild asset/prices parquet files from the core fundamentals+technical CSV.

Key behavior:
- drops `gross_margin` even if present in source
- uses `volume` (if present) to compute `volume_zscore_20d`
- recomputes common technical features from `close_adj`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build asset_features.parquet and prices.parquet from CSV.")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("data/fundamentals & technical features.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--asset-out",
        type=Path,
        default=Path("data/asset_features.parquet"),
        help="Output asset feature parquet path.",
    )
    parser.add_argument(
        "--prices-out",
        type=Path,
        default=Path("data/prices.parquet"),
        help="Output prices parquet path.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/asset_feature_build_report.json"),
        help="Output build report path.",
    )
    return parser.parse_args()


def _rsi_14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _macd_signal(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    return macd.ewm(span=9, adjust=False).mean()


def _volume_column(columns: list[str]) -> str | None:
    preferred = ["volume", "adj_volume", "share_volume", "trading_volume"]
    lower_map = {c.lower(): c for c in columns}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]
    for c in columns:
        cl = c.lower()
        if "volume" in cl and "spy_vol" not in cl:
            return c
    return None


def build_asset_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    unavailable: dict[str, str] = {}

    # Drop duplicated one-hot sector suffixes (".1", ".2", ...).
    keep_cols = [c for c in df.columns if not c.startswith("sector_") or "." not in c]
    dropped_sector_dupes = [c for c in df.columns if c.startswith("sector_") and "." in c]
    if dropped_sector_dupes:
        df = df[keep_cols].copy()
        unavailable["duplicate_sector_columns"] = (
            f"Dropped {len(dropped_sector_dupes)} duplicated sector columns with .N suffix."
        )

    required = ["date", "ticker", "close_adj"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required source columns: {missing_required}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Convert likely numeric columns where possible.
    for c in df.columns:
        if c in {"ticker", "date"}:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby("ticker", sort=False)

    df["ret_1d"] = g["close_adj"].pct_change(1)
    df["ret_5d"] = g["close_adj"].pct_change(5)
    df["ret_20d"] = g["close_adj"].pct_change(20)
    df["ret_60d"] = g["close_adj"].pct_change(60)
    df["ret_252d"] = g["close_adj"].pct_change(252)
    df["vol_20d"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["vol_60d"] = g["ret_1d"].transform(lambda s: s.rolling(60, min_periods=60).std())

    ma20 = g["close_adj"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    ma50 = g["close_adj"].transform(lambda s: s.rolling(50, min_periods=50).mean())
    df["price_to_ma20"] = df["close_adj"] / ma20
    df["price_to_ma50"] = df["close_adj"] / ma50

    df["rsi_14"] = g["close_adj"].transform(_rsi_14)
    df["macd_signal"] = g["close_adj"].transform(_macd_signal)

    vol_col = _volume_column(df.columns.tolist())
    if vol_col is not None:
        vmean = g[vol_col].transform(lambda s: s.rolling(20, min_periods=20).mean())
        vstd = g[vol_col].transform(lambda s: s.rolling(20, min_periods=20).std())
        df["volume_zscore_20d"] = (df[vol_col] - vmean) / vstd.replace(0, np.nan)
        # Keep raw volume as a first-class feature when present in source.
        if vol_col != "volume":
            df["volume"] = df[vol_col]
    else:
        df["volume_zscore_20d"] = 0.0
        unavailable["volume_zscore_20d"] = "Source CSV has no usable volume column."

    if "high" in df.columns and "low" in df.columns:
        hi_5 = g["high"].transform(lambda s: s.rolling(5, min_periods=5).max())
        lo_5 = g["low"].transform(lambda s: s.rolling(5, min_periods=5).min())
        df["high_low_range_5d"] = (hi_5 - lo_5) / df["close_adj"].replace(0, np.nan)
    else:
        df["high_low_range_5d"] = 0.0
        unavailable["high_low_range_5d"] = "Source CSV has no high/low price columns."

    # Explicitly remove gross_margin regardless of source presence/absence.
    if "gross_margin" in df.columns:
        df = df.drop(columns=["gross_margin"])
    unavailable["gross_margin"] = "Excluded by request due to high missingness."

    fundamentals = [
        "pe",
        "pb",
        "roe",
        "de_ratio",
        "profit_margin",
        "asset_turnover",
        "current_ratio",
        "days_since_filing",
    ]
    market = ["vix", "spy_ret_20d", "spy_vol_20d", "vix_change_20d", "tsy_10y_2y_spread"]
    sector_cols = [c for c in df.columns if c.startswith("sector_") and "." not in c]
    core = ["ticker", "date", "close_adj"]
    optional_volume = ["volume"] if "volume" in df.columns else []
    derived = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "ret_252d",
        "vol_20d",
        "vol_60d",
        "volume_zscore_20d",
        "price_to_ma20",
        "price_to_ma50",
        "high_low_range_5d",
        "rsi_14",
        "macd_signal",
    ]

    all_candidates = core + optional_volume + derived + fundamentals + market + sector_cols
    present = [c for c in all_candidates if c in df.columns]
    asset = df[present].copy()
    asset = asset.sort_values(["ticker", "date"]).reset_index(drop=True)

    for c in asset.columns:
        if c in {"ticker", "date"}:
            continue
        asset[c] = asset[c].astype(np.float32)

    return asset, unavailable


def build_prices(asset: pd.DataFrame) -> pd.DataFrame:
    prices = asset[["ticker", "date", "close_adj"]].copy()
    prices["is_trading_day"] = 1
    prices["is_trading_day"] = prices["is_trading_day"].astype(np.int8)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices


def main() -> None:
    args = parse_args()
    if not args.source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV: {args.source_csv}")

    df = pd.read_csv(args.source_csv)
    asset, unavailable = build_asset_features(df)
    prices = build_prices(asset)

    args.asset_out.parent.mkdir(parents=True, exist_ok=True)
    args.prices_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    asset.to_parquet(args.asset_out, index=False)
    prices.to_parquet(args.prices_out, index=False)

    report = {
        "source_csv": str(args.source_csv),
        "asset_out": str(args.asset_out),
        "prices_out": str(args.prices_out),
        "rows": int(len(asset)),
        "tickers": int(asset["ticker"].nunique()),
        "date_min": str(pd.Timestamp(asset["date"].min()).date()),
        "date_max": str(pd.Timestamp(asset["date"].max()).date()),
        "derived_columns": [
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "ret_252d",
            "vol_20d",
            "vol_60d",
            "volume_zscore_20d",
            "price_to_ma20",
            "price_to_ma50",
            "high_low_range_5d",
            "rsi_14",
            "macd_signal",
        ],
        "unavailable_columns": unavailable,
    }
    with args.report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[save] asset:  {args.asset_out} rows={len(asset)} cols={len(asset.columns)}")
    print(f"[save] prices: {args.prices_out} rows={len(prices)} cols={len(prices.columns)}")
    print(f"[save] report: {args.report_out}")


if __name__ == "__main__":
    main()
