#!/usr/bin/env python3
"""
Export broad US-listed ticker universe to TXT for discover_gdelt_consistent_tickers.py.

Output format (tab-separated):
    ticker<TAB>company_name<TAB>exchange

Example:
    python "INDENG242B Final Project/export_us_ticker_universe_txt.py" \
      --output "INDENG242B Final Project/us_listed_universe.txt"
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export US-listed ticker universe to TXT.")
    parser.add_argument(
        "--output",
        type=str,
        default="us_listed_universe.txt",
        help="Output TXT path.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Optional cap for number of tickers (0 means all).",
    )
    return parser.parse_args()


def load_us_listed_universe() -> pd.DataFrame:
    nasdaq_txt = requests.get(NASDAQ_LISTED_URL, timeout=60).text
    nasdaq = pd.read_csv(StringIO(nasdaq_txt), sep="|")
    nasdaq = nasdaq[nasdaq["Symbol"] != "File Creation Time"].copy()
    nasdaq = nasdaq.rename(columns={"Symbol": "ticker", "Security Name": "company_name"})
    nasdaq["exchange"] = "NASDAQ"
    nasdaq = nasdaq[["ticker", "company_name", "exchange"]]

    other_txt = requests.get(OTHER_LISTED_URL, timeout=60).text
    other = pd.read_csv(StringIO(other_txt), sep="|")
    other = other[other["ACT Symbol"] != "File Creation Time"].copy()
    other = other.rename(columns={"ACT Symbol": "ticker", "Security Name": "company_name", "Exchange": "exchange"})
    other = other[["ticker", "company_name", "exchange"]]

    universe = pd.concat([nasdaq, other], ignore_index=True)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["company_name"] = universe["company_name"].fillna("").astype(str).str.strip()
    universe["exchange"] = universe["exchange"].fillna("UNKNOWN").astype(str).str.strip()

    # Keep common-trading-like symbols; filter noisy instrument formats.
    universe = universe[~universe["ticker"].str.contains(r"[\^\$]", regex=True)]
    universe = universe[~universe["ticker"].str.contains(r"\.", regex=False)]
    universe = universe[universe["ticker"].str.len().between(1, 5)]
    universe = universe.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return universe


def main() -> None:
    args = parse_args()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    universe = load_us_listed_universe()
    if args.max_tickers > 0:
        universe = universe.head(args.max_tickers).copy()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ticker\\tcompany_name\\texchange\n")
        for _, row in universe.iterrows():
            f.write(f"{row['ticker']}\t{row['company_name']}\t{row['exchange']}\n")

    print(f"Saved {len(universe)} tickers to: {out_path}")


if __name__ == "__main__":
    main()

