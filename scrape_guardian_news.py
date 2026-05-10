#!/usr/bin/env python3
"""
Scrape Guardian company news for a ticker list and save yearly JSON files.

Example:
  python scrape_guardian_news.py \
    --tickers-file tickers.txt \
    --output-dir data \
    --start-year 2018 \
    --end-year 2025

Requires:
  - GUARDIAN_API_KEY in environment (or pass --api-key)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests

API_URL = "https://content.guardianapis.com/search"

# Optional aliases improve relevance beyond ticker-only matching.
TICKER_ALIASES: dict[str, list[str]] = {
    "AAPL": ["Apple", "Apple Inc", "AAPL"],
    "AAL": ["American Airlines", "AAL"],
    "ABEO": ["Abeona Therapeutics", "ABEO"],
    "ABNB": ["Airbnb", "Airbnb Inc", "ABNB"],
    "ACAD": ["ACADIA Pharmaceuticals", "ACAD"],
    "ACGL": ["Arch Capital Group", "ACGL"],
    "ACHC": ["Acadia Healthcare", "ACHC"],
    "ACHV": ["Achieve Life Sciences", "ACHV"],
    "ACLS": ["Axcelis Technologies", "ACLS"],
    "ACMR": ["ACM Research", "ACMR"],
    "ACRS": ["Aclaris Therapeutics", "ACRS"],
    "ADBE": ["Adobe", "Adobe Inc", "ADBE"],
    "ADI": ["Analog Devices", "ADI"],
    "ADIL": ["Adial Pharmaceuticals", "ADIL"],
    "ADMA": ["ADMA Biologics", "ADMA"],
    "ADP": ["Automatic Data Processing", "ADP"],
    "ADSK": ["Autodesk", "ADSK"],
    "AEHR": ["Aehr Test Systems", "AEHR"],
    "AEIS": ["Advanced Energy Industries", "AEIS"],
    "AEP": ["American Electric Power", "AEP"],
    "AERO": ["AERO"],
    "CCO": ["Clear Channel Outdoor", "CCO"],
    "CCRN": ["Cross Country Healthcare", "CCRN"],
    "CCS": ["Century Communities", "CCS"],
    "TSLA": ["Tesla", "Tesla Inc", "TSLA", "Elon Musk"],
    "NFLX": ["Netflix", "NFLX"],
    "MSFT": ["Microsoft", "Microsoft Corporation", "MSFT"],
    "META": ["Meta", "Meta Platforms", "Facebook", "META"],
    "GOOGL": ["Alphabet", "Google", "GOOGL"],
    "AMZN": ["Amazon", "Amazon.com", "AMZN", "AWS"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Guardian news by ticker and write yearly JSON files."
    )
    parser.add_argument(
        "--tickers-file",
        default="tickers.txt",
        help="Path to ticker file (one symbol per line).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for guardian_news_<year>.json outputs.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="First year to scrape (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=date.today().year,
        help="Last year to scrape (inclusive).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GUARDIAN_API_KEY", "").strip(),
        help="Guardian API key. Defaults to GUARDIAN_API_KEY env var.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Guardian page size (max 200).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Pause between requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=40,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for transient request failures.",
    )
    return parser.parse_args()


def load_tickers(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")
    tickers: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        symbol = line.strip().upper()
        if symbol and not symbol.startswith("#"):
            tickers.append(symbol)
    if not tickers:
        raise ValueError("No tickers found in ticker file.")
    return tickers


def build_query(ticker: str) -> str:
    terms = TICKER_ALIASES.get(ticker, [ticker])
    clauses = [f"\"{t}\"" if " " in t else t for t in terms]
    return "(" + " OR ".join(clauses) + ")"


def request_with_retry(
    session: requests.Session,
    params: dict[str, Any],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = session.get(API_URL, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt > max_retries:
                    resp.raise_for_status()
                backoff = min(60.0, 1.5**attempt)
                print(
                    f"  transient HTTP {resp.status_code}, retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt > max_retries:
                raise
            backoff = min(60.0, 1.5**attempt)
            print(f"  request failed, retrying in {backoff:.1f}s...")
            time.sleep(backoff)


def normalize_article(
    ticker: str,
    query: str,
    year_start: str,
    year_end: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    fields = item.get("fields") or {}
    tags_in = item.get("tags") or []
    tags = [
        {
            "id": t.get("id"),
            "type": t.get("type"),
            "webTitle": t.get("webTitle"),
        }
        for t in tags_in
    ]

    headline = fields.get("headline") or item.get("webTitle")
    trail_text = fields.get("trailText")
    body_text = fields.get("bodyText")
    parts = [headline, trail_text, body_text]
    text_for_finbert = " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
    guardian_id = item.get("id", "")
    uid_seed = f"{ticker}|{guardian_id}|{item.get('webPublicationDate', '')}"
    uid = hashlib.md5(uid_seed.encode("utf-8")).hexdigest()

    return {
        "uid": uid,
        "ticker": ticker,
        "webPublicationDate": item.get("webPublicationDate"),
        "publication_date": (item.get("webPublicationDate") or "")[:10] or None,
        "trading_date": None,
        "guardian_id": guardian_id,
        "type": item.get("type"),
        "sectionId": item.get("sectionId"),
        "sectionName": item.get("sectionName"),
        "pillarId": item.get("pillarId"),
        "pillarName": item.get("pillarName"),
        "webTitle": item.get("webTitle"),
        "webUrl": item.get("webUrl"),
        "apiUrl": item.get("apiUrl"),
        "headline": headline,
        "trailText": trail_text,
        "bodyText": body_text,
        "text_for_finbert": text_for_finbert,
        "wordcount": fields.get("wordcount"),
        "guardian_tone": None,
        "tags": tags,
        "query": query,
        "from_date_window": year_start,
        "to_date_window": year_end,
        "source": "guardian",
    }


def fetch_ticker_year(
    session: requests.Session,
    ticker: str,
    year: int,
    api_key: str,
    page_size: int,
    timeout: int,
    max_retries: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    year_start = f"{year}-01-01"
    year_end = f"{year}-12-31"
    query = build_query(ticker)

    page = 1
    pages = 1
    out: list[dict[str, Any]] = []

    while page <= pages:
        params = {
            "q": query,
            "from-date": year_start,
            "to-date": year_end,
            "order-by": "oldest",
            "page-size": min(max(page_size, 1), 200),
            "page": page,
            "show-fields": "headline,trailText,bodyText,wordcount",
            "show-tags": "keyword,contributor",
            "api-key": api_key,
        }
        payload = request_with_retry(session, params, timeout, max_retries)
        response = payload.get("response") or {}
        pages = int(response.get("pages") or 1)
        results = response.get("results") or []

        for item in results:
            out.append(normalize_article(ticker, query, year_start, year_end, item))

        print(f"    page {page}/{pages} -> +{len(results)}")
        page += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return out


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("ticker") or "", row.get("guardian_id") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError(
            "Guardian API key is required. Set GUARDIAN_API_KEY or pass --api-key."
        )
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year.")

    tickers = load_tickers(args.tickers_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(tickers)} tickers from {args.tickers_file}")
    print(f"Writing outputs to {output_dir.resolve()}")

    with requests.Session() as session:
        for year in range(args.start_year, args.end_year + 1):
            print(f"\n===== Year {year} =====")
            all_rows: list[dict[str, Any]] = []
            for i, ticker in enumerate(tickers, start=1):
                print(f"[{i:02d}/{len(tickers):02d}] {ticker}")
                rows = fetch_ticker_year(
                    session=session,
                    ticker=ticker,
                    year=year,
                    api_key=args.api_key,
                    page_size=args.page_size,
                    timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    sleep_seconds=args.sleep_seconds,
                )
                all_rows.extend(rows)
                print(f"    total rows this ticker/year: {len(rows)}")

            deduped = dedupe_rows(all_rows)
            out_path = output_dir / f"guardian_news_{year}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(deduped, f, ensure_ascii=False, indent=2)
            print(
                f"Saved {out_path.name}: {len(deduped)} rows "
                f"(raw={len(all_rows)}, deduped={len(deduped)})"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
