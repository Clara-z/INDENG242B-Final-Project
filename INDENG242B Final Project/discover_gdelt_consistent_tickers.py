#!/usr/bin/env python3
"""
Discover tickers with consistent GDELT news coverage.

This script queries GDELT DOC API directly and scores each ticker by
coverage consistency (weekly or monthly).

Key features:
- Configurable frequency: weekly or monthly
- Configurable thresholds (min articles/period, pass ratio, total articles)
- Universe source:
  1) US listed symbols from NasdaqTrader files (broad coverage)
  2) custom CSV/JSON file (your own candidates)
- Outputs:
  - full summary CSV for all attempted tickers
  - keep-list CSV for tickers passing your consistency rule

Examples:
  # Broad scan using US-listed symbols, weekly consistency target
  python discover_gdelt_consistent_tickers.py \
    --universe-source us_listed \
    --start-date 2020-01-01 \
    --end-date 2025-12-31 \
    --freq W \
    --min-articles-per-period 2 \
    --min-pass-ratio 0.60 \
    --min-total-articles 100 \
    --max-tickers 1500

  # Use your own ticker file
  python discover_gdelt_consistent_tickers.py \
    --universe-source file \
    --universe-file candidates.csv \
    --freq M \
    --min-articles-per-period 8 \
    --min-pass-ratio 0.60
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass
class Thresholds:
    min_articles_per_period: float
    min_pass_ratio: float
    min_total_articles: int
    min_active_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find tickers with consistent GDELT coverage.")
    parser.add_argument(
        "--api-client",
        choices=["gdeltdoc", "raw"],
        default="gdeltdoc",
        help=(
            "Which client to use for timeline calls. "
            "'gdeltdoc' uses the python package; 'raw' uses direct HTTP requests."
        ),
    )
    parser.add_argument(
        "--universe-source",
        choices=["us_listed", "file"],
        default="us_listed",
        help="Ticker universe source.",
    )
    parser.add_argument(
        "--universe-file",
        type=str,
        default=None,
        help="Path to custom universe file (CSV or JSON). Required when --universe-source=file.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--freq",
        choices=["W", "M"],
        default="W",
        help="Coverage consistency period: W=weekly, M=monthly.",
    )
    parser.add_argument(
        "--min-articles-per-period",
        type=float,
        default=2.0,
        help="Minimum articles in a period to count as passing.",
    )
    parser.add_argument(
        "--min-pass-ratio",
        type=float,
        default=0.60,
        help="Minimum fraction of periods meeting min-articles-per-period.",
    )
    parser.add_argument(
        "--min-total-articles",
        type=int,
        default=100,
        help="Minimum total articles across the full date range.",
    )
    parser.add_argument(
        "--min-active-ratio",
        type=float,
        default=0.40,
        help="Minimum fraction of periods with at least one article.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Optional cap for number of tickers (0 means all).",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=0,
        help=(
            "Query range chunk size in days per API call. "
            "Use 0 for single-call-per-ticker over full date range (default)."
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Extra delay between API calls (on top of min-request-interval).",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=5.2,
        help="Minimum seconds between any two GDELT requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per API call.",
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=120.0,
        help="Maximum backoff sleep when rate-limited (HTTP 429).",
    )
    parser.add_argument(
        "--query-template",
        type=str,
        default='"{company_name}" AND sourcelang:english',
        help=(
            "Template for GDELT query. Available placeholders: {ticker}, {company_name}. "
            "Example: '(\"{company_name}\" OR \"{ticker}\") AND (stock OR earnings)'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output CSV files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppress per-ticker live logs.",
    )
    parser.add_argument(
        "--show-period-preview",
        type=int,
        default=4,
        help="How many most-recent periods to print per ticker in live logs.",
    )
    parser.add_argument(
        "--debug-tickers",
        type=str,
        default="",
        help="Comma-separated tickers for raw timeline debug print (e.g., AAPL,MSFT).",
    )
    parser.add_argument(
        "--debug-raw-limit",
        type=int,
        default=5,
        help="How many raw rows to print per debug ticker.",
    )
    return parser.parse_args()


def to_gdelt_dt(d: date, end_of_day: bool) -> str:
    suffix = "235959" if end_of_day else "000000"
    return d.strftime("%Y%m%d") + suffix


def daterange_chunks(start_d: date, end_d: date, chunk_days: int) -> Iterable[tuple[date, date]]:
    cur = start_d
    while cur <= end_d:
        nxt = min(cur + timedelta(days=chunk_days - 1), end_d)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def request_with_retries(
    session: requests.Session,
    params: dict[str, str],
    max_retries: int,
    sleep_seconds: float,
    max_backoff_seconds: float,
    min_request_interval: float,
    rate_state: dict[str, float],
    quiet: bool,
) -> dict:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            now = time.time()
            elapsed = now - rate_state.get("last_request_ts", 0.0)
            if elapsed < min_request_interval:
                wait = min_request_interval - elapsed
                if wait > 0:
                    time.sleep(wait)
            resp = session.get(GDELT_DOC_API, params=params, timeout=60)
            rate_state["last_request_ts"] = time.time()

            if resp.status_code == 429:
                retry_after_raw = resp.headers.get("Retry-After", "").strip()
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw else 0.0
                except ValueError:
                    retry_after = 0.0

                backoff = max(retry_after, sleep_seconds * (2 ** attempt) * 8, 2.0)
                backoff = min(backoff, max_backoff_seconds)
                if not quiet:
                    print(
                        f"[rate-limit] 429 from GDELT; sleeping {backoff:.1f}s "
                        f"(attempt {attempt+1}/{max_retries+1})",
                        flush=True,
                    )
                time.sleep(backoff)
                if attempt == max_retries:
                    break
                continue

            if 500 <= resp.status_code < 600:
                backoff = min(max(sleep_seconds * (2 ** attempt), 1.0), max_backoff_seconds)
                if not quiet:
                    print(
                        f"[server] {resp.status_code} from GDELT; sleeping {backoff:.1f}s "
                        f"(attempt {attempt+1}/{max_retries+1})",
                        flush=True,
                    )
                time.sleep(backoff)
                if attempt == max_retries:
                    break
                continue

            text_lower = resp.text.lower()
            if "please limit requests to one every 5 seconds" in text_lower:
                backoff = min(max(5.5, sleep_seconds * (2 ** attempt) * 8), max_backoff_seconds)
                if not quiet:
                    print(
                        f"[rate-limit] GDELT text throttle message; sleeping {backoff:.1f}s "
                        f"(attempt {attempt+1}/{max_retries+1})",
                        flush=True,
                    )
                time.sleep(backoff)
                if attempt == max_retries:
                    break
                continue

            if "the specified phrase is too short" in text_lower:
                raise RuntimeError(
                    "GDELT rejected query as too short. "
                    "Try a query-template emphasizing company_name and avoid short ticker-only phrases."
                )

            resp.raise_for_status()
            data = resp.json()
            if "timeline" not in data:
                raise ValueError(f"Unexpected response keys: {list(data.keys())}")
            return data
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == max_retries:
                break
            backoff = sleep_seconds * (2 ** attempt)
            time.sleep(max(0.2, backoff))
    raise RuntimeError(f"GDELT request failed after retries: {last_err}") from last_err


def parse_timeline_points(payload: dict) -> list[tuple[pd.Timestamp, float]]:
    timeline = payload.get("timeline", [])
    if not timeline:
        return []

    # Some modes return [{"series": "...", "data": [...]}], others return direct points.
    if isinstance(timeline, list) and timeline and isinstance(timeline[0], dict) and "data" in timeline[0]:
        points = timeline[0]["data"]
    else:
        points = timeline

    parsed: list[tuple[pd.Timestamp, float]] = []
    for p in points:
        dt_raw = str(p.get("date", ""))[:8]
        if len(dt_raw) != 8 or not dt_raw.isdigit():
            continue
        dt = pd.to_datetime(dt_raw, format="%Y%m%d", errors="coerce")
        if pd.isna(dt):
            continue
        value = float(p.get("value", 0.0))
        parsed.append((dt.normalize(), value))
    return parsed


def fetch_daily_counts_for_ticker(
    session: requests.Session,
    ticker: str,
    company_name: str,
    start_d: date,
    end_d: date,
    chunk_days: int,
    sleep_seconds: float,
    min_request_interval: float,
    max_retries: int,
    max_backoff_seconds: float,
    rate_state: dict[str, float],
    quiet: bool,
    query_template: str,
) -> pd.Series:
    daily_counts: dict[pd.Timestamp, float] = {}
    query = query_template.format(ticker=ticker, company_name=company_name)

    if chunk_days <= 0:
        chunks = [(start_d, end_d)]
    else:
        chunks = list(daterange_chunks(start_d, end_d, chunk_days))

    for chunk_start, chunk_end in chunks:
        params = {
            "query": query,
            "mode": "TimelineVolRaw",
            "format": "json",
            "startdatetime": to_gdelt_dt(chunk_start, end_of_day=False),
            "enddatetime": to_gdelt_dt(chunk_end, end_of_day=True),
        }
        payload = request_with_retries(
            session,
            params,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
            max_backoff_seconds=max_backoff_seconds,
            min_request_interval=min_request_interval,
            rate_state=rate_state,
            quiet=quiet,
        )
        points = parse_timeline_points(payload)
        for d, value in points:
            daily_counts[d] = daily_counts.get(d, 0.0) + value
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    idx = pd.date_range(pd.to_datetime(start_d), pd.to_datetime(end_d), freq="D")
    s = pd.Series(0.0, index=idx)
    for d, v in daily_counts.items():
        if d in s.index:
            s.loc[d] = s.loc[d] + v
    return s


def timeline_df_to_daily_series(
    timeline_df: pd.DataFrame,
    start_d: date,
    end_d: date,
) -> pd.Series:
    idx = pd.date_range(pd.to_datetime(start_d), pd.to_datetime(end_d), freq="D")
    out = pd.Series(0.0, index=idx)
    if timeline_df is None or timeline_df.empty:
        return out

    date_col = None
    for c in ["datetime", "date", "DateTime", "timebin", "bin"]:
        if c in timeline_df.columns:
            date_col = c
            break
    if date_col is None:
        # best-effort fallback
        date_col = timeline_df.columns[0]

    value_col = None
    # Direct matches first (covers raw API and gdeltdoc timelinevolraw outputs).
    for c in [
        "value",
        "count",
        "Value",
        "Count",
        "Article Count",
        "article_count",
        "articles",
    ]:
        if c in timeline_df.columns:
            value_col = c
            break
    if value_col is None:
        # Prefer columns that look like article counts even if dtype is object.
        for c in timeline_df.columns:
            lc = c.lower()
            if c == date_col:
                continue
            if "article" in lc and "all articles" not in lc and "norm" not in lc:
                value_col = c
                break

    if value_col is None:
        # pick first numeric column that isn't date_col
        numeric_cols = [
            c
            for c in timeline_df.columns
            if c != date_col and pd.api.types.is_numeric_dtype(timeline_df[c])
        ]
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            # nothing usable
            return out

    tmp = timeline_df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    # gdeltdoc often returns timezone-aware UTC timestamps; convert to naive
    # before mapping into our naive daily index.
    if pd.api.types.is_datetime64tz_dtype(tmp[date_col]):
        tmp[date_col] = tmp[date_col].dt.tz_convert("UTC").dt.tz_localize(None)
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        return out
    tmp["d"] = tmp[date_col].dt.normalize()
    daily = tmp.groupby("d")[value_col].sum()
    for d, v in daily.items():
        if d in out.index:
            out.loc[d] = float(v)
    return out


def fetch_daily_counts_for_ticker_gdeltdoc(
    gd: Any,
    ticker: str,
    company_name: str,
    start_d: date,
    end_d: date,
    max_retries: int,
    min_request_interval: float,
    max_backoff_seconds: float,
    sleep_seconds: float,
    rate_state: dict[str, float],
    quiet: bool,
    debug_enabled: bool,
    debug_raw_limit: int,
) -> pd.Series:
    # Import lazily so raw mode works without dependency.
    from gdeltdoc import Filters

    def build_keyword_candidates(name: str, tk: str) -> list[str]:
        """Create fallback keyword variants to avoid invalid phrase errors."""
        raw = (name or "").strip()
        cleaned = re.sub(r"\s+", " ", raw).strip()

        # Remove common listing/share-class suffixes that trigger invalid phrase checks.
        cleaned = re.sub(
            r"\s*-\s*(common stock|ordinary shares?|american depositary shares?.*|class [a-z].*|units?.*|warrants?.*)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r",\s*(common stock|ordinary shares?|american depositary shares?.*|class [a-z].*|units?.*|warrants?.*)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        first_part = cleaned.split(" - ")[0].strip() if " - " in cleaned else cleaned
        first_part = first_part.split(",")[0].strip() if "," in first_part else first_part
        short_words = " ".join(first_part.split()[:4]).strip()

        candidates = [cleaned, first_part, short_words, tk]
        out: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            c = re.sub(r"\s+", " ", str(c or "")).strip()
            if not c:
                continue
            if len(c) < 3:
                continue
            # Keep phrase length moderate to avoid invalid query errors.
            if len(c) > 90:
                c = c[:90].rsplit(" ", 1)[0].strip() or c[:90]
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    keyword_candidates = build_keyword_candidates(company_name, ticker)
    if not keyword_candidates:
        keyword_candidates = [ticker]
    last_exc: Exception | None = None
    invalid_phrase_hit = False

    for keyword in keyword_candidates:
        if not quiet:
            print(f"[query] {ticker} using keyword: {keyword}", flush=True)

        for attempt in range(max_retries + 1):
            try:
                now = time.time()
                elapsed = now - rate_state.get("last_request_ts", 0.0)
                if elapsed < min_request_interval:
                    wait = min_request_interval - elapsed
                    if wait > 0:
                        time.sleep(wait)

                filters = Filters(
                    keyword=keyword,
                    start_date=start_d.strftime("%Y-%m-%d"),
                    end_date=end_d.strftime("%Y-%m-%d"),
                    language="english",
                )
                timeline_df = gd.timeline_search("timelinevolraw", filters)
                if debug_enabled:
                    print(
                        f"[debug] {ticker} raw timeline columns: {list(timeline_df.columns)}",
                        flush=True,
                    )
                    if timeline_df.empty:
                        print(f"[debug] {ticker} raw timeline is EMPTY", flush=True)
                    else:
                        print(
                            f"[debug] {ticker} raw head({debug_raw_limit}):\n"
                            f"{timeline_df.head(debug_raw_limit).to_string(index=False)}",
                            flush=True,
                        )
                rate_state["last_request_ts"] = time.time()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                daily = timeline_df_to_daily_series(timeline_df, start_d, end_d)
                if debug_enabled:
                    print(
                        f"[debug] {ticker} daily nonzero days={(daily > 0).sum()} "
                        f"total={float(daily.sum()):.2f}",
                        flush=True,
                    )
                return daily
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc).lower()
                invalid_query = (
                    "query was not valid" in msg
                    or "too short or too long" in msg
                    or "specified phrase is too short" in msg
                )
                if invalid_query:
                    invalid_phrase_hit = True
                    if not quiet:
                        print(
                            f"[query-fallback] invalid phrase for {ticker}: '{keyword}'. "
                            "Trying next keyword variant.",
                            flush=True,
                        )
                    # No point retrying the same invalid query string.
                    break

                if "429" in msg or "too many requests" in msg or "please limit requests" in msg:
                    backoff = min(max(min_request_interval * (2 ** attempt), 2.0), max_backoff_seconds)
                    if not quiet:
                        print(
                            f"[rate-limit] gdeltdoc throttle; sleeping {backoff:.1f}s "
                            f"(attempt {attempt+1}/{max_retries+1})",
                            flush=True,
                        )
                    time.sleep(backoff)
                    if attempt == max_retries:
                        break
                    continue
                if attempt == max_retries:
                    break
                backoff = min(max(1.0, min_request_interval * (2 ** attempt)), max_backoff_seconds)
                if not quiet:
                    print(
                        f"[retry] gdeltdoc error; sleeping {backoff:.1f}s "
                        f"(attempt {attempt+1}/{max_retries+1}) | {exc}",
                        flush=True,
                    )
                time.sleep(backoff)

    if invalid_phrase_hit and last_exc is not None:
        raise RuntimeError(
            f"All keyword variants were rejected for {ticker}. Last error: {last_exc}"
        ) from last_exc
    if last_exc is not None:
        raise RuntimeError(f"GDELT gdeltdoc failed for {ticker}: {last_exc}") from last_exc
    return pd.Series(dtype=float)


def evaluate_consistency(
    daily_counts: pd.Series,
    freq: str,
    th: Thresholds,
) -> dict[str, float | int | bool]:
    if freq == "W":
        period = daily_counts.resample("W-MON").sum()
    else:
        period = daily_counts.resample("MS").sum()

    periods_total = int(len(period))
    total_articles = float(period.sum())
    periods_active = int((period > 0).sum())
    periods_pass = int((period >= th.min_articles_per_period).sum())

    active_ratio = periods_active / periods_total if periods_total else 0.0
    pass_ratio = periods_pass / periods_total if periods_total else 0.0
    avg_per_period = float(period.mean()) if periods_total else 0.0
    median_per_period = float(period.median()) if periods_total else 0.0
    p25 = float(period.quantile(0.25)) if periods_total else 0.0

    keep = (
        total_articles >= th.min_total_articles
        and pass_ratio >= th.min_pass_ratio
        and active_ratio >= th.min_active_ratio
    )

    return {
        "periods_total": periods_total,
        "total_articles": round(total_articles, 2),
        "periods_active": periods_active,
        "periods_pass": periods_pass,
        "active_ratio": round(active_ratio, 4),
        "pass_ratio": round(pass_ratio, 4),
        "avg_articles_per_period": round(avg_per_period, 3),
        "median_articles_per_period": round(median_per_period, 3),
        "p25_articles_per_period": round(p25, 3),
        "keep": bool(keep),
    }


def build_period_series(daily_counts: pd.Series, freq: str) -> pd.Series:
    if freq == "W":
        return daily_counts.resample("W-MON").sum()
    return daily_counts.resample("MS").sum()


def format_period_preview(period: pd.Series, n: int) -> str:
    if period.empty or n <= 0:
        return "-"
    tail = period.tail(n)
    items = [f"{idx.strftime('%Y-%m')}:{int(val)}" for idx, val in tail.items()]
    return ", ".join(items)


def load_us_listed_universe() -> pd.DataFrame:
    # Nasdaq listed
    nasdaq_txt = requests.get(NASDAQ_LISTED_URL, timeout=60).text
    nasdaq = pd.read_csv(StringIO(nasdaq_txt), sep="|")
    nasdaq = nasdaq[nasdaq["Symbol"] != "File Creation Time"].copy()
    nasdaq = nasdaq.rename(columns={"Symbol": "ticker", "Security Name": "company_name"})
    nasdaq["exchange"] = "NASDAQ"
    nasdaq = nasdaq[["ticker", "company_name", "exchange"]]

    # Other listed: NYSE/NYSE American/etc.
    other_txt = requests.get(OTHER_LISTED_URL, timeout=60).text
    other = pd.read_csv(StringIO(other_txt), sep="|")
    other = other[other["ACT Symbol"] != "File Creation Time"].copy()
    other = other.rename(columns={"ACT Symbol": "ticker", "Security Name": "company_name", "Exchange": "exchange"})
    other = other[["ticker", "company_name", "exchange"]]

    universe = pd.concat([nasdaq, other], ignore_index=True)
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    universe["company_name"] = universe["company_name"].fillna("").astype(str).str.strip()

    # Filter common bad symbols for this use case.
    universe = universe[~universe["ticker"].str.contains(r"[\^\$]", regex=True)]
    universe = universe[~universe["ticker"].str.contains(r"\.", regex=False)]
    universe = universe[universe["ticker"].str.len().between(1, 5)]
    universe = universe.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return universe


def load_file_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        df = pd.DataFrame(obj)
    elif path.suffix.lower() == ".txt":
        # Supported TXT formats:
        # 1) ticker only:
        #      AAPL
        #      MSFT
        # 2) tab or comma separated:
        #      AAPL\tApple Inc\tNASDAQ
        #      MSFT,Microsoft Corp,NASDAQ
        rows: list[dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in re.split(r"[\t,]", line) if p.strip() != ""]
                if not parts:
                    continue
                ticker = parts[0]
                company_name = parts[1] if len(parts) >= 2 else ticker
                exchange = parts[2] if len(parts) >= 3 else "UNKNOWN"
                rows.append(
                    {
                        "ticker": ticker,
                        "company_name": company_name,
                        "exchange": exchange,
                    }
                )
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Universe file must be CSV, JSON, or TXT.")

    if "ticker" not in df.columns:
        raise ValueError("Universe file must contain a 'ticker' column.")
    if "company_name" not in df.columns:
        df["company_name"] = df["ticker"]

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    if "exchange" not in df.columns:
        df["exchange"] = "UNKNOWN"
    return df[["ticker", "company_name", "exchange"]].drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def append_csv_row(path: Path, row: dict, columns: list[str]) -> None:
    """Append one row to CSV, creating file with header if needed."""
    exists = path.exists()
    row_df = pd.DataFrame([{c: row.get(c, "") for c in columns}], columns=columns)
    row_df.to_csv(path, mode="a", index=False, header=not exists)


def main() -> None:
    args = parse_args()
    start_d = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end_d < start_d:
        raise ValueError("end-date must be >= start-date")

    th = Thresholds(
        min_articles_per_period=args.min_articles_per_period,
        min_pass_ratio=args.min_pass_ratio,
        min_total_articles=args.min_total_articles,
        min_active_ratio=args.min_active_ratio,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.universe_source == "us_listed":
        universe = load_us_listed_universe()
    else:
        if not args.universe_file:
            raise ValueError("--universe-file is required when --universe-source=file")
        universe = load_file_universe(Path(args.universe_file))

    if args.max_tickers > 0:
        print(
            "[note] --max-tickers is deprecated and ignored; "
            "the script now scans the full universe unless you stop it.",
            flush=True,
        )

    tag = (
        f"{args.freq}_"
        f"{start_d.strftime('%Y%m%d')}_{end_d.strftime('%Y%m%d')}_"
        f"minp{args.min_articles_per_period}_mpr{args.min_pass_ratio}_"
        f"mint{args.min_total_articles}"
    )
    summary_path = out_dir / f"gdelt_ticker_consistency_summary_{tag}.csv"
    keep_path = out_dir / f"gdelt_ticker_keep_list_{tag}.csv"
    keep_tickers_path = out_dir / f"gdelt_ticker_keep_tickers_{tag}.csv"

    # Fresh keep file for this run (streaming writes below).
    if keep_path.exists():
        keep_path.unlink()

    print(f"Universe size: {len(universe)}")
    print(f"API client: {args.api_client}")
    print(f"Date range: {start_d} -> {end_d}")
    print(
        "Request mode: "
        f"{'single-call per ticker' if args.chunk_days <= 0 else f'chunked ({args.chunk_days} days per call)'}"
    )
    print(f"Min request interval: {args.min_request_interval:.2f}s")
    print(
        "Thresholds: "
        f"min_articles_per_period={th.min_articles_per_period}, "
        f"min_pass_ratio={th.min_pass_ratio}, "
        f"min_total_articles={th.min_total_articles}, "
        f"min_active_ratio={th.min_active_ratio}"
    )
    print(
        "Live logging: "
        f"{'off' if args.quiet else 'on'} "
        f"(preview periods={args.show_period_preview})"
    )

    rows: list[dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "gdelt-consistency-scanner/1.0"})
    gd = None
    if args.api_client == "gdeltdoc":
        try:
            from gdeltdoc import GdeltDoc
        except ImportError as exc:
            raise RuntimeError(
                "gdeltdoc is not installed. Run: pip install gdeltdoc "
                "or set --api-client raw"
            ) from exc
        gd = GdeltDoc()
    rate_state = {"last_request_ts": 0.0}
    debug_set = {t.strip().upper() for t in args.debug_tickers.split(",") if t.strip()}
    keep_columns = [
        "ticker",
        "company_name",
        "exchange",
        "periods_total",
        "total_articles",
        "periods_active",
        "periods_pass",
        "active_ratio",
        "pass_ratio",
        "avg_articles_per_period",
        "median_articles_per_period",
        "p25_articles_per_period",
        "avg_articles_per_month",
        "median_articles_per_month",
        "keep",
    ]

    total = len(universe)
    for i, rec in universe.iterrows():
        ticker = rec["ticker"]
        company_name = rec["company_name"] if rec["company_name"] else ticker
        if not args.quiet:
            print(f"[{i+1}/{total}] Processing {ticker} ...", flush=True)
        try:
            if args.api_client == "gdeltdoc":
                debug_enabled = ticker.upper() in debug_set
                daily = fetch_daily_counts_for_ticker_gdeltdoc(
                    gd=gd,
                    ticker=ticker,
                    company_name=company_name,
                    start_d=start_d,
                    end_d=end_d,
                    max_retries=args.max_retries,
                    min_request_interval=args.min_request_interval,
                    max_backoff_seconds=args.max_backoff_seconds,
                    sleep_seconds=args.sleep_seconds,
                    rate_state=rate_state,
                    quiet=args.quiet,
                    debug_enabled=debug_enabled,
                    debug_raw_limit=args.debug_raw_limit,
                )
            else:
                daily = fetch_daily_counts_for_ticker(
                    session=session,
                    ticker=ticker,
                    company_name=company_name,
                    start_d=start_d,
                    end_d=end_d,
                    chunk_days=args.chunk_days,
                    sleep_seconds=args.sleep_seconds,
                    min_request_interval=args.min_request_interval,
                    max_retries=args.max_retries,
                    max_backoff_seconds=args.max_backoff_seconds,
                    rate_state=rate_state,
                    quiet=args.quiet,
                    query_template=args.query_template,
                )
            period = build_period_series(daily, freq=args.freq)
            monthly = daily.resample("MS").sum()
            stats = evaluate_consistency(daily, freq=args.freq, th=th)
            row = {
                "ticker": ticker,
                "company_name": company_name,
                "exchange": rec.get("exchange", "UNKNOWN"),
                **stats,
                "avg_articles_per_month": round(float(monthly.mean()), 3) if len(monthly) else 0.0,
                "median_articles_per_month": round(float(monthly.median()), 3) if len(monthly) else 0.0,
            }
            rows.append(row)
            if row["keep"]:
                append_csv_row(keep_path, row, keep_columns)
            if not args.quiet:
                keep_label = "KEEP" if stats["keep"] else "DROP"
                period_unit = "week" if args.freq == "W" else "month"
                preview = format_period_preview(period, args.show_period_preview)
                print(
                    f"[{i+1}/{total}] {ticker} -> {keep_label} | "
                    f"total={stats['total_articles']:.0f} | "
                    f"pass_ratio={stats['pass_ratio']:.2f} | "
                    f"active_ratio={stats['active_ratio']:.2f} | "
                    f"avg/{period_unit}={stats['avg_articles_per_period']:.2f} | "
                    f"avg/month={row['avg_articles_per_month']:.2f} | "
                    f"recent={preview}",
                    flush=True,
                )
            elif (i + 1) % 50 == 0 or i == total - 1:
                keep_count = int(sum(r["keep"] for r in rows))
                print(f"[{i+1}/{total}] processed; keep={keep_count}")
        except Exception as exc:  # noqa: BLE001
            if not args.quiet:
                print(f"[{i+1}/{total}] {ticker} -> ERROR: {exc}", flush=True)
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "exchange": rec.get("exchange", "UNKNOWN"),
                    "periods_total": math.nan,
                    "total_articles": math.nan,
                    "periods_active": math.nan,
                    "periods_pass": math.nan,
                    "active_ratio": math.nan,
                    "pass_ratio": math.nan,
                    "avg_articles_per_period": math.nan,
                    "median_articles_per_period": math.nan,
                    "p25_articles_per_period": math.nan,
                    "avg_articles_per_month": math.nan,
                    "median_articles_per_month": math.nan,
                    "keep": False,
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)
    result = result.sort_values(["keep", "pass_ratio", "total_articles"], ascending=[False, False, False])

    result.to_csv(summary_path, index=False)
    result[result["keep"] == True][["ticker", "company_name", "exchange"]].to_csv(keep_tickers_path, index=False)  # noqa: E712

    print("\nDone.")
    print(f"Summary:  {summary_path}")
    print(f"Keep list (streamed, full stats): {keep_path}")
    print(f"Keep tickers (compact): {keep_tickers_path}")
    print(f"Kept tickers: {(result['keep'] == True).sum()} / {len(result)}")  # noqa: E712


if __name__ == "__main__":
    main()

