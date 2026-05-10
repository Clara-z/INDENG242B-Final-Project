#!/usr/bin/env python3
"""
Scrape GDELT news for a fixed ticker list and export one JSON file per year.

No API key is required for GDELT DOC API.

Example:
  python scrape_guardian_news.py \
    --tickers-file tickers.txt \
    --start-year 2018 \
    --end-year 2026 \
    --output-dir data
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape GDELT news for each ticker and write yearly JSON files."
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        default=Path("tickers.txt"),
        help="Path to newline-separated ticker file.",
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
        default=2026,
        help="Last year to scrape (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for yearly output JSON files.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Optional cap for number of tickers to process (0 means all).",
    )
    parser.add_argument(
        "--query-template",
        type=str,
        default="{ticker} AND sourcelang:english",
        help=(
            "GDELT query template. Available placeholder: {ticker}. "
            "Example: '({ticker} OR Apple) AND (earnings OR stock) AND sourcelang:english'"
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=250,
        help="Max records per GDELT ArtList query (usually <= 250).",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=5.2,
        help="Minimum seconds between any two GDELT requests.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Extra sleep after each request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retry attempts per request for transient errors.",
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=float,
        default=120.0,
        help="Maximum retry backoff seconds after throttling/server errors.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="HTTP timeout (seconds) for each request.",
    )
    parser.add_argument(
        "--min-split-seconds",
        type=int,
        default=3600,
        help=(
            "If query returns max-records, split time window recursively until this floor "
            "to reduce truncation."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed request/retry/splitting debug logs.",
    )
    return parser.parse_args()


def load_tickers(path: Path, max_tickers: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        sym = line.strip().upper()
        if not sym or sym.startswith("#"):
            continue
        tickers.append(sym)
    if not tickers:
        raise ValueError(f"No tickers found in {path}")
    if max_tickers > 0:
        return tickers[:max_tickers]
    return tickers


def to_gdelt_dt(ts: datetime) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug] {message}", flush=True)


def request_gdelt_json(
    params: dict[str, Any],
    timeout: float,
    max_retries: int,
    max_backoff_seconds: float,
    min_request_interval: float,
    sleep_seconds: float,
    rate_state: dict[str, float],
    debug: bool,
) -> dict[str, Any]:
    last_error: Exception | None = None
    query_str = urlencode(params)
    url = f"{GDELT_DOC_API}?{query_str}"

    for attempt in range(max_retries + 1):
        try:
            elapsed = time.time() - rate_state.get("last_request_ts", 0.0)
            if elapsed < min_request_interval:
                wait = min_request_interval - elapsed
                debug_log(debug, f"rate guard sleeping {wait:.2f}s before next request")
                time.sleep(wait)

            debug_log(
                debug,
                (
                    f"request attempt {attempt + 1}/{max_retries + 1} "
                    f"window={params.get('startdatetime')}..{params.get('enddatetime')} "
                    f"maxrecords={params.get('maxrecords')}"
                ),
            )

            req = Request(url, headers={"User-Agent": "gdelt-news-scraper/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            rate_state["last_request_ts"] = time.time()

            body_stripped = body.strip()
            body_lower = body_stripped.lower()
            if "please limit requests to one every 5 seconds" in body_lower:
                backoff = max(5.5, sleep_seconds * (2**attempt))
                debug_log(debug, f"text throttle response; sleeping {backoff:.1f}s then retrying")
                time.sleep(backoff)
                if attempt < max_retries:
                    continue
                raise RuntimeError("GDELT returned text throttle message repeatedly.")

            if not body_stripped:
                raise RuntimeError("GDELT returned empty response body.")
            if not body_stripped.startswith("{"):
                snippet = body_stripped[:240]
                raise RuntimeError(f"GDELT returned non-JSON response: {snippet}")

            payload = json.loads(body_stripped)
            if sleep_seconds > 0:
                debug_log(debug, f"post-request sleep {sleep_seconds:.2f}s")
                time.sleep(sleep_seconds)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected GDELT payload type: {type(payload)}")
            debug_log(
                debug,
                f"response ok with {len(payload.get('articles', [])) if isinstance(payload.get('articles', []), list) else 0} articles",
            )
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break

            code = getattr(exc, "code", None)
            if code == 429:
                backoff = max(10.0, min_request_interval * (2 ** (attempt + 1)))
            elif isinstance(code, int) and 500 <= code < 600:
                backoff = max(2.0, min_request_interval * (2**attempt))
            else:
                backoff = max(1.0, min_request_interval * (2**attempt))
            backoff = min(backoff, max_backoff_seconds)
            debug_log(debug, f"request failed: {exc}; retrying in {backoff:.1f}s")
            time.sleep(backoff)
    raise RuntimeError(f"GDELT request failed after retries: {last_error}") from last_error


def fetch_articles_for_time_window(
    ticker: str,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    max_records: int,
    timeout: float,
    max_retries: int,
    max_backoff_seconds: float,
    min_request_interval: float,
    sleep_seconds: float,
    min_split_seconds: int,
    rate_state: dict[str, float],
    debug: bool,
    depth: int = 0,
) -> list[dict[str, Any]]:
    indent = "  " * depth
    debug_log(
        debug,
        f"{indent}{ticker} window {to_gdelt_dt(start_dt)} -> {to_gdelt_dt(end_dt)}",
    )
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": to_gdelt_dt(start_dt),
        "enddatetime": to_gdelt_dt(end_dt),
    }
    payload = request_gdelt_json(
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        max_backoff_seconds=max_backoff_seconds,
        min_request_interval=min_request_interval,
        sleep_seconds=sleep_seconds,
        rate_state=rate_state,
        debug=debug,
    )
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        articles = []

    span_seconds = int((end_dt - start_dt).total_seconds())
    if len(articles) < max_records or span_seconds <= max(min_split_seconds, 1):
        debug_log(
            debug,
            f"{indent}returning {len(articles)} articles (span_seconds={span_seconds})",
        )
        return articles

    debug_log(
        debug,
        (
            f"{indent}hit max_records={max_records} for {ticker}; "
            f"splitting window (span_seconds={span_seconds})"
        ),
    )
    mid = start_dt + (end_dt - start_dt) / 2
    left_end = mid
    right_start = mid
    left = fetch_articles_for_time_window(
        ticker=ticker,
        query=query,
        start_dt=start_dt,
        end_dt=left_end,
        max_records=max_records,
        timeout=timeout,
        max_retries=max_retries,
        max_backoff_seconds=max_backoff_seconds,
        min_request_interval=min_request_interval,
        sleep_seconds=sleep_seconds,
        min_split_seconds=min_split_seconds,
        rate_state=rate_state,
        debug=debug,
        depth=depth + 1,
    )
    right = fetch_articles_for_time_window(
        ticker=ticker,
        query=query,
        start_dt=right_start,
        end_dt=end_dt,
        max_records=max_records,
        timeout=timeout,
        max_retries=max_retries,
        max_backoff_seconds=max_backoff_seconds,
        min_request_interval=min_request_interval,
        sleep_seconds=sleep_seconds,
        min_split_seconds=min_split_seconds,
        rate_state=rate_state,
        debug=debug,
        depth=depth + 1,
    )
    debug_log(debug, f"{indent}merged split windows => {len(left) + len(right)} articles")
    return left + right


def normalize_article(ticker: str, year: int, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "year": year,
        "url": item.get("url"),
        "url_mobile": item.get("url_mobile"),
        "title": item.get("title"),
        "seendate": item.get("seendate"),
        "socialimage": item.get("socialimage"),
        "domain": item.get("domain"),
        "language": item.get("language"),
        "sourcecountry": item.get("sourcecountry"),
    }


def fetch_ticker_year_articles(
    ticker: str,
    year: int,
    query_template: str,
    max_records: int,
    timeout: float,
    max_retries: int,
    max_backoff_seconds: float,
    min_request_interval: float,
    sleep_seconds: float,
    min_split_seconds: int,
    rate_state: dict[str, float],
    debug: bool,
) -> list[dict[str, Any]]:
    start_dt = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    query = query_template.format(ticker=ticker)
    debug_log(debug, f"{ticker} year={year} query={query}")
    raw_articles = fetch_articles_for_time_window(
        ticker=ticker,
        query=query,
        start_dt=start_dt,
        end_dt=end_dt,
        max_records=max_records,
        timeout=timeout,
        max_retries=max_retries,
        max_backoff_seconds=max_backoff_seconds,
        min_request_interval=min_request_interval,
        sleep_seconds=sleep_seconds,
        min_split_seconds=min_split_seconds,
        rate_state=rate_state,
        debug=debug,
    )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_articles:
        if not isinstance(item, dict):
            continue
        key = f"{item.get('url', '')}|{item.get('seendate', '')}|{ticker}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalize_article(ticker=ticker, year=year, item=item))
    debug_log(debug, f"{ticker} year={year} deduped {len(raw_articles)} -> {len(deduped)}")
    return deduped


def write_year_file(
    output_dir: Path,
    year: int,
    tickers: list[str],
    records: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"guardian_news_{year}.json"
    payload = {
        "year": year,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GDELT DOC API (ArtList)",
        "tickers": tickers,
        "article_count": len(records),
        "articles": records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year")

    tickers = load_tickers(args.tickers_file, max_tickers=max(args.max_tickers, 0))
    output_dir = args.output_dir.resolve()
    max_records = min(max(int(args.max_records), 1), 250)
    rate_state = {"last_request_ts": 0.0}

    print(f"Loaded {len(tickers)} tickers from {args.tickers_file}")
    print("Source: GDELT DOC API (no API key)")
    print(f"Year range: {args.start_year} -> {args.end_year}")
    print(f"Output directory: {output_dir}")
    print(f"max_records/request={max_records}, min_request_interval={args.min_request_interval:.2f}s")
    if args.debug:
        print("Debug logging: ON")

    summary: list[dict[str, Any]] = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"\n=== Year {year} ===", flush=True)
        year_records: list[dict[str, Any]] = []
        year_errors: list[dict[str, str]] = []
        for idx, ticker in enumerate(tickers, start=1):
            print(f"[{year}] {idx}/{len(tickers)} {ticker}: fetching...", flush=True)
            try:
                ticker_records = fetch_ticker_year_articles(
                    ticker=ticker,
                    year=year,
                    query_template=args.query_template,
                    max_records=max_records,
                    timeout=float(args.request_timeout),
                    max_retries=max(int(args.max_retries), 0),
                    max_backoff_seconds=max(float(args.max_backoff_seconds), 1.0),
                    min_request_interval=max(float(args.min_request_interval), 0.0),
                    sleep_seconds=max(float(args.sleep_seconds), 0.0),
                    min_split_seconds=max(int(args.min_split_seconds), 1),
                    rate_state=rate_state,
                    debug=bool(args.debug),
                )
                print(f"[{year}] {ticker}: {len(ticker_records)} articles", flush=True)
                year_records.extend(ticker_records)
            except Exception as exc:  # noqa: BLE001
                print(f"[{year}] {ticker}: ERROR - {exc}", flush=True)
                year_errors.append({"ticker": ticker, "error": str(exc)})

        out_path = write_year_file(output_dir=output_dir, year=year, tickers=tickers, records=year_records)
        summary.append(
            {
                "year": year,
                "articles": len(year_records),
                "errors": len(year_errors),
                "path": str(out_path),
            }
        )
        print(f"[{year}] wrote {len(year_records)} records -> {out_path}", flush=True)
        if year_errors:
            print(f"[{year}] warnings: {len(year_errors)} tickers failed.", flush=True)

    summary_path = output_dir / "guardian_news_scrape_summary.json"
    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GDELT DOC API (ArtList)",
        "tickers_file": str(args.tickers_file),
        "tickers_count": len(tickers),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "output_dir": str(output_dir),
        "yearly_counts": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
