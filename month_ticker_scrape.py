# month_ticker_scrape_only.py
# Scraping-only version of month_ticker_pipeline.py
# - Uses config_companies.py / COMPANIES
# - Queries GDELT month-by-month for article URLs
# - Downloads article HTML and extracts full text
# - Saves metadata + full-text article table
# - Does NOT run FinBERT, embeddings, sentiment, or trading-day aggregation

import os
import re
import json
import time
import argparse
import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from config_companies import COMPANIES


# ============================================================
# Defaults
# ============================================================

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_OUTPUT_DIR = "data/processed/month_ticker"
DEFAULT_LOG_DIR = "logs"
DEFAULT_UNIVERSE_PATH = "data/universe.json"


# ============================================================
# Runtime config
# ============================================================

@dataclass
class RuntimeConfig:
    year: int
    output_dir: str = DEFAULT_OUTPUT_DIR
    log_dir: str = DEFAULT_LOG_DIR
    universe_path: str = DEFAULT_UNIVERSE_PATH
    overwrite: bool = False

    # GDELT settings. The original pipeline uses monthly windows.
    max_articles_per_month: int = 15
    gdelt_maxrecords: int = 80
    gdelt_timeout: int = 15
    gdelt_retries: int = 1
    gdelt_sleep: float = 0.05
    min_request_interval: float = 0.0
    sort: str = "hybridrel"

    # Article fetching
    article_timeout: int = 5
    article_retries: int = 0
    article_workers: int = 16
    min_body_chars: int = 50
    save_failed_body_snippet: bool = False

    # Time budget. Use 0 to disable deadline.
    max_runtime_minutes: float = 0.0
    stop_before_deadline_seconds: int = 60


class TimeBudget:
    def __init__(self, max_minutes: float):
        self.start = time.time()
        self.max_minutes = float(max_minutes or 0.0)
        self.deadline = None if self.max_minutes <= 0 else self.start + self.max_minutes * 60.0

    def elapsed_seconds(self) -> float:
        return time.time() - self.start

    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds() / 60.0

    def remaining_seconds(self) -> float:
        if self.deadline is None:
            return float("inf")
        return self.deadline - time.time()

    def has_time(self, reserve: float = 0) -> bool:
        if self.deadline is None:
            return True
        return self.remaining_seconds() > reserve


# ============================================================
# Setup / helpers
# ============================================================

def ensure_dirs(cfg: RuntimeConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.universe_path) or ".", exist_ok=True)


def setup_logging(cfg: RuntimeConfig) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(cfg.log_dir, f"scrape_only_{cfg.year}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_article_id(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def compile_entity_regex(patterns: List[str]):
    cleaned = []
    for p in patterns:
        p = clean_text(p)
        if p:
            cleaned.append(re.escape(p))
    if not cleaned:
        return None
    return re.compile("(" + "|".join(cleaned) + ")", flags=re.IGNORECASE)


def get_company_patterns(company: Dict) -> List[str]:
    patterns = []
    if company.get("entity_match_patterns"):
        patterns.extend(company["entity_match_patterns"])
    if company.get("company_name"):
        patterns.append(company["company_name"])
    patterns.append(company["ticker"])

    seen = set()
    out = []
    for p in patterns:
        p = clean_text(p)
        key = p.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def build_gdelt_query(company: Dict) -> str:
    terms = company["query_terms"]
    exclude_terms = company.get("exclude_terms", [])

    if len(terms) == 1:
        base = terms[0]
    else:
        base = "(" + " OR ".join(terms) + ")"

    exclusions = " ".join([f"-{x}" for x in exclude_terms])
    return f"{base} sourcelang:english {exclusions}".strip()


def save_universe_json(cfg: RuntimeConfig) -> None:
    universe = {
        "tickers": [c["ticker"] for c in COMPANIES],
        "companies": [
            {
                "ticker": c["ticker"],
                "company_name": c.get("company_name", c["ticker"]),
                "entity_match_patterns": get_company_patterns(c),
            }
            for c in COMPANIES
        ],
    }
    with open(cfg.universe_path, "w", encoding="utf-8") as f:
        json.dump(universe, f, ensure_ascii=False, indent=2)


def dt_to_gdelt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def parse_gdelt_timestamp(value) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        pass
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return pd.Timestamp(datetime.strptime(str(value), fmt), tz="UTC")
        except Exception:
            pass
    return None


def iter_month_windows_for_year(year: int):
    """Original pipeline behavior: one GDELT artlist query per company per month."""
    for month in range(1, 13):
        start = datetime(year, month, 1)
        end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
        yield start, end


# ============================================================
# GDELT article-list scraping
# ============================================================

def request_gdelt_articlelist(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    cfg: RuntimeConfig,
    budget: TimeBudget,
    rate_state: Dict[str, float],
) -> List[Dict]:
    if not budget.has_time(cfg.stop_before_deadline_seconds):
        return []

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": cfg.gdelt_maxrecords,
        "sort": cfg.sort,
        "startdatetime": dt_to_gdelt(start_dt),
        "enddatetime": dt_to_gdelt(end_dt),
    }

    for attempt in range(cfg.gdelt_retries + 1):
        if not budget.has_time(cfg.stop_before_deadline_seconds):
            return []

        elapsed = time.time() - rate_state.get("last_request_ts", 0.0)
        if cfg.min_request_interval > 0 and elapsed < cfg.min_request_interval:
            time.sleep(cfg.min_request_interval - elapsed)

        try:
            r = requests.get(GDELT_DOC_API, params=params, timeout=cfg.gdelt_timeout)
            rate_state["last_request_ts"] = time.time()

            if r.status_code == 200:
                try:
                    data = r.json()
                    return data.get("articles", []) or []
                except Exception:
                    logging.warning("GDELT JSON parse error: %s", r.text[:200])
                    return []

            if r.status_code in [429, 500, 502, 503, 504]:
                retry_after_raw = r.headers.get("Retry-After", "").strip()
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw else 0.0
                except ValueError:
                    retry_after = 0.0
                wait = max(retry_after, min(60.0, max(2.0, cfg.min_request_interval) * (attempt + 1)))
                logging.warning(
                    "GDELT status=%s. Sleeping %.1fs then retry/skip. query=%s",
                    r.status_code,
                    wait,
                    query,
                )
                time.sleep(wait)
                continue

            logging.warning("GDELT status=%s. Skip window. query=%s", r.status_code, query)
            return []

        except Exception as e:
            logging.warning("GDELT request failed: %s. Retry/skip.", str(e)[:160])
            time.sleep(min(30.0, 2.0 * (attempt + 1)))

    return []


def scrape_metadata_for_year(cfg: RuntimeConfig, budget: TimeBudget) -> pd.DataFrame:
    metadata_path = os.path.join(cfg.output_dir, f"news_metadata_{cfg.year}.parquet")
    if os.path.exists(metadata_path) and not cfg.overwrite:
        logging.info("Loading existing metadata: %s", metadata_path)
        return pd.read_parquet(metadata_path)

    rows = []
    seen = set()
    windows = list(iter_month_windows_for_year(cfg.year))
    rate_state = {"last_request_ts": 0.0}

    for company in tqdm(COMPANIES, desc=f"Companies {cfg.year}"):
        if not budget.has_time(cfg.stop_before_deadline_seconds):
            logging.warning("Stop GDELT metadata scraping due to time budget.")
            break

        ticker = company["ticker"]
        query = build_gdelt_query(company)

        for start_dt, end_dt in tqdm(windows, desc=f"GDELT {ticker} {cfg.year}", leave=False):
            if not budget.has_time(cfg.stop_before_deadline_seconds):
                logging.warning("Stop GDELT metadata scraping due to time budget.")
                break

            raw_articles = request_gdelt_articlelist(query, start_dt, end_dt, cfg, budget, rate_state)
            if cfg.gdelt_sleep > 0:
                time.sleep(cfg.gdelt_sleep)

            raw_articles = raw_articles[: cfg.max_articles_per_month]

            for rank, raw in enumerate(raw_articles, start=1):
                url = raw.get("url")
                if not url:
                    continue

                article_id = stable_article_id(url)
                key = (article_id, ticker)
                if key in seen:
                    continue

                ts_raw = raw.get("seendate") or raw.get("date")
                ts = parse_gdelt_timestamp(ts_raw)
                if ts is None:
                    continue

                seen.add(key)
                rows.append({
                    "article_id": article_id,
                    "url": url,
                    "ticker": ticker,
                    "company_name": company.get("company_name", ticker),
                    "title": clean_text(raw.get("title")),
                    "gdelt_timestamp_utc": ts,
                    "source_domain": raw.get("domain") or urlparse(url).netloc,
                    "language": raw.get("language"),
                    "source_country": raw.get("sourcecountry"),
                    "gdelt_query": query,
                    "window_start_utc": pd.Timestamp(start_dt, tz="UTC"),
                    "window_end_utc": pd.Timestamp(end_dt, tz="UTC"),
                    "rank_within_month": rank,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No GDELT metadata collected for year={cfg.year}")

    df["gdelt_timestamp_utc"] = pd.to_datetime(df["gdelt_timestamp_utc"], utc=True)
    df.to_parquet(metadata_path, index=False)
    df.to_csv(os.path.join(cfg.output_dir, f"news_metadata_{cfg.year}.csv"), index=False, encoding="utf-8-sig")

    logging.info("Saved metadata: %s rows=%s", metadata_path, len(df))
    return df


# ============================================================
# Article full-text fetching
# ============================================================

def fetch_article_body(url: str, cfg: RuntimeConfig) -> Dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    last_reason = ""
    for attempt in range(cfg.article_retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=cfg.article_timeout)
            if r.status_code != 200:
                last_reason = f"http_{r.status_code}"
                time.sleep(0.5 * attempt)
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                tag.decompose()

            page_title = ""
            if soup.title and soup.title.string:
                page_title = clean_text(soup.title.string)

            paragraphs = [clean_text(p.get_text(" ")) for p in soup.find_all("p")]
            paragraphs = [p for p in paragraphs if len(p) >= 30]

            if paragraphs:
                body = clean_text(" ".join(paragraphs))
            else:
                body = clean_text(soup.get_text(" "))

            if len(body) < cfg.min_body_chars:
                return {
                    "url": url,
                    "ok": False,
                    "page_title": page_title,
                    "body": body if cfg.save_failed_body_snippet else "",
                    "body_chars": len(body),
                    "reason": "body_too_short",
                }

            return {
                "url": url,
                "ok": True,
                "page_title": page_title,
                "body": body,
                "body_chars": len(body),
                "reason": "",
            }

        except requests.Timeout:
            last_reason = "timeout"
        except Exception as e:
            last_reason = str(e)[:160]

        if attempt < cfg.article_retries:
            time.sleep(1.0 * (attempt + 1))

    return {"url": url, "ok": False, "page_title": "", "body": "", "body_chars": 0, "reason": last_reason}


def fetch_bodies_concurrently(metadata_df: pd.DataFrame, cfg: RuntimeConfig, budget: TimeBudget) -> Dict[str, Dict]:
    unique_urls = metadata_df["url"].dropna().unique().tolist()
    url_results = {}

    with ThreadPoolExecutor(max_workers=cfg.article_workers) as executor:
        futures = {}
        for url in unique_urls:
            if not budget.has_time(cfg.stop_before_deadline_seconds):
                logging.warning("Stop scheduling article fetch due to time budget.")
                break
            futures[executor.submit(fetch_article_body, url, cfg)] = url

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetch article bodies"):
            if not budget.has_time(cfg.stop_before_deadline_seconds):
                logging.warning("Stop collecting article bodies due to time budget.")
                break
            url = futures[fut]
            try:
                url_results[url] = fut.result()
            except Exception as e:
                url_results[url] = {
                    "url": url,
                    "ok": False,
                    "page_title": "",
                    "body": "",
                    "body_chars": 0,
                    "reason": str(e)[:160],
                }

    return url_results


def build_fulltext_article_table(
    metadata_df: pd.DataFrame,
    url_results: Dict[str, Dict],
    cfg: RuntimeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    article_path = os.path.join(cfg.output_dir, f"news_articles_fulltext_{cfg.year}.parquet")
    failed_path = os.path.join(cfg.output_dir, f"news_article_fetch_failed_{cfg.year}.parquet")

    if os.path.exists(article_path) and not cfg.overwrite:
        article_df = pd.read_parquet(article_path)
        failed_df = pd.read_parquet(failed_path) if os.path.exists(failed_path) else pd.DataFrame()
        return article_df, failed_df

    company_map = {c["ticker"]: c for c in COMPANIES}
    rows = []
    failed = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Build full-text table"):
        ticker = row["ticker"]
        company = company_map.get(ticker, {})
        patterns = get_company_patterns(company) if company else [ticker]
        regex = compile_entity_regex(patterns)

        url = row["url"]
        result = url_results.get(url, {
            "ok": False,
            "page_title": "",
            "body": "",
            "body_chars": 0,
            "reason": "not_fetched_due_to_budget",
        })

        gdelt_title = clean_text(row.get("title"))
        page_title = clean_text(result.get("page_title"))
        title = gdelt_title or page_title
        full_text = clean_text(result.get("body")) if result.get("ok") else clean_text(result.get("body"))

        title_match = bool(regex.search(title)) if regex and title else False
        body_match = bool(regex.search(full_text)) if regex and full_text else False
        matched_entity = bool(title_match or body_match)

        base = {
            "article_id": row["article_id"],
            "url": url,
            "ticker": ticker,
            "company_name": row.get("company_name", company.get("company_name", ticker)),
            "gdelt_title": gdelt_title,
            "page_title": page_title,
            "title": title,
            "gdelt_timestamp_utc": row["gdelt_timestamp_utc"],
            "source_domain": row.get("source_domain", ""),
            "language": row.get("language", ""),
            "source_country": row.get("source_country", ""),
            "gdelt_query": row.get("gdelt_query", ""),
            "window_start_utc": row.get("window_start_utc"),
            "window_end_utc": row.get("window_end_utc"),
            "body_fetch_ok": bool(result.get("ok")),
            "fetch_reason": result.get("reason", ""),
            "text_len": int(len(full_text)),
            "title_match": bool(title_match),
            "body_match": bool(body_match),
            "matched_entity": bool(matched_entity),
            "full_text": full_text,
        }

        if result.get("ok"):
            rows.append(base)
        else:
            failed.append(base)

    article_df = pd.DataFrame(rows)
    failed_df = pd.DataFrame(failed)

    if not article_df.empty:
        article_df["gdelt_timestamp_utc"] = pd.to_datetime(article_df["gdelt_timestamp_utc"], utc=True)
    if not failed_df.empty:
        failed_df["gdelt_timestamp_utc"] = pd.to_datetime(failed_df["gdelt_timestamp_utc"], utc=True)

    article_df.to_parquet(article_path, index=False)
    article_df.to_csv(os.path.join(cfg.output_dir, f"news_articles_fulltext_{cfg.year}.csv"), index=False, encoding="utf-8-sig")

    if failed_df.empty:
        failed_df = pd.DataFrame(columns=list(article_df.columns) if not article_df.empty else [
            "article_id", "url", "ticker", "company_name", "title", "body_fetch_ok", "fetch_reason", "text_len"
        ])
    failed_df.to_parquet(failed_path, index=False)
    failed_df.to_csv(os.path.join(cfg.output_dir, f"news_article_fetch_failed_{cfg.year}.csv"), index=False, encoding="utf-8-sig")

    logging.info("Saved full-text articles: %s rows=%s", article_path, len(article_df))
    logging.info("Saved failed fetch log: %s rows=%s", failed_path, len(failed_df))
    return article_df, failed_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Scraping-only monthly GDELT full-text pipeline.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--universe_path", type=str, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_articles_per_month", type=int, default=15)
    parser.add_argument("--max_runtime_minutes", type=float, default=0.0, help="0 disables deadline.")

    parser.add_argument("--gdelt_maxrecords", type=int, default=80)
    parser.add_argument("--gdelt_timeout", type=int, default=15)
    parser.add_argument("--gdelt_retries", type=int, default=1)
    parser.add_argument("--gdelt_sleep", type=float, default=0.05)
    parser.add_argument("--min_request_interval", type=float, default=0.0)
    parser.add_argument("--sort", type=str, default="hybridrel", choices=["hybridrel", "datedesc", "dateasc"])

    parser.add_argument("--article_workers", type=int, default=16)
    parser.add_argument("--article_timeout", type=int, default=5)
    parser.add_argument("--article_retries", type=int, default=0)
    parser.add_argument("--min_body_chars", type=int, default=50)
    parser.add_argument("--save_failed_body_snippet", action="store_true")

    args = parser.parse_args()

    cfg = RuntimeConfig(
        year=args.year,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        universe_path=args.universe_path,
        overwrite=args.overwrite,
        max_articles_per_month=args.max_articles_per_month,
        max_runtime_minutes=args.max_runtime_minutes,
        gdelt_maxrecords=args.gdelt_maxrecords,
        gdelt_timeout=args.gdelt_timeout,
        gdelt_retries=args.gdelt_retries,
        gdelt_sleep=args.gdelt_sleep,
        min_request_interval=args.min_request_interval,
        sort=args.sort,
        article_workers=args.article_workers,
        article_timeout=args.article_timeout,
        article_retries=args.article_retries,
        min_body_chars=args.min_body_chars,
        save_failed_body_snippet=args.save_failed_body_snippet,
    )

    ensure_dirs(cfg)
    setup_logging(cfg)
    save_universe_json(cfg)
    budget = TimeBudget(cfg.max_runtime_minutes)

    logging.info("Starting scraping-only pipeline year=%s", cfg.year)
    logging.info("Config=%s", cfg)
    logging.info("Companies=%s", len(COMPANIES))

    metadata_df = scrape_metadata_for_year(cfg, budget)
    print(f"Metadata rows: {len(metadata_df):,}")

    url_results = fetch_bodies_concurrently(metadata_df, cfg, budget)
    print(f"Fetched URL results: {len(url_results):,}")

    article_df, failed_df = build_fulltext_article_table(metadata_df, url_results, cfg)
    print(f"Full-text article rows: {len(article_df):,}")
    print(f"Failed fetch rows: {len(failed_df):,}")

    if not article_df.empty:
        print("\nArticles per ticker:")
        print(article_df.groupby("ticker").size().sort_values().to_string())

    elapsed = budget.elapsed_minutes()
    print(f"\nDone year={cfg.year}. Runtime: {elapsed:.2f} minutes")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Metadata: {os.path.join(cfg.output_dir, f'news_metadata_{cfg.year}.parquet')}")
    print(f"Full text: {os.path.join(cfg.output_dir, f'news_articles_fulltext_{cfg.year}.parquet')}")
    print(f"Failed log: {os.path.join(cfg.output_dir, f'news_article_fetch_failed_{cfg.year}.parquet')}")


if __name__ == "__main__":
    main()
