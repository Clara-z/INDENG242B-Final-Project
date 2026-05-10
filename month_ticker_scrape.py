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
import random
import argparse
import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Iterable, Tuple, Set

import pandas as pd
import requests
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
    tickers_file: str = "tickers.txt"
    overwrite: bool = False

    # GDELT settings. The original pipeline uses monthly windows.
    max_articles_per_month: int = 30
    max_articles_per_ticker_year: int = 0
    window_days: int = 30
    max_windows_per_ticker: int = 0
    gdelt_maxrecords: int = 80
    gdelt_timeout: int = 15
    gdelt_retries: int = 3
    gdelt_sleep: float = 0.05
    min_request_interval: float = 7.0
    sort: str = "hybridrel"
    gdelt_backoff_base: float = 2.0
    gdelt_max_backoff: float = 90.0
    gdelt_domain_filter: str = "theguardian.com"

    # Article fetching
    article_timeout: int = 5
    article_retries: int = 0
    article_workers: int = 16
    min_body_chars: int = 50
    save_failed_body_snippet: bool = False
    skip_fulltext: bool = False
    guardian_api_key: str = "test"
    guardian_timeout: int = 10
    guardian_retries: int = 1

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


def build_gdelt_query(company: Dict, cfg: RuntimeConfig) -> str:
    """Build GDELT query using only the company name (avoids 'phrase too short' errors).
    
    GDELT rejects short quoted phrases like "Apple" or "AAPL" (< 6 chars).
    Using the full company_name (e.g., "Apple Inc.") is more reliable.
    """
    company_name = company.get("company_name", "")
    exclude_terms = company.get("exclude_terms", [])

    # Use company_name as the primary search term (quoted for exact phrase match)
    if company_name:
        # Ensure it's quoted
        if not (company_name.startswith('"') and company_name.endswith('"')):
            base = f'"{company_name}"'
        else:
            base = company_name
    else:
        # Fallback to first query_term if no company_name
        terms = company.get("query_terms", [])
        base = terms[0] if terms else company.get("ticker", "UNKNOWN")

    exclusions = " ".join([f"-{x}" for x in exclude_terms])
    domain_filter = ""
    if clean_text(cfg.gdelt_domain_filter):
        domain_filter = f" domainis:{clean_text(cfg.gdelt_domain_filter)}"
    return f"{base} sourcelang:english{domain_filter} {exclusions}".strip()


def save_universe_json(cfg: RuntimeConfig) -> None:
    save_universe_json_for_companies(cfg, COMPANIES)


def load_tickers_from_file(path: str) -> Set[str]:
    p = (path or "").strip()
    if not p or not os.path.exists(p):
        return set()
    out: Set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = clean_text(line).upper()
            if not s or s.startswith("#"):
                continue
            out.add(s)
    return out


def select_companies(cfg: RuntimeConfig) -> List[Dict]:
    ticker_filter = load_tickers_from_file(cfg.tickers_file)
    if not ticker_filter:
        logging.info("Using all tickers from config_companies.py (no valid tickers file filter).")
        return list(COMPANIES)

    selected = [c for c in COMPANIES if c.get("ticker", "").upper() in ticker_filter]
    missing = sorted([t for t in ticker_filter if t not in {c["ticker"].upper() for c in COMPANIES}])
    logging.info(
        "Ticker filter loaded from %s: requested=%s, selected=%s, missing_in_config=%s",
        cfg.tickers_file,
        len(ticker_filter),
        len(selected),
        len(missing),
    )
    if missing:
        logging.warning("Tickers not found in config_companies.py: %s", ",".join(missing))
    return selected


def save_universe_json_for_companies(cfg: RuntimeConfig, companies: List[Dict]) -> None:
    universe = {
        "tickers": [c["ticker"] for c in companies],
        "companies": [
            {
                "ticker": c["ticker"],
                "company_name": c.get("company_name", c["ticker"]),
                "entity_match_patterns": get_company_patterns(c),
            }
            for c in companies
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


def is_guardian_url(url: str) -> bool:
    host = (urlparse(clean_text(url)).netloc or "").lower()
    return ("theguardian.com" in host) or ("content.guardianapis.com" in host)


def fetch_guardian_content(url: str, cfg: RuntimeConfig) -> Dict:
    if not is_guardian_url(url):
        return {}

    parsed = urlparse(clean_text(url))
    path = (parsed.path or "").strip("/")
    if not path:
        return {}
    if path.startswith("http"):
        path = path.split("://", 1)[-1].split("/", 1)[-1]
    endpoint = f"https://content.guardianapis.com/{path}"
    params = {
        "api-key": cfg.guardian_api_key or "test",
        "show-fields": "headline,trailText,bodyText,wordcount,tone",
        "show-tags": "all",
    }

    for attempt in range(cfg.guardian_retries + 1):
        try:
            resp = requests.get(endpoint, params=params, timeout=cfg.guardian_timeout)
            if resp.status_code == 200:
                payload = resp.json()
                content = payload.get("response", {}).get("content", {}) or {}
                fields = content.get("fields", {}) or {}
                tags_raw = content.get("tags", []) or []
                tags = []
                for t in tags_raw:
                    if isinstance(t, dict):
                        tags.append(
                            {
                                "id": t.get("id"),
                                "type": t.get("type"),
                                "webTitle": t.get("webTitle"),
                            }
                        )
                return {
                    "guardian_id": content.get("id"),
                    "type": content.get("type") or "article",
                    "sectionId": content.get("sectionId"),
                    "sectionName": content.get("sectionName"),
                    "pillarId": content.get("pillarId"),
                    "pillarName": content.get("pillarName"),
                    "webPublicationDate": content.get("webPublicationDate"),
                    "webTitle": clean_text(content.get("webTitle")),
                    "webUrl": clean_text(content.get("webUrl")) or clean_text(url),
                    "apiUrl": clean_text(content.get("apiUrl")),
                    "headline": clean_text(fields.get("headline")),
                    "trailText": clean_text(fields.get("trailText")),
                    "bodyText": clean_text(fields.get("bodyText")),
                    "wordcount": str(fields.get("wordcount")) if fields.get("wordcount") is not None else None,
                    "guardian_tone": fields.get("tone"),
                    "tags": tags,
                    "source": "guardian",
                }
            if resp.status_code in [429, 500, 502, 503, 504] and attempt < cfg.guardian_retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
                continue
            return {}
        except Exception:
            if attempt < cfg.guardian_retries:
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
                continue
            return {}
    return {}


def iter_windows_for_year(year: int, window_days: int) -> Iterable[Tuple[datetime, datetime]]:
    """Generate fixed-size windows (default 30 days) within the target year."""
    start = datetime(year, 1, 1)
    year_end = datetime(year + 1, 1, 1)
    step_days = max(1, int(window_days))
    cur = start
    while cur < year_end:
        nxt = cur + pd.Timedelta(days=step_days)
        if nxt > year_end:
            nxt = year_end
        yield cur, nxt
        cur = nxt


# ============================================================
# GDELT article-list scraping
# ============================================================

def request_gdelt_articlelist(
    session: requests.Session,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    cfg: RuntimeConfig,
    budget: TimeBudget,
    rate_state: Dict[str, float],
) -> Optional[List[Dict]]:
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
            return None

        elapsed = time.time() - rate_state.get("last_request_ts", 0.0)
        if cfg.min_request_interval > 0 and elapsed < cfg.min_request_interval:
            time.sleep(cfg.min_request_interval - elapsed)

        try:
            r = session.get(GDELT_DOC_API, params=params, timeout=cfg.gdelt_timeout)
            rate_state["last_request_ts"] = time.time()

            if r.status_code == 200:
                try:
                    data = r.json()
                    return data.get("articles", []) or []
                except Exception:
                    msg = (r.text or "").strip().lower()
                    if "please limit requests to one every 5 seconds" in msg:
                        wait = min(
                            cfg.gdelt_max_backoff,
                            max(
                                5.5,
                                cfg.gdelt_backoff_base * (2 ** attempt),
                                cfg.min_request_interval * (attempt + 1),
                            ),
                        )
                        wait = wait + random.uniform(0.0, 0.8)
                        logging.warning(
                            "GDELT text throttle on 200 body. Sleeping %.1fs then retry. query=%s",
                            wait,
                            query,
                        )
                        time.sleep(wait)
                        continue
                    logging.warning("GDELT JSON parse error: %s", r.text[:200])
                    wait = min(
                        cfg.gdelt_max_backoff,
                        max(cfg.gdelt_backoff_base, cfg.gdelt_backoff_base * (2 ** attempt)),
                    )
                    wait = wait + random.uniform(0.0, 0.8)
                    if attempt < cfg.gdelt_retries:
                        logging.warning(
                            "Retry after parse error in %.1fs. query=%s",
                            wait,
                            query,
                        )
                        time.sleep(wait)
                        continue
                    return None

            if r.status_code in [429, 500, 502, 503, 504]:
                retry_after_raw = r.headers.get("Retry-After", "").strip()
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw else 0.0
                except ValueError:
                    retry_after = 0.0
                wait = min(
                    cfg.gdelt_max_backoff,
                    max(
                        retry_after,
                        cfg.gdelt_backoff_base * (2 ** attempt),
                        max(2.0, cfg.min_request_interval) * (attempt + 1),
                    ),
                )
                wait = wait + random.uniform(0.0, 1.0)
                logging.warning(
                    "GDELT status=%s. Sleeping %.1fs then retry/skip. query=%s",
                    r.status_code,
                    wait,
                    query,
                )
                time.sleep(wait)
                continue

            logging.warning("GDELT status=%s. Skip window. query=%s", r.status_code, query)
            return None

        except Exception as e:
            logging.warning("GDELT request failed: %s. Retry/skip.", str(e)[:160])
            wait = min(cfg.gdelt_max_backoff, cfg.gdelt_backoff_base * (2 ** attempt))
            wait = wait + random.uniform(0.0, 0.8)
            time.sleep(wait)

    return None


def scrape_metadata_for_year(cfg: RuntimeConfig, budget: TimeBudget, companies: List[Dict]) -> pd.DataFrame:
    metadata_path = os.path.join(cfg.output_dir, f"news_metadata_{cfg.year}.parquet")
    if os.path.exists(metadata_path) and not cfg.overwrite:
        logging.info("Loading existing metadata: %s", metadata_path)
        return pd.read_parquet(metadata_path)

    rows = []
    seen = set()
    windows = list(iter_windows_for_year(cfg.year, cfg.window_days))
    rate_state = {"last_request_ts": 0.0}
    session = requests.Session()
    session.headers.update({"User-Agent": "month-ticker-scraper/1.0"})

    for company in tqdm(companies, desc=f"Companies {cfg.year}"):
        if not budget.has_time(cfg.stop_before_deadline_seconds):
            logging.warning("Stop GDELT metadata scraping due to time budget.")
            break

        ticker = company["ticker"]
        query = build_gdelt_query(company, cfg)
        ticker_kept = 0
        ticker_months_done = 0

        for start_dt, end_dt in tqdm(windows, desc=f"GDELT {ticker} {cfg.year}", leave=False):
            if not budget.has_time(cfg.stop_before_deadline_seconds):
                logging.warning("Stop GDELT metadata scraping due to time budget.")
                break

            if cfg.max_articles_per_ticker_year > 0 and ticker_kept >= cfg.max_articles_per_ticker_year:
                logging.info(
                    "Ticker %s reached max_articles_per_ticker_year=%s. Stop further windows.",
                    ticker,
                    cfg.max_articles_per_ticker_year,
                )
                break

            if cfg.max_windows_per_ticker > 0 and ticker_months_done >= cfg.max_windows_per_ticker:
                logging.info(
                    "Ticker %s reached max_windows_per_ticker=%s. Stop further windows.",
                    ticker,
                    cfg.max_windows_per_ticker,
                )
                break

            raw_articles = request_gdelt_articlelist(
                session=session,
                query=query,
                start_dt=start_dt,
                end_dt=end_dt,
                cfg=cfg,
                budget=budget,
                rate_state=rate_state,
            )

            # Fallback: if one month query failed, split into two half-month windows once.
            if raw_articles is None:
                mid_dt = start_dt + (end_dt - start_dt) / 2
                first_half = request_gdelt_articlelist(
                    session=session,
                    query=query,
                    start_dt=start_dt,
                    end_dt=mid_dt,
                    cfg=cfg,
                    budget=budget,
                    rate_state=rate_state,
                )
                second_half = request_gdelt_articlelist(
                    session=session,
                    query=query,
                    start_dt=mid_dt,
                    end_dt=end_dt,
                    cfg=cfg,
                    budget=budget,
                    rate_state=rate_state,
                )
                raw_articles = (first_half or []) + (second_half or [])
                logging.info(
                    "Ticker %s fallback split window %s -> %s got %s articles",
                    ticker,
                    start_dt.date(),
                    end_dt.date(),
                    len(raw_articles),
                )

            ticker_months_done += 1
            if cfg.gdelt_sleep > 0:
                time.sleep(cfg.gdelt_sleep)

            raw_articles = raw_articles[: cfg.max_articles_per_month]

            for rank, raw in enumerate(raw_articles, start=1):
                if cfg.max_articles_per_ticker_year > 0 and ticker_kept >= cfg.max_articles_per_ticker_year:
                    break
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
                    "window_days": cfg.window_days,
                })
                ticker_kept += 1

        logging.info(
            "Ticker %s done. months_processed=%s kept_articles=%s",
            ticker,
            ticker_months_done,
            ticker_kept,
        )

    session.close()

    expected_cols = [
        "article_id",
        "url",
        "ticker",
        "company_name",
        "title",
        "gdelt_timestamp_utc",
        "source_domain",
        "language",
        "source_country",
        "gdelt_query",
        "window_start_utc",
        "window_end_utc",
        "rank_within_month",
        "window_days",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=expected_cols)
        df.to_parquet(metadata_path, index=False)
        df.to_csv(
            os.path.join(cfg.output_dir, f"news_metadata_{cfg.year}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        logging.warning("No GDELT metadata collected for year=%s. Wrote empty metadata files.", cfg.year)
        return df

    df["gdelt_timestamp_utc"] = pd.to_datetime(df["gdelt_timestamp_utc"], utc=True)
    df.to_parquet(metadata_path, index=False)
    df.to_csv(os.path.join(cfg.output_dir, f"news_metadata_{cfg.year}.csv"), index=False, encoding="utf-8-sig")

    logging.info("Saved metadata: %s rows=%s", metadata_path, len(df))
    return df


# ============================================================
# Article full-text fetching
# ============================================================

def fetch_article_body(url: str, cfg: RuntimeConfig) -> Dict:
    guardian_meta = fetch_guardian_content(url, cfg)
    guardian_body = clean_text(guardian_meta.get("bodyText", ""))

    try:
        from bs4 import BeautifulSoup
    except Exception:
        if len(guardian_body) >= cfg.min_body_chars:
            page_title = clean_text(guardian_meta.get("webTitle") or guardian_meta.get("headline"))
            return {
                "url": url,
                "ok": True,
                "page_title": page_title,
                "body": guardian_body,
                "body_chars": len(guardian_body),
                "reason": "",
                "guardian_meta": guardian_meta,
            }
        return {
            "url": url,
            "ok": False,
            "page_title": "",
            "body": "",
            "body_chars": 0,
            "reason": "missing_dependency_bs4",
            "guardian_meta": guardian_meta,
        }

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
            if len(body) < cfg.min_body_chars and len(guardian_body) >= cfg.min_body_chars:
                body = guardian_body
                if not page_title:
                    page_title = clean_text(guardian_meta.get("webTitle") or guardian_meta.get("headline"))

            if len(body) < cfg.min_body_chars:
                return {
                    "url": url,
                    "ok": False,
                    "page_title": page_title,
                    "body": body if cfg.save_failed_body_snippet else "",
                    "body_chars": len(body),
                    "reason": "body_too_short",
                    "guardian_meta": guardian_meta,
                }

            return {
                "url": url,
                "ok": True,
                "page_title": page_title,
                "body": body,
                "body_chars": len(body),
                "reason": "",
                "guardian_meta": guardian_meta,
            }

        except requests.Timeout:
            last_reason = "timeout"
        except Exception as e:
            last_reason = str(e)[:160]

        if attempt < cfg.article_retries:
            time.sleep(1.0 * (attempt + 1))

    if len(guardian_body) >= cfg.min_body_chars:
        return {
            "url": url,
            "ok": True,
            "page_title": clean_text(guardian_meta.get("webTitle") or guardian_meta.get("headline")),
            "body": guardian_body,
            "body_chars": len(guardian_body),
            "reason": "guardian_api_fallback",
            "guardian_meta": guardian_meta,
        }

    return {
        "url": url,
        "ok": False,
        "page_title": "",
        "body": "",
        "body_chars": 0,
        "reason": last_reason,
        "guardian_meta": guardian_meta,
    }


def fetch_bodies_concurrently(metadata_df: pd.DataFrame, cfg: RuntimeConfig, budget: TimeBudget) -> Dict[str, Dict]:
    if metadata_df.empty or "url" not in metadata_df.columns:
        return {}
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
                    "guardian_meta": {},
                }

    return url_results


def build_fulltext_article_table(
    metadata_df: pd.DataFrame,
    url_results: Dict[str, Dict],
    cfg: RuntimeConfig,
    companies: List[Dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    article_path = os.path.join(cfg.output_dir, f"news_articles_fulltext_{cfg.year}.parquet")
    failed_path = os.path.join(cfg.output_dir, f"news_article_fetch_failed_{cfg.year}.parquet")

    if os.path.exists(article_path) and not cfg.overwrite:
        article_df = pd.read_parquet(article_path)
        failed_df = pd.read_parquet(failed_path) if os.path.exists(failed_path) else pd.DataFrame()
        return article_df, failed_df

    company_map = {c["ticker"]: c for c in companies}
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
            "guardian_meta": {},
        })

        gdelt_title = clean_text(row.get("title"))
        guardian_meta = result.get("guardian_meta", {}) or {}
        derived = _derive_guardian_fields(url)
        page_title = clean_text(result.get("page_title"))
        title = clean_text(guardian_meta.get("headline")) or gdelt_title or page_title
        full_text = clean_text(result.get("body")) if result.get("ok") else clean_text(result.get("body"))
        body_text = clean_text(guardian_meta.get("bodyText")) or full_text
        web_title = clean_text(guardian_meta.get("webTitle")) or title

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
            "webPublicationDate": guardian_meta.get("webPublicationDate"),
            "guardian_id": guardian_meta.get("guardian_id") or derived["guardian_id"],
            "type": guardian_meta.get("type") or "article",
            "sectionId": guardian_meta.get("sectionId") or derived["sectionId"],
            "sectionName": guardian_meta.get("sectionName") or derived["sectionName"],
            "pillarId": guardian_meta.get("pillarId"),
            "pillarName": guardian_meta.get("pillarName"),
            "webTitle": web_title,
            "webUrl": clean_text(guardian_meta.get("webUrl")) or clean_text(url),
            "apiUrl": guardian_meta.get("apiUrl") or derived["apiUrl"],
            "headline": clean_text(guardian_meta.get("headline")) or title,
            "trailText": clean_text(guardian_meta.get("trailText")),
            "bodyText": body_text,
            "wordcount": guardian_meta.get("wordcount"),
            "guardian_tone": guardian_meta.get("guardian_tone"),
            "tags": guardian_meta.get("tags") if isinstance(guardian_meta.get("tags"), list) else [],
            "source_domain": row.get("source_domain", ""),
            "language": row.get("language", ""),
            "source_country": row.get("source_country", ""),
            "gdelt_query": row.get("gdelt_query", ""),
            "window_start_utc": row.get("window_start_utc"),
            "window_end_utc": row.get("window_end_utc"),
            "source": guardian_meta.get("source") or derived["source"],
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


def _to_iso_utc(ts_value) -> Optional[str]:
    ts = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_date_str(ts_value) -> Optional[str]:
    ts = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _derive_guardian_fields(web_url: str) -> Dict[str, Optional[str]]:
    url = clean_text(web_url)
    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    section_id = path.split("/")[0] if path else None
    section_name = section_id.replace("-", " ").title() if section_id else None

    host = (parsed.netloc or "").lower()
    is_guardian = ("theguardian.com" in host) or ("content.guardianapis.com" in host)
    api_url = f"https://content.guardianapis.com/{path}" if is_guardian and path else None

    return {
        "guardian_id": path or None,
        "sectionId": section_id,
        "sectionName": section_name,
        "apiUrl": api_url,
        "source": "guardian" if is_guardian else "gdelt",
    }


def export_sample_style_json(article_df: pd.DataFrame, cfg: RuntimeConfig) -> str:
    out_path = os.path.join(cfg.output_dir, f"sample_guardian_news_{cfg.year}.json")
    rows: List[Dict] = []

    for _, row in article_df.iterrows():
        web_url = clean_text(row.get("url"))
        title = clean_text(row.get("title"))
        body_text = clean_text(row.get("bodyText")) or clean_text(row.get("full_text"))
        trail_text = clean_text(row.get("trailText"))
        headline = clean_text(row.get("headline")) or title
        text_for_finbert = clean_text(" ".join([headline, trail_text, body_text]))
        wc = str(row.get("wordcount")) if row.get("wordcount") not in [None, ""] else str(len(body_text.split()) if body_text else 0)

        derived = _derive_guardian_fields(web_url)
        web_pub_dt = row.get("webPublicationDate") or _to_iso_utc(row.get("gdelt_timestamp_utc"))
        from_date_window = _to_date_str(row.get("window_start_utc"))
        to_date_window = _to_date_str(row.get("window_end_utc"))
        tags = row.get("tags")
        if not isinstance(tags, list):
            tags = []

        rows.append(
            {
                "uid": row.get("article_id"),
                "ticker": row.get("ticker"),
                "webPublicationDate": web_pub_dt,
                "publication_date": _to_date_str(row.get("gdelt_timestamp_utc")),
                "trading_date": None,
                "guardian_id": row.get("guardian_id") or derived["guardian_id"],
                "type": row.get("type") or "article",
                "sectionId": row.get("sectionId") or derived["sectionId"],
                "sectionName": row.get("sectionName") or derived["sectionName"],
                "pillarId": row.get("pillarId"),
                "pillarName": row.get("pillarName"),
                "webTitle": clean_text(row.get("webTitle")) or title,
                "webUrl": web_url,
                "apiUrl": row.get("apiUrl") or derived["apiUrl"],
                "headline": headline,
                "trailText": trail_text,
                "bodyText": body_text,
                "text_for_finbert": text_for_finbert,
                "wordcount": wc,
                "guardian_tone": row.get("guardian_tone"),
                "tags": tags,
                "query": row.get("gdelt_query", ""),
                "from_date_window": from_date_window,
                "to_date_window": to_date_window,
                "source": row.get("source") or derived["source"],
                "backfill": True,
                "backfill_order_by": cfg.sort,
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    logging.info("Saved sample-style JSON: %s rows=%s", out_path, len(rows))
    return out_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Scraping-only monthly GDELT full-text pipeline.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    parser.add_argument("--universe_path", type=str, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument(
        "--tickers_file",
        type=str,
        default="tickers.txt",
        help="Optional ticker whitelist file. If missing/empty, uses all config_companies.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_articles_per_month", type=int, default=30)
    parser.add_argument(
        "--max_articles_per_ticker_year",
        type=int,
        default=0,
        help="Cap kept metadata rows per ticker per year (0 disables cap).",
    )
    parser.add_argument("--max_runtime_minutes", type=float, default=0.0, help="0 disables deadline.")
    parser.add_argument("--window_days", type=int, default=30, help="GDELT query window size in days.")
    parser.add_argument(
        "--max_windows_per_ticker",
        type=int,
        default=0,
        help="Stop after N windows per ticker (0 means all windows in year).",
    )

    parser.add_argument("--gdelt_maxrecords", type=int, default=80)
    parser.add_argument("--gdelt_timeout", type=int, default=15)
    parser.add_argument("--gdelt_retries", type=int, default=1)
    parser.add_argument("--gdelt_sleep", type=float, default=0.05)
    parser.add_argument("--min_request_interval", type=float, default=7.0)
    parser.add_argument("--gdelt_backoff_base", type=float, default=2.0)
    parser.add_argument("--gdelt_max_backoff", type=float, default=90.0)
    parser.add_argument(
        "--gdelt_domain_filter",
        type=str,
        default="theguardian.com",
        help="Restrict GDELT article source domain (empty string disables).",
    )
    parser.add_argument("--sort", type=str, default="hybridrel", choices=["hybridrel", "datedesc", "dateasc"])

    parser.add_argument("--article_workers", type=int, default=16)
    parser.add_argument("--article_timeout", type=int, default=5)
    parser.add_argument("--article_retries", type=int, default=0)
    parser.add_argument("--min_body_chars", type=int, default=50)
    parser.add_argument("--save_failed_body_snippet", action="store_true")
    parser.add_argument(
        "--skip_fulltext",
        action="store_true",
        help="Only run GDELT metadata scraping (skip article-body downloads).",
    )

    args = parser.parse_args()

    cfg = RuntimeConfig(
        year=args.year,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        universe_path=args.universe_path,
        tickers_file=args.tickers_file,
        overwrite=args.overwrite,
        max_articles_per_month=args.max_articles_per_month,
        max_articles_per_ticker_year=args.max_articles_per_ticker_year,
        max_runtime_minutes=args.max_runtime_minutes,
        window_days=args.window_days,
        max_windows_per_ticker=args.max_windows_per_ticker,
        gdelt_maxrecords=args.gdelt_maxrecords,
        gdelt_timeout=args.gdelt_timeout,
        gdelt_retries=args.gdelt_retries,
        gdelt_sleep=args.gdelt_sleep,
        min_request_interval=max(7.0, float(args.min_request_interval)),
        gdelt_backoff_base=args.gdelt_backoff_base,
        gdelt_max_backoff=args.gdelt_max_backoff,
        gdelt_domain_filter=args.gdelt_domain_filter,
        sort=args.sort,
        article_workers=args.article_workers,
        article_timeout=args.article_timeout,
        article_retries=args.article_retries,
        min_body_chars=args.min_body_chars,
        save_failed_body_snippet=args.save_failed_body_snippet,
        skip_fulltext=args.skip_fulltext,
    )

    ensure_dirs(cfg)
    setup_logging(cfg)
    companies = select_companies(cfg)
    if not companies:
        raise RuntimeError("No companies selected. Check --tickers_file and config_companies.py.")
    save_universe_json_for_companies(cfg, companies)
    budget = TimeBudget(cfg.max_runtime_minutes)

    logging.info("Starting scraping-only pipeline year=%s", cfg.year)
    logging.info("Config=%s", cfg)
    logging.info("Companies selected=%s", len(companies))
    logging.info("Ticker source file=%s", cfg.tickers_file)
    logging.info("Window days=%s", cfg.window_days)
    logging.info("Max windows per ticker=%s", cfg.max_windows_per_ticker)
    logging.info("Effective min_request_interval=%s seconds", cfg.min_request_interval)

    metadata_df = scrape_metadata_for_year(cfg, budget, companies)
    print(f"Metadata rows: {len(metadata_df):,}")

    if metadata_df.empty:
        print("No metadata rows collected. Exiting gracefully.")
        elapsed = budget.elapsed_minutes()
        print(f"\nDone year={cfg.year}. Runtime: {elapsed:.2f} minutes")
        print(f"Output dir: {cfg.output_dir}")
        print(f"Metadata: {os.path.join(cfg.output_dir, f'news_metadata_{cfg.year}.parquet')}")
        return

    if cfg.skip_fulltext:
        print("Skip fulltext stage: enabled")
        article_df = pd.DataFrame()
        failed_df = pd.DataFrame()
    else:
        url_results = fetch_bodies_concurrently(metadata_df, cfg, budget)
        print(f"Fetched URL results: {len(url_results):,}")

        article_df, failed_df = build_fulltext_article_table(metadata_df, url_results, cfg, companies)
        print(f"Full-text article rows: {len(article_df):,}")
        print(f"Failed fetch rows: {len(failed_df):,}")

    if not cfg.skip_fulltext:
        export_df = pd.concat([article_df, failed_df], ignore_index=True)
        if not export_df.empty:
            print("\nArticles per ticker:")
            print(export_df.groupby("ticker").size().sort_values().to_string())
            sample_json_path = export_sample_style_json(export_df, cfg)
            print(f"Sample-style JSON: {sample_json_path}")

    elapsed = budget.elapsed_minutes()
    print(f"\nDone year={cfg.year}. Runtime: {elapsed:.2f} minutes")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Metadata: {os.path.join(cfg.output_dir, f'news_metadata_{cfg.year}.parquet')}")
    if not cfg.skip_fulltext:
        print(f"Full text: {os.path.join(cfg.output_dir, f'news_articles_fulltext_{cfg.year}.parquet')}")
        print(f"Failed log: {os.path.join(cfg.output_dir, f'news_article_fetch_failed_{cfg.year}.parquet')}")


if __name__ == "__main__":
    main()
