# 01_strict_1h_year_pipeline.py

import os
import re
import json
import time
import argparse
import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas_market_calendars as mcal

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk = None
    sent_tokenize = None

from config_companies import COMPANIES


# ============================================================
# Fixed paths
# ============================================================

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

OUTPUT_DIR = "data/processed/month_ticker"
LOG_DIR = "logs"
UNIVERSE_PATH = "data/universe.json"

FINBERT_MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Runtime config
# ============================================================

@dataclass
class RuntimeConfig:
    year: int
    max_articles_per_month: int = 15
    max_runtime_minutes: float = 55.0

    # GDELT settings
    gdelt_maxrecords: int = 80
    gdelt_timeout: int = 15
    gdelt_retries: int = 1
    gdelt_sleep: float = 0.05

    # Article fetching
    article_timeout: int = 5
    article_retries: int = 0
    article_workers: int = 16

    # FinBERT
    batch_size: int = 64
    use_fp16: bool = True

    # Context rules
    min_body_chars: int = 50
    min_title_tokens: int = 10

    # Time reservation
    reserve_seconds_for_finbert_and_aggregation: int = 420
    stop_before_deadline_seconds: int = 30


class TimeBudget:
    def __init__(self, max_minutes: float):
        self.start = time.time()
        self.deadline = self.start + max_minutes * 60.0

    def elapsed_seconds(self) -> float:
        return time.time() - self.start

    def remaining_seconds(self) -> float:
        return self.deadline - time.time()

    def has_time(self, reserve: float = 0) -> bool:
        return self.remaining_seconds() > reserve

    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds() / 60.0


# ============================================================
# Setup
# ============================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)


def setup_logging(year: int):
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, f"strict_1h_pipeline_{year}.log"),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def init_nltk():
    if nltk is None:
        return

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass


# ============================================================
# Text helpers
# ============================================================

def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_sentence_split(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    if sent_tokenize is not None:
        try:
            return [clean_text(s) for s in sent_tokenize(text) if clean_text(s)]
        except Exception:
            pass

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clean_text(s) for s in parts if clean_text(s)]


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


def truncate_to_512(tokenizer, text: str) -> Tuple[str, int, int]:
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    was_truncated = int(len(ids) > MAX_LENGTH)

    if was_truncated:
        ids = ids[:MAX_LENGTH]
        text = tokenizer.decode(ids, skip_special_tokens=True)

    return clean_text(text), int(min(len(ids), MAX_LENGTH)), was_truncated


# ============================================================
# Company helpers
# ============================================================

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
        key = clean_text(p).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(clean_text(p))

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


def save_universe_json():
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

    with open(UNIVERSE_PATH, "w", encoding="utf-8") as f:
        json.dump(universe, f, ensure_ascii=False, indent=2)


# ============================================================
# Calendar helpers
# ============================================================

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
    for month in range(1, 13):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        yield start, end


def get_trading_dates_for_year(year: int) -> List:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
    return [d.date() for d in schedule.index]


def next_trading_day(date_obj, trading_dates: List):
    trading_set = set(trading_dates)
    max_date = trading_dates[-1]

    cur = date_obj
    while cur <= max_date:
        if cur in trading_set:
            return cur
        cur += timedelta(days=1)

    return None


def attribute_to_trading_day_market_open(utc_ts, trading_dates: List):
    """
    For market-open prediction:
    - Article before 09:30 ET -> same trading day.
    - Article at/after 09:30 ET -> next trading day.
    """

    if not isinstance(utc_ts, pd.Timestamp):
        utc_ts = pd.Timestamp(utc_ts, tz="UTC")
    else:
        if utc_ts.tzinfo is None:
            utc_ts = utc_ts.tz_localize("UTC")
        else:
            utc_ts = utc_ts.tz_convert("UTC")

    eastern = ZoneInfo("America/New_York")
    t_et = utc_ts.to_pydatetime().astimezone(eastern)

    candidate = t_et.date()

    if t_et.time() >= dtime(9, 30):
        candidate += timedelta(days=1)

    return next_trading_day(candidate, trading_dates)


# ============================================================
# GDELT scraping: monthly, low retry, time-budgeted
# ============================================================

def request_gdelt_articlelist(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    cfg: RuntimeConfig,
    budget: TimeBudget,
) -> List[Dict]:
    if not budget.has_time(60):
        return []

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": cfg.gdelt_maxrecords,
        "sort": "hybridrel",
        "startdatetime": dt_to_gdelt(start_dt),
        "enddatetime": dt_to_gdelt(end_dt),
    }

    for attempt in range(cfg.gdelt_retries + 1):
        if not budget.has_time(60):
            return []

        try:
            r = requests.get(GDELT_DOC_API, params=params, timeout=cfg.gdelt_timeout)

            if r.status_code == 200:
                try:
                    data = r.json()
                    return data.get("articles", []) or []
                except Exception:
                    logging.warning("GDELT JSON parse error: %s", r.text[:200])
                    return []

            if r.status_code in [429, 500, 502, 503, 504]:
                # Very short wait: this is a strict runtime pipeline.
                wait = min(8, 2 * (attempt + 1))
                logging.warning("GDELT status=%s. Short wait=%ss then retry/skip.", r.status_code, wait)
                time.sleep(wait)
            else:
                logging.warning("GDELT status=%s. Skip window.", r.status_code)
                return []

        except Exception as e:
            logging.warning("GDELT request failed: %s. Skip/retry quickly.", str(e)[:120])
            time.sleep(1)

    return []


def scrape_metadata_for_year(cfg: RuntimeConfig, budget: TimeBudget) -> pd.DataFrame:
    metadata_path = os.path.join(OUTPUT_DIR, f"news_metadata_{cfg.year}.parquet")

    if os.path.exists(metadata_path):
        logging.info("Loading existing metadata: %s", metadata_path)
        return pd.read_parquet(metadata_path)

    rows = []
    seen = set()
    windows = list(iter_month_windows_for_year(cfg.year))

    for company in COMPANIES:
        ticker = company["ticker"]
        query = build_gdelt_query(company)

        for start_dt, end_dt in tqdm(windows, desc=f"GDELT {ticker} {cfg.year}"):
            if not budget.has_time(cfg.reserve_seconds_for_finbert_and_aggregation):
                logging.warning("Stop GDELT early due to time budget.")
                break

            raw_articles = request_gdelt_articlelist(query, start_dt, end_dt, cfg, budget)
            time.sleep(cfg.gdelt_sleep)

            raw_articles = raw_articles[:cfg.max_articles_per_month]

            for raw in raw_articles:
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
                    "title": clean_text(raw.get("title")),
                    "gdelt_timestamp_utc": ts,
                    "source_domain": raw.get("domain") or urlparse(url).netloc,
                    "language": raw.get("language"),
                    "source_country": raw.get("sourcecountry"),
                    "gdelt_query": query,
                    "window_start_utc": pd.Timestamp(start_dt, tz="UTC"),
                    "window_end_utc": pd.Timestamp(end_dt, tz="UTC"),
                })

        if not budget.has_time(cfg.reserve_seconds_for_finbert_and_aggregation):
            break

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No GDELT metadata collected for year={cfg.year}")

    df["gdelt_timestamp_utc"] = pd.to_datetime(df["gdelt_timestamp_utc"], utc=True)
    df.to_parquet(metadata_path, index=False)

    logging.info("Saved metadata: %s rows=%s", metadata_path, len(df))
    return df


# ============================================================
# Article body fetching: concurrent, short timeout
# ============================================================

def fetch_article_body(url: str, cfg: RuntimeConfig) -> Dict:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        r = requests.get(url, headers=headers, timeout=cfg.article_timeout)

        if r.status_code != 200:
            return {
                "url": url,
                "ok": False,
                "page_title": "",
                "body": "",
                "reason": f"http_{r.status_code}",
            }

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        page_title = ""
        if soup.title and soup.title.string:
            page_title = clean_text(soup.title.string)

        # Prefer paragraph text to avoid menus/sidebar.
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
                "body": body,
                "reason": "body_too_short",
            }

        return {
            "url": url,
            "ok": True,
            "page_title": page_title,
            "body": body,
            "reason": "",
        }

    except requests.Timeout:
        return {"url": url, "ok": False, "page_title": "", "body": "", "reason": "timeout"}
    except Exception as e:
        return {"url": url, "ok": False, "page_title": "", "body": "", "reason": str(e)[:120]}


def fetch_bodies_concurrently(metadata_df: pd.DataFrame, cfg: RuntimeConfig, budget: TimeBudget) -> Dict[str, Dict]:
    """
    Fetch each URL once, then reuse body for multiple tickers if duplicated.
    """

    unique_urls = metadata_df["url"].dropna().unique().tolist()

    # If time is already low, reduce URL count.
    if not budget.has_time(300):
        unique_urls = unique_urls[:500]

    url_results = {}

    with ThreadPoolExecutor(max_workers=cfg.article_workers) as executor:
        futures = {}

        for url in unique_urls:
            if not budget.has_time(180):
                logging.warning("Stop scheduling article fetch due to time budget.")
                break

            futures[executor.submit(fetch_article_body, url, cfg)] = url

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetch article bodies"):
            if not budget.has_time(120):
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
                    "reason": str(e)[:120],
                }

    return url_results


# ============================================================
# Context extraction
# ============================================================

def extract_context(
    *,
    title: str,
    body: str,
    ticker: str,
    patterns: List[str],
    tokenizer,
    cfg: RuntimeConfig,
) -> Tuple[Optional[str], str, int, int, int]:
    title = clean_text(title)
    body = clean_text(body)

    regex = compile_entity_regex(patterns)
    title_has_match = bool(regex.search(title)) if regex else False

    body_sentences = simple_sentence_split(body)

    # Case C: title-only article
    if not body_sentences:
        context = title
        context, n_tokens, was_truncated = truncate_to_512(tokenizer, context)

        if n_tokens < cfg.min_title_tokens:
            return None, "drop_title_too_short", 0, n_tokens, was_truncated

        return context, "title_short_article", 1, n_tokens, was_truncated

    # Case D: short article
    if len(body_sentences) < 5:
        context = clean_text(title + " " + body)
        context, n_tokens, was_truncated = truncate_to_512(tokenizer, context)
        return context, "full_short", len(body_sentences), n_tokens, was_truncated

    # Case A: body match
    matched_indices = []
    if regex:
        for i, s in enumerate(body_sentences):
            if regex.search(s):
                matched_indices.append(i)

    if matched_indices:
        keep = set()
        for idx in matched_indices:
            start = max(0, idx - 2)
            end = min(len(body_sentences), idx + 3)
            keep.update(range(start, end))

        selected = [body_sentences[i] for i in sorted(keep)]
        context = " ".join(selected)
        context, n_tokens, was_truncated = truncate_to_512(tokenizer, context)

        return context, "body_match", len(selected), n_tokens, was_truncated

    # Case B: title match but no body match
    if title_has_match:
        selected = body_sentences[:3]
        context = clean_text(title + " " + " ".join(selected))
        context, n_tokens, was_truncated = truncate_to_512(tokenizer, context)

        return context, "title_only", len(selected), n_tokens, was_truncated

    return None, "drop_no_ticker_match_in_body_or_title", len(body_sentences), 0, 0


# ============================================================
# FinBERT
# ============================================================

def map_sentiment_probs(logits, model):
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)

    id2label = getattr(model.config, "id2label", {})
    label_map = {int(k): str(v).lower() for k, v in id2label.items()}

    outputs = []

    for row in probs:
        out = {"positive": None, "neutral": None, "negative": None}

        for idx, p in enumerate(row):
            label = label_map.get(idx, "")
            if "pos" in label:
                out["positive"] = float(p)
            elif "neu" in label:
                out["neutral"] = float(p)
            elif "neg" in label:
                out["negative"] = float(p)

        # ProsusAI/finbert common fallback:
        # 0 positive, 1 negative, 2 neutral
        if any(v is None for v in out.values()):
            out = {
                "positive": float(row[0]),
                "negative": float(row[1]),
                "neutral": float(row[2]),
            }

        outputs.append([
            np.float32(out["positive"]),
            np.float32(out["neutral"]),
            np.float32(out["negative"]),
        ])

    return np.array(outputs, dtype=np.float32)


@torch.no_grad()
def finbert_batch(contexts: List[str], tokenizer, model, cfg: RuntimeConfig):
    encoded = tokenizer(
        contexts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    if DEVICE == "cuda" and cfg.use_fp16:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(**encoded, output_hidden_states=True)
    else:
        outputs = model(**encoded, output_hidden_states=True)

    emb = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy().astype(np.float32)
    sent = map_sentiment_probs(outputs.logits, model)

    return emb, sent


# ============================================================
# Article-level cache construction
# ============================================================

def build_article_cache(
    metadata_df: pd.DataFrame,
    url_results: Dict[str, Dict],
    cfg: RuntimeConfig,
    budget: TimeBudget,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    article_cache_path = os.path.join(OUTPUT_DIR, f"news_articles_finbert_{cfg.year}.parquet")
    dropped_path = os.path.join(OUTPUT_DIR, f"news_dropped_{cfg.year}.parquet")
    preview_path = os.path.join(OUTPUT_DIR, f"news_context_preview_{cfg.year}.csv")

    if os.path.exists(article_cache_path):
        article_df = pd.read_parquet(article_cache_path)
        dropped_df = pd.read_parquet(dropped_path) if os.path.exists(dropped_path) else pd.DataFrame()
        return article_df, dropped_df

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME).to(DEVICE)
    model.eval()

    company_map = {c["ticker"]: c for c in COMPANIES}
    trading_dates = get_trading_dates_for_year(cfg.year)

    context_items = []
    dropped = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Extract contexts"):
        if not budget.has_time(180):
            logging.warning("Stop context extraction due to time budget.")
            break

        ticker = row["ticker"]
        company = company_map[ticker]
        patterns = get_company_patterns(company)

        url = row["url"]
        title_from_gdelt = clean_text(row["title"])

        fetch_result = url_results.get(url, {
            "ok": False,
            "page_title": "",
            "body": "",
            "reason": "not_fetched_due_to_budget",
        })

        title = title_from_gdelt or fetch_result.get("page_title", "")
        body = fetch_result.get("body", "") if fetch_result.get("ok") else ""

        context, extraction_case, n_sent, n_tokens, was_truncated = extract_context(
            title=title,
            body=body,
            ticker=ticker,
            patterns=patterns,
            tokenizer=tokenizer,
            cfg=cfg,
        )

        if not fetch_result.get("ok"):
            dropped.append({
                "article_id": row["article_id"],
                "url": url,
                "ticker": ticker,
                "drop_reason": f"body_fetch_failed_{fetch_result.get('reason', '')}",
                "title": title,
            })

        if context is None:
            dropped.append({
                "article_id": row["article_id"],
                "url": url,
                "ticker": ticker,
                "drop_reason": extraction_case,
                "title": title,
            })
            continue

        trading_date = attribute_to_trading_day_market_open(row["gdelt_timestamp_utc"], trading_dates)

        if trading_date is None:
            dropped.append({
                "article_id": row["article_id"],
                "url": url,
                "ticker": ticker,
                "drop_reason": "outside_trading_calendar",
                "title": title,
            })
            continue

        context_items.append({
            "article_id": row["article_id"],
            "url": url,
            "ticker": ticker,
            "gdelt_timestamp_utc": row["gdelt_timestamp_utc"],
            "trading_date": trading_date,
            "source_domain": row["source_domain"],
            "extraction_case": extraction_case,
            "n_sentences_extracted": np.int32(n_sent),
            "n_tokens": np.int32(n_tokens),
            "was_truncated": np.int8(was_truncated),
            "context_text": context,
            "title": title,
        })

    if not context_items:
        raise RuntimeError("No valid contexts generated.")

    preview_df = pd.DataFrame(context_items)[
        ["ticker", "trading_date", "extraction_case", "title", "context_text", "url"]
    ].sample(min(80, len(context_items)), random_state=42)

    preview_df.to_csv(preview_path, index=False, encoding="utf-8-sig")

    rows = []

    for start in tqdm(range(0, len(context_items), cfg.batch_size), desc="FinBERT"):
        if not budget.has_time(cfg.stop_before_deadline_seconds):
            logging.warning("Stop FinBERT early due to time budget.")
            break

        batch_items = context_items[start:start + cfg.batch_size]
        contexts = [x["context_text"] for x in batch_items]

        emb, sent = finbert_batch(contexts, tokenizer, model, cfg)

        for i, item in enumerate(batch_items):
            out = {
                "article_id": item["article_id"],
                "url": item["url"],
                "ticker": item["ticker"],
                "gdelt_timestamp_utc": item["gdelt_timestamp_utc"],
                "trading_date": item["trading_date"],
                "source_domain": item["source_domain"],
                "extraction_case": item["extraction_case"],
                "n_sentences_extracted": np.int32(item["n_sentences_extracted"]),
                "n_tokens": np.int32(item["n_tokens"]),
                "was_truncated": np.int8(item["was_truncated"]),
                "sentiment_pos": np.float32(sent[i][0]),
                "sentiment_neu": np.float32(sent[i][1]),
                "sentiment_neg": np.float32(sent[i][2]),
            }

            for j in range(768):
                out[f"emb_{j}"] = np.float32(emb[i][j])

            rows.append(out)

    article_df = pd.DataFrame(rows)

    if article_df.empty:
        raise RuntimeError("No FinBERT article rows generated before deadline.")

    article_df["gdelt_timestamp_utc"] = pd.to_datetime(article_df["gdelt_timestamp_utc"], utc=True)
    article_df["trading_date"] = pd.to_datetime(article_df["trading_date"]).dt.date

    emb_cols = [f"emb_{i}" for i in range(768)]
    for col in emb_cols:
        article_df[col] = article_df[col].astype("float32")

    for col in ["sentiment_pos", "sentiment_neu", "sentiment_neg"]:
        article_df[col] = article_df[col].astype("float32")

    article_df["n_sentences_extracted"] = article_df["n_sentences_extracted"].astype("int32")
    article_df["n_tokens"] = article_df["n_tokens"].astype("int32")
    article_df["was_truncated"] = article_df["was_truncated"].astype("int8")

    dropped_df = pd.DataFrame(dropped)

    article_df.to_parquet(article_cache_path, index=False)

    if dropped_df.empty:
        dropped_df = pd.DataFrame(columns=["article_id", "url", "ticker", "drop_reason", "title"])

    dropped_df.to_parquet(dropped_path, index=False)

    logging.info("Saved article cache: %s rows=%s", article_cache_path, len(article_df))
    logging.info("Saved dropped log: %s rows=%s", dropped_path, len(dropped_df))
    logging.info("Saved preview: %s", preview_path)

    return article_df, dropped_df


# ============================================================
# Aggregation
# ============================================================

def aggregate_year(article_df: pd.DataFrame, cfg: RuntimeConfig) -> pd.DataFrame:
    out_path = os.path.join(OUTPUT_DIR, f"news_embeddings_{cfg.year}.parquet")

    trading_dates = get_trading_dates_for_year(cfg.year)
    tickers = [c["ticker"] for c in COMPANIES]
    emb_cols = [f"emb_{i}" for i in range(768)]

    article_df = article_df.copy()
    article_df["trading_date"] = pd.to_datetime(article_df["trading_date"]).dt.date

    grouped = {
        key: g
        for key, g in article_df.groupby(["ticker", "trading_date"])
    }

    rows = []

    for ticker in tqdm(tickers, desc="Aggregate"):
        for date in trading_dates:
            group = grouped.get((ticker, date))

            if group is None or len(group) == 0:
                row = {
                    "ticker": ticker,
                    "date": date,
                    "sentiment_pos": np.float32(0.0),
                    "sentiment_neu": np.float32(0.0),
                    "sentiment_neg": np.float32(0.0),
                    "n_articles": np.int32(0),
                    "has_news": np.int8(0),
                }

                for col in emb_cols:
                    row[col] = np.float32(0.0)

            else:
                mean_emb = group[emb_cols].to_numpy(dtype=np.float32).mean(axis=0)

                row = {
                    "ticker": ticker,
                    "date": date,
                    "sentiment_pos": np.float32(group["sentiment_pos"].mean()),
                    "sentiment_neu": np.float32(group["sentiment_neu"].mean()),
                    "sentiment_neg": np.float32(group["sentiment_neg"].mean()),
                    "n_articles": np.int32(len(group)),
                    "has_news": np.int8(1),
                }

                for i, col in enumerate(emb_cols):
                    row[col] = np.float32(mean_emb[i])

            rows.append(row)

    out = pd.DataFrame(rows)

    ordered_cols = (
        ["ticker", "date"]
        + emb_cols
        + ["sentiment_pos", "sentiment_neu", "sentiment_neg", "n_articles", "has_news"]
    )
    out = out[ordered_cols]

    for col in emb_cols:
        out[col] = out[col].astype("float32")

    out["sentiment_pos"] = out["sentiment_pos"].astype("float32")
    out["sentiment_neu"] = out["sentiment_neu"].astype("float32")
    out["sentiment_neg"] = out["sentiment_neg"].astype("float32")
    out["n_articles"] = out["n_articles"].astype("int32")
    out["has_news"] = out["has_news"].astype("int8")

    out.to_parquet(out_path, index=False)

    logging.info("Saved final embeddings: %s rows=%s", out_path, len(out))
    return out


# ============================================================
# Validation
# ============================================================

def validate_year(final_df: pd.DataFrame, cfg: RuntimeConfig):
    emb_cols = [c for c in final_df.columns if c.startswith("emb_")]

    assert not final_df[emb_cols].isna().any().any(), "NaN found in embeddings"

    has_news = final_df[final_df["has_news"] == 1]
    if len(has_news) > 0:
        sums = has_news["sentiment_pos"] + has_news["sentiment_neu"] + has_news["sentiment_neg"]
        assert ((sums - 1.0).abs() < 0.02).all(), "Sentiment probabilities do not sum to 1"

    no_news = final_df[final_df["has_news"] == 0]
    assert (no_news["sentiment_pos"] == 0).all()
    assert (no_news["sentiment_neu"] == 0).all()
    assert (no_news["sentiment_neg"] == 0).all()

    expected_rows = len(COMPANIES) * len(get_trading_dates_for_year(cfg.year))
    assert len(final_df) == expected_rows, f"Expected {expected_rows}, got {len(final_df)}"


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--max_articles_per_month", type=int, default=15)
    parser.add_argument("--max_runtime_minutes", type=float, default=55.0)
    parser.add_argument("--article_workers", type=int, default=16)
    parser.add_argument("--article_timeout", type=int, default=5)
    parser.add_argument("--gdelt_timeout", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = RuntimeConfig(
        year=args.year,
        max_articles_per_month=args.max_articles_per_month,
        max_runtime_minutes=args.max_runtime_minutes,
        article_workers=args.article_workers,
        article_timeout=args.article_timeout,
        gdelt_timeout=args.gdelt_timeout,
        batch_size=args.batch_size,
    )

    ensure_dirs()
    setup_logging(cfg.year)
    init_nltk()
    save_universe_json()

    budget = TimeBudget(cfg.max_runtime_minutes)

    logging.info("Starting strict 1h pipeline year=%s DEVICE=%s", cfg.year, DEVICE)
    logging.info("Config=%s", cfg)

    metadata_df = scrape_metadata_for_year(cfg, budget)
    print(f"Metadata rows: {len(metadata_df):,}")

    url_results = fetch_bodies_concurrently(metadata_df, cfg, budget)
    print(f"Fetched URL results: {len(url_results):,}")

    article_df, dropped_df = build_article_cache(metadata_df, url_results, cfg, budget)
    print(f"Article FinBERT rows: {len(article_df):,}")
    print(f"Dropped rows: {len(dropped_df):,}")

    final_df = aggregate_year(article_df, cfg)
    print(f"Final rows: {len(final_df):,}")

    validate_year(final_df, cfg)

    elapsed = budget.elapsed_minutes()
    print(f"\nDone year={cfg.year}. Runtime: {elapsed:.2f} minutes")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Article cache: {os.path.join(OUTPUT_DIR, f'news_articles_finbert_{cfg.year}.parquet')}")
    print(f"Final features: {os.path.join(OUTPUT_DIR, f'news_embeddings_{cfg.year}.parquet')}")
    print(f"Preview CSV: {os.path.join(OUTPUT_DIR, f'news_context_preview_{cfg.year}.csv')}")


if __name__ == "__main__":
    main()