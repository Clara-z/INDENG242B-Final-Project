#!/usr/bin/env python3
"""
Temporary helper: dedupe universe tickers using simple keyword rules only.

Input:
  - TXT from export_us_ticker_universe_txt.py
    format: ticker<TAB>company_name<TAB>exchange

Output:
  - kept universe TXT (same format)
  - dropped candidates CSV with reason

Example:
  python "INDENG242B Final Project/dedupe_universe_by_price.py" \
    --input-txt "INDENG242B Final Project/us_listed_universe.txt" \
    --output-kept-txt "INDENG242B Final Project/us_listed_universe_deduped.txt" \
    --output-dropped-csv "INDENG242B Final Project/us_listed_universe_dropped.csv"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dedupe ticker universe by keyword filtering only.")
    parser.add_argument("--input-txt", type=str, required=True, help="Path to source universe TXT.")
    parser.add_argument("--output-kept-txt", type=str, required=True, help="Path to kept TXT.")
    parser.add_argument("--output-dropped-csv", type=str, required=True, help="Path to dropped CSV.")
    parser.add_argument(
        "--drop-keywords",
        type=str,
        default=(
            "warrant,warrants,right,rights,unit,units,preferred,"
            "depositary,notes,note,debenture,subscription,when issued,"
            "series,redeemable,cumulative,fixed-to-floating,fixed to floating"
        ),
        help=(
            "Comma-separated keywords; rows containing any are dropped before dedupe. "
            "Case-insensitive match on company_name."
        ),
    )
    parser.add_argument(
        "--max-duplicate-groups",
        type=int,
        default=0,
        help="Limit duplicate groups for quick tests (0 means all groups).",
    )
    return parser.parse_args()


def load_universe_txt(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in re.split(r"[\t,]", line) if p.strip() != ""]
            if not parts:
                continue
            ticker = parts[0].upper()
            company_name = parts[1] if len(parts) >= 2 else ticker
            exchange = parts[2] if len(parts) >= 3 else "UNKNOWN"
            rows.append({"ticker": ticker, "company_name": company_name, "exchange": exchange})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows found in input TXT.")
    return df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def normalize_keyword(company_name: str) -> str:
    x = str(company_name).lower().strip()
    # Drop common security-type suffixes.
    x = re.sub(r"\s*-\s*.*$", "", x)  # everything after first " - " is usually instrument descriptor.
    x = re.sub(
        r"\b(common stock|ordinary shares?|depositary shares?|american depositary shares?|"
        r"warrants?|rights?|units?|preferred stock|class [a-z]\b)\b",
        " ",
        x,
        flags=re.IGNORECASE,
    )
    x = re.sub(r"\beach representing\b.*$", " ", x)
    x = re.sub(r"[^a-z0-9]+", " ", x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def tradability_rank(company_name: str) -> int:
    x = str(company_name).lower()
    # Higher score = more likely primary common-stock listing.
    if re.search(r"\b(warrant|warrants|right|rights|unit|units|preferred|depositary|notes|debenture)\b", x):
        return 0
    if re.search(r"\b(common stock|ordinary shares?|class [a-z])\b", x):
        return 3
    return 2


def compile_drop_regex(drop_keywords_csv: str) -> re.Pattern:
    tokens = [t.strip().lower() for t in drop_keywords_csv.split(",") if t.strip()]
    if not tokens:
        # Regex that never matches.
        return re.compile(r"$^")
    escaped = [re.escape(t) for t in tokens]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def find_suffix_variant_mask(df: pd.DataFrame) -> pd.Series:
    """
    Mark ticker suffix variants as drop candidates when base ticker exists
    for the same normalized keyword.

    Examples:
      ADACU -> base ADAC
      ADACW -> base ADAC
      XYZWS -> base XYZ
      XYZRT -> base XYZ
    """
    suffixes = ("U", "W", "R", "WS", "WT", "RT")
    ticker_to_keyword = dict(zip(df["ticker"], df["keyword"]))
    all_tickers = set(df["ticker"].tolist())

    flags: list[bool] = []
    for _, row in df.iterrows():
        tk = row["ticker"]
        kw = row["keyword"]
        mark = False
        for suf in suffixes:
            if not tk.endswith(suf):
                continue
            base = tk[: -len(suf)]
            if not base:
                continue
            if base in all_tickers and ticker_to_keyword.get(base) == kw:
                mark = True
                break
        flags.append(mark)
    return pd.Series(flags, index=df.index)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_txt).resolve()
    out_kept = Path(args.output_kept_txt).resolve()
    out_drop = Path(args.output_dropped_csv).resolve()
    out_kept.parent.mkdir(parents=True, exist_ok=True)
    out_drop.parent.mkdir(parents=True, exist_ok=True)

    universe = load_universe_txt(in_path)
    universe["keyword"] = universe["company_name"].apply(normalize_keyword)
    universe["tradability_rank"] = universe["company_name"].apply(tradability_rank)
    universe["company_name_lc"] = universe["company_name"].astype(str).str.lower()

    drop_re = compile_drop_regex(args.drop_keywords)
    keyword_mask = universe["company_name_lc"].str.contains(drop_re, regex=True, na=False)
    dropped_keyword = universe[keyword_mask].copy()
    dropped_keyword["dropped_for_ticker"] = ""
    dropped_keyword["reason"] = "keyword_filter"
    universe = universe[~keyword_mask].copy()

    suffix_mask = find_suffix_variant_mask(universe)
    dropped_suffix = universe[suffix_mask].copy()
    dropped_suffix["dropped_for_ticker"] = ""
    dropped_suffix["reason"] = "ticker_suffix_variant"
    universe = universe[~suffix_mask].copy()

    groups = universe.groupby("keyword", sort=False)
    dup_keys = [k for k, g in groups if len(g) > 1]
    if args.max_duplicate_groups > 0:
        dup_keys = dup_keys[: args.max_duplicate_groups]

    print(f"Loaded tickers: {len(universe) + len(dropped_keyword) + len(dropped_suffix)}")
    print(f"Dropped by keyword filter: {len(dropped_keyword)}")
    print(f"Dropped by ticker suffix rule: {len(dropped_suffix)}")
    print(f"Duplicate keyword groups: {len(dup_keys)}")

    kept_rows: list[dict] = []
    dropped_rows: list[dict] = (
        pd.concat([dropped_keyword, dropped_suffix], ignore_index=True).to_dict(orient="records")
    )

    # First pass: keep all unique-keyword rows automatically.
    for key, grp in groups:
        if len(grp) == 1:
            rec = grp.iloc[0].to_dict()
            rec["reason"] = "unique_keyword"
            kept_rows.append(rec)

    # Second pass: resolve duplicate-keyword groups with keyword ranking.
    for i, key in enumerate(dup_keys, 1):
        grp = groups.get_group(key).copy()
        print(f"[{i}/{len(dup_keys)}] keyword='{key}' candidates={len(grp)}")

        # Prefer likely tradable symbols first.
        grp["name_len"] = grp["company_name"].astype(str).str.len()
        grp = grp.sort_values(
            ["tradability_rank", "name_len", "ticker"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

        # Keep top-ranked candidate.
        best = grp.iloc[0]
        best_tk = best["ticker"]
        kept_rows.append(
            {
                **best.to_dict(),
                "reason": "best_in_duplicate_keyword_group",
            }
        )

        # Drop the rest.
        for idx, row in grp.iterrows():
            if idx == 0:
                continue
            dropped_rows.append(
                {
                    **row.to_dict(),
                    "dropped_for_ticker": best_tk,
                    "reason": "duplicate_keyword_group",
                }
            )

    kept_df = pd.DataFrame(kept_rows)
    kept_df = kept_df.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    # Write kept txt in compatible format.
    with open(out_kept, "w", encoding="utf-8") as f:
        f.write("# ticker\tcompany_name\texchange\n")
        for _, r in kept_df.iterrows():
            f.write(f"{r['ticker']}\t{r['company_name']}\t{r['exchange']}\n")

    dropped_df = pd.DataFrame(dropped_rows)
    if "company_name_lc" in dropped_df.columns:
        dropped_df = dropped_df.drop(columns=["company_name_lc"])
    dropped_df.to_csv(out_drop, index=False)

    print("\nDone.")
    print(f"Kept tickers:   {len(kept_df)}")
    print(f"Dropped tickers:{len(dropped_df)}")
    print(f"Saved kept TXT: {out_kept}")
    print(f"Saved drop CSV: {out_drop}")


if __name__ == "__main__":
    main()

